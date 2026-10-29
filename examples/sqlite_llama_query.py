"""Interactive llama.cpp-to-SQL example for a small SQLite telemetry table."""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "flight_lens.yaml"
DEFAULT_DATABASE = Path("sqlite_llama_demo.db")
ROW_COUNT = 1000
TIME_STEP_S = 0.1
MAX_QUERY_ROWS = 50
DISPLAY_ROWS = 20
EXAMPLE_PROMPTS = (
    "Show the first 10 samples.",
    "What is the average value between 20 and 40 seconds?",
    "Find the maximum value and its time.",
    "Return samples where value is above 0.8, ordered by time.",
)
SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
SQL_START_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)


def load_model_path(config_path: Path, override: str | None) -> str:
    if override:
        return override
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SystemExit(f"Cannot read config {config_path}: {error}") from error
    model = config.get("assistant-model")
    if not isinstance(model, str) or not model.strip():
        raise SystemExit("Set assistant-model in the config or pass --assistant-model.")
    return model


def create_demo_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE IF EXISTS samples")
    connection.execute("CREATE TABLE samples (time REAL NOT NULL, value REAL NOT NULL)")
    rows = []
    for index in range(ROW_COUNT):
        time_s = index * TIME_STEP_S
        value = math.sin(time_s / 5.0) + 0.15 * math.cos(time_s / 1.7)
        rows.append((time_s, value))
    connection.executemany("INSERT INTO samples(time, value) VALUES (?, ?)", rows)
    connection.commit()
    return connection


def extract_sql(response: str) -> str:
    match = SQL_FENCE_RE.search(response or "")
    sql = match.group(1) if match else response
    sql = sql.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql


def _authorizer(action, first, second, database, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if first == "samples" and (second is None or second in {"time", "value"}) else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def validate_sql(connection: sqlite3.Connection, sql: str) -> str:
    if not SQL_START_RE.match(sql) or ";" in sql or "--" in sql or "/*" in sql:
        raise ValueError("Only one plain SELECT statement is allowed.")
    limit = LIMIT_RE.search(sql)
    if limit is None:
        sql = f"{sql} LIMIT {MAX_QUERY_ROWS}"
    elif int(limit.group(1)) > MAX_QUERY_ROWS:
        raise ValueError(f"LIMIT must be {MAX_QUERY_ROWS} or smaller.")
    connection.set_authorizer(_authorizer)
    try:
        connection.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"SQL rejected: {error}") from error
    finally:
        connection.set_authorizer(None)
    return sql


def query_model(model, prompt: str) -> str:
    system = (
        "Convert the user's request into exactly one SQLite SELECT query. "
        "The only table is samples(time REAL, value REAL). "
        "Use only time and value, never write data, and include LIMIT 50 or less. "
        "Return SQL only, without explanation."
    )
    result = model.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or choice.get("text") or ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned no SQL.")
    return extract_sql(content)


def print_results(sql: str, rows: list[tuple], description) -> None:
    print(f"\nSQL: {sql}")
    columns = [item[0] for item in description]
    print(" | ".join(columns))
    print("-+-".join("-" * max(5, len(column)) for column in columns))
    for row in rows[:DISPLAY_ROWS]:
        print(" | ".join(str(value) for value in row))
    if len(rows) > DISPLAY_ROWS:
        print(f"… ({len(rows) - DISPLAY_ROWS} more rows not displayed)")
    print(f"Rows returned: {len(rows)}")


def run_repl(connection: sqlite3.Connection, model) -> None:
    print("SQLite llama.cpp query example")
    print("Demo table: samples(time, value), 1,000 rows")
    print("Example prompts:")
    for prompt in EXAMPLE_PROMPTS:
        print(f"  - {prompt}")
    print("Type exit or quit to stop.\n")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.lower() in {"exit", "quit"}:
            return
        if not prompt:
            continue
        try:
            sql = validate_sql(connection, query_model(model, prompt))
            cursor = connection.execute(sql)
            print_results(sql, cursor.fetchmany(MAX_QUERY_ROWS), cursor.description)
        except (ValueError, sqlite3.Error) as error:
            print(f"Rejected: {error}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="FlightLens YAML config")
    parser.add_argument("--assistant-model", help="Override the GGUF model path")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="Demo SQLite database path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_path = load_model_path(args.config, args.assistant_model)
    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise SystemExit("Install the assistant extra: uv sync --extra assistant") from error
    connection = create_demo_database(args.database)
    try:
        model = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
        run_repl(connection, model)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
