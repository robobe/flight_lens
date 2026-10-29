# Assistant SQL query and ERD viewer

## Goal

Allow the FlightLens Assistant tab to translate prompts into safe SQL against
the selected episode, show the SQL and bounded data as a table or graph, and
open a built-in ERD for query help.

## Implementation steps

1. Add selected-episode SQL views and a read-only bounded `query_sql` tool.
2. Separate LLM SQL generation/validation from SQL execution so users can
   inspect a query before running it.
3. Extend the Assistant tab with a custom prompt editor, SQL preview, result
   table, table/graph toggle, SQL examples, and an ERD dialog.
4. Add backend and Qt regression tests.

## Interface changes

The assistant tool catalog gains `query_sql`. No MCAP, recorder, or flight
runtime schema changes are required.

## Verification

Run the FlightLens unit and Qt test suite and manually exercise SQL, graph,
episode scoping, and ERD behavior with a real recording.

## Status

Implemented; tests passed

## Implementation commit

`44dc931`
