"""Bounded MCAP analysis tools and an optional local llama.cpp assistant."""
import json
from dataclasses import dataclass
from math import isfinite
import re

import numpy as np

from .model import Episode, FlightData, Series
from .sqlite_index import list_fields, query_fields


MAX_QUERY_POINTS = 80
MAX_EVENTS = 30
MAX_RESPONSE_CHARS = 12000
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CONTENT_CHARS = 1800
MAX_CONTEXT_CHARS = 7000
MAX_OUTPUT_TOKENS = 512
ASSISTANT_GPU_LAYERS = -1
SERIES_NAMES = ('x', 'y', 'z', 'slant', 'velocity_x', 'velocity_y', 'velocity_z',
                'slant_velocity', 'altitude', 'barometer_altitude', 'vertical_speed',
                'attitude_roll', 'attitude_pitch', 'attitude_yaw', 'thrust')
CHART_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
XML_TOOL_RE = re.compile(r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>', re.DOTALL)
XML_PARAMETER_RE = re.compile(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', re.DOTALL)


class AssistantCancelled(Exception):
    pass


@dataclass(frozen=True)
class AssistantResult:
    text: str
    chart: dict | None = None


def _bounded_series(series: Series, max_points=MAX_QUERY_POINTS):
    max_points = min(MAX_QUERY_POINTS, max(1, int(max_points)))
    if len(series.times) <= max_points:
        indexes = np.arange(len(series.times))
    else:
        indexes = np.linspace(0, len(series.times) - 1, max_points, dtype=int)
        indexes = np.unique(indexes)
    return dict(times=[float(series.times[i]) for i in indexes],
                values=[[float(value) for value in series.values[i]] for i in indexes])


def _series_map(episode: Episode):
    result = {
        'trajectory_x': Series(episode.trajectory.times, episode.trajectory.values[:, [0]]),
        'trajectory_z': Series(episode.trajectory.times, episode.trajectory.values[:, [2]]),
        'altitude': Series(episode.trajectory.times, episode.trajectory.values[:, [2]]),
        'barometer_altitude': episode.barometer,
        'vertical_speed': episode.barometer_velocity,
        'velocity_x': Series(episode.velocity.times, episode.velocity.values[:, [0]]),
        'velocity_y': Series(episode.velocity.times, episode.velocity.values[:, [1]]),
        'velocity_z': Series(episode.velocity.times, episode.velocity.values[:, [2]]),
        'attitude_roll': Series(episode.attitude.times, episode.attitude.values[:, [0]]),
        'attitude_pitch': Series(episode.attitude.times, episode.attitude.values[:, [1]]),
        'attitude_yaw': Series(episode.attitude.times, episode.attitude.values[:, [2]]),
        'thrust': episode.thrust,
    }
    result.update(x=result['velocity_x'], y=result['velocity_y'], z=result['velocity_z'])
    if len(episode.velocity.times):
        values = np.hypot(episode.velocity.values[:, 0], episode.velocity.values[:, 2]).reshape((-1, 1))
        result['slant_velocity'] = Series(episode.velocity.times, values)
        result['slant'] = result['slant_velocity']
    return result


class McapAnalysisTools:
    def __init__(self, data: FlightData, episode: Episode):
        self.data = data
        self.episode = episode

    def call(self, name, arguments):
        if name == 'recording_summary':
            return self.recording_summary()
        if name == 'find_events':
            return self.find_events(arguments.get('severity'), arguments.get('limit', MAX_EVENTS))
        if name == 'query_series':
            return self.query_series(arguments.get('name'), arguments.get('start_time_s'),
                                     arguments.get('end_time_s'), arguments.get('max_points', MAX_QUERY_POINTS))
        if name == 'query_data':
            return self.query_data(arguments.get('fields'), arguments.get('start_time_s'),
                                   arguments.get('end_time_s'), arguments.get('max_points', MAX_QUERY_POINTS),
                                   arguments.get('aggregation', 'raw'))
        if name == 'list_series':
            return self.list_series()
        if name == 'context_at':
            return self.context_at(arguments.get('time_s'))
        raise ValueError(f'unknown analysis tool: {name}')

    def recording_summary(self):
        summary = self.episode.summary
        return dict(file=self.data.path.name, size_bytes=self.data.size, episode=self.episode.key[1],
                    duration_s=summary.duration_s, outcome=summary.outcome,
                    peak_altitude_m=summary.max_altitude_m, first_fault=summary.first_fault,
                    warnings=summary.warnings, errors=summary.errors, integrity=summary.integrity,
                    notices=list(self.episode.notices))

    def find_events(self, severity=None, limit=MAX_EVENTS):
        limit = min(MAX_EVENTS, max(1, int(limit)))
        events = [event for event in self.episode.events if severity is None or event.severity == severity]
        return [dict(time_s=event.time_s, end_s=event.end_s, severity=event.severity,
                     component=event.component, reason=event.reason, phase=event.phase)
                for event in events[:limit]]

    def query_series(self, name, start_time_s=None, end_time_s=None, max_points=MAX_QUERY_POINTS):
        name = str(name).strip().lower().replace(' ', '_')
        samples = _series_map(self.episode).get(name)
        if samples is None:
            raise ValueError(f'unknown or unavailable series: {name}')
        start = -float('inf') if start_time_s is None else float(start_time_s)
        end = float('inf') if end_time_s is None else float(end_time_s)
        if not isfinite(start) and start != -float('inf') or not isfinite(end) and end != float('inf'):
            raise ValueError('time bounds must be finite')
        if start > end:
            raise ValueError('start_time_s must not exceed end_time_s')
        mask = (samples.times >= start) & (samples.times <= end)
        return dict(name=name, **_bounded_series(Series(samples.times[mask], samples.values[mask]), max_points))

    def context_at(self, time_s):
        time_s = float(time_s)
        if not isfinite(time_s):
            raise ValueError('time_s must be finite')
        return dict(time_s=time_s, context=dict(self.episode.context_at(time_s)))

    def query_data(self, fields, start_time_s=None, end_time_s=None, max_points=MAX_QUERY_POINTS, aggregation='raw'):
        if self.data.index_path is None:
            raise ValueError('recording index is unavailable')
        return dict(aggregation=aggregation, fields=query_fields(
            self.data.index_path, self.episode.key, fields, start_time_s, end_time_s,
            max_points, aggregation))

    def list_series(self):
        if self.data.index_path is None:
            raise ValueError('recording index is unavailable')
        return list_fields(self.data.index_path, self.episode.key)


TOOL_DEFINITIONS = [
    {'type': 'function', 'function': {'name': 'recording_summary', 'description': 'Return the selected flight summary.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'find_events', 'description': 'Find recorded flight events.', 'parameters': {'type': 'object', 'properties': {'severity': {'type': 'string', 'enum': ['info', 'warning', 'error']}, 'limit': {'type': 'integer', 'maximum': MAX_EVENTS}}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'query_series', 'description': 'Return bounded numeric telemetry samples. Names include x, z, slant, altitude, vertical_speed, attitude_roll, attitude_pitch, attitude_yaw, and thrust.', 'parameters': {'type': 'object', 'required': ['name'], 'properties': {'name': {'type': 'string', 'enum': SERIES_NAMES}, 'start_time_s': {'type': 'number'}, 'end_time_s': {'type': 'number'}, 'max_points': {'type': 'integer', 'maximum': MAX_QUERY_POINTS}}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'query_data', 'description': 'Query bounded numeric telemetry fields from the indexed recording. Use topic and JSON field path from the recording.', 'parameters': {'type': 'object', 'required': ['fields'], 'properties': {'fields': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': {'type': 'object', 'required': ['topic', 'path'], 'properties': {'topic': {'type': 'string'}, 'path': {'type': 'string'}, 'name': {'type': 'string'}}, 'additionalProperties': False}}, 'start_time_s': {'type': 'number'}, 'end_time_s': {'type': 'number'}, 'max_points': {'type': 'integer', 'maximum': 500}, 'aggregation': {'type': 'string', 'enum': ['raw', 'mean', 'min', 'max', 'rms']}}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'list_series', 'description': 'List numeric topic and JSON field paths available in the indexed episode.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'context_at', 'description': 'Return recorded context near a simulation time.', 'parameters': {'type': 'object', 'required': ['time_s'], 'properties': {'time_s': {'type': 'number'}}, 'additionalProperties': False}}},
]


SYSTEM_PROMPT = """You are FlightLens, a read-only flight-recording analyst.
Use the provided analysis tools instead of guessing. Never claim data that was not returned by a tool.
Use query_series for common aliases and query_data for indexed topic/field paths.
Keep answers concise and reference simulation times when useful.
If a graph helps, return one JSON object in a fenced json block with this shape:
{"type":"line_chart","title":"...","x_label":"Simulation time","x_unit":"s","x":[...],"series":[{"name":"...","unit":"...","values":[...]}]}
Only create charts from tool data. Do not return Python or plotting code.
"""


class LlamaCppBackend:
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.model = None

    def _load(self):
        if self.model is None:
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise RuntimeError('Install the assistant extra: uv sync --extra assistant') from error
            self.model = Llama(model_path=self.model_path, n_ctx=8192,
                               n_gpu_layers=ASSISTANT_GPU_LAYERS, verbose=False)

    def complete(self, messages, tools):
        self._load()
        return self.model.create_chat_completion(messages=messages, tools=tools, temperature=.2,
                                                 max_tokens=MAX_OUTPUT_TOKENS)


def validate_chart(chart):
    if not isinstance(chart, dict) or chart.get('type') != 'line_chart':
        raise ValueError('unsupported chart type')
    x = chart.get('x')
    series = chart.get('series')
    if not isinstance(x, list) or not x or len(x) > MAX_QUERY_POINTS or not isinstance(series, list) or not series:
        raise ValueError('invalid chart bounds')
    if not all(isinstance(value, (int, float)) and isfinite(value) for value in x):
        raise ValueError('chart x values must be finite numbers')
    checked = []
    for item in series[:8]:
        values = item.get('values') if isinstance(item, dict) else None
        if not isinstance(values, list) or len(values) != len(x) or not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
            raise ValueError('chart series values must match x values')
        checked.append(dict(name=str(item.get('name', 'Series')), unit=str(item.get('unit', '')), values=values))
    return dict(type='line_chart', title=str(chart.get('title', 'Assistant result')), x_label=str(chart.get('x_label', 'Time')),
                x_unit=str(chart.get('x_unit', 's')), x=x, series=checked)


def extract_chart(text):
    match = CHART_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return validate_chart(json.loads(match.group(1)))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def _tool_content(result):
    content = json.dumps(result)
    if len(content) <= MAX_TOOL_CONTENT_CHARS:
        return content
    if isinstance(result, dict) and isinstance(result.get('times'), list) and isinstance(result.get('values'), list):
        count = max(2, min(len(result['times']), MAX_QUERY_POINTS // 2))
        reduced = dict(result, times=result['times'][:count], values=result['values'][:count], truncated=True)
        content = json.dumps(reduced)
    return content[:MAX_TOOL_CONTENT_CHARS] + '\n[tool result truncated]'


def _trim_context(messages):
    while len(json.dumps(messages)) > MAX_CONTEXT_CHARS:
        removed = False
        for index in range(2, len(messages)):
            message = messages[index]
            if message.get('role') == 'assistant' and message.get('tool_calls'):
                end = index + 1
                while end < len(messages) and messages[end].get('role') == 'tool':
                    end += 1
                del messages[index:end]
                removed = True
                break
        if not removed:
            break
def parse_xml_tool_calls(content):
    calls = []
    for index, match in enumerate(XML_TOOL_RE.finditer(content or '')):
        arguments = {}
        for name, raw_value in XML_PARAMETER_RE.findall(match.group(2)):
            value = raw_value.strip()
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            arguments[name.strip()] = value
        calls.append({'id': f'xml-call-{index}', 'function': {
            'name': match.group(1).strip(), 'arguments': json.dumps(arguments)}})
    return calls


class McapAssistant:
    def __init__(self, backend):
        self.backend = backend

    def answer(self, data, episode, prompt, cancel=lambda: False, progress=lambda message: None):
        tools = McapAnalysisTools(data, episode)
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
        for _ in range(MAX_TOOL_ROUNDS):
            if cancel():
                raise AssistantCancelled()
            progress('Asking local flight assistant…')
            _trim_context(messages)
            response = self.backend.complete(messages, TOOL_DEFINITIONS)
            message = response['choices'][0]['message']
            tool_calls = message.get('tool_calls') or parse_xml_tool_calls(message.get('content', ''))
            messages.append(message)
            if not tool_calls:
                text = str(message.get('content') or '').strip()
                return AssistantResult(text[:MAX_RESPONSE_CHARS], extract_chart(text))
            for call in tool_calls:
                if cancel():
                    raise AssistantCancelled()
                function = call['function']
                result = tools.call(function['name'], json.loads(function.get('arguments') or '{}'))
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', function['name']),
                                 'name': function['name'], 'content': _tool_content(result)})
        raise RuntimeError('assistant exceeded the tool-call limit')
