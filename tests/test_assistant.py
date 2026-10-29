import json
import sys
from types import SimpleNamespace

import pytest

from flight_lens.assistant import (ASSISTANT_GPU_LAYERS, MAX_CONTEXT_CHARS, LlamaCppBackend,
                                   McapAnalysisTools, McapAssistant, validate_chart)
from flight_lens.reader import load_flight


def test_mcap_tools_return_bounded_summary_and_series(flight_file):
    data = load_flight(flight_file)
    tools = McapAnalysisTools(data, data.episodes[0])

    summary = tools.call('recording_summary', {})
    series = tools.call('query_series', {'name': 'velocity_x', 'max_points': 1})
    aliases = tools.call('query_series', {'name': 'x', 'max_points': 1})

    assert summary['outcome'].startswith('Mission fault')
    assert len(series['times']) == 1
    assert aliases['name'] == 'x'
    with pytest.raises(ValueError, match='unknown'):
        tools.call('query_series', {'name': 'raw_payload'})

    queried = tools.call('query_data', {'fields': [{'topic': 'analysis/app/msp', 'path': 'attitude.yaw_rad'}],
                                        'start_time_s': .1, 'end_time_s': .12, 'max_points': 10})
    assert queried['fields'][0]['times'] == [.1, .12]
    assert {'topic': 'analysis/app/msp', 'path': 'attitude.yaw_rad'} in tools.call('list_series', {})


def test_sql_tool_scopes_episode_and_rejects_unsafe_queries(flight_file):
    data = load_flight(flight_file)
    tools = McapAnalysisTools(data, data.episodes[0])
    result = tools.call('query_sql', {'sql': 'SELECT time_s, value FROM selected_telemetry ORDER BY time_s'})
    assert result['columns'] == ['time_s', 'value']
    assert result['row_count'] <= 500
    assert result['sql'].endswith('LIMIT 500')
    with pytest.raises(ValueError, match='read-only'):
        tools.call('query_sql', {'sql': 'DELETE FROM selected_telemetry'})
    with pytest.raises(ValueError, match='cannot exceed'):
        tools.call('query_sql', {'sql': 'SELECT * FROM selected_events LIMIT 501'})


def test_generate_sql_validates_without_executing(flight_file):
    data = load_flight(flight_file)

    class Backend:
        def complete(self, messages, tools):
            assert tools == []
            return {'choices': [{'message': {'content':
                '```sql\nSELECT time_s, value FROM selected_telemetry LIMIT 2\n```'}}]}

    result = McapAssistant(Backend()).generate_sql(data, data.episodes[0], 'Show two telemetry samples.')
    assert result == {'sql': 'SELECT time_s, value FROM selected_telemetry LIMIT 2', 'valid': True}


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {'choices': [{'message': {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': 'call-1', 'function': {
                    'name': 'recording_summary', 'arguments': '{}'}}],
            }}]}
        return {'choices': [{'message': {'role': 'assistant', 'content':
            'The recording ended normally.\n```json\n' + json.dumps({
                'type': 'line_chart', 'title': 'Altitude', 'x_label': 'Time', 'x_unit': 's',
                'x': [0., 1.], 'series': [{'name': 'Altitude', 'unit': 'm', 'values': [1., 2.]}]
            }) + '\n```'}}]}


class XmlToolBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {'choices': [{'message': {'role': 'assistant', 'content':
                '<tool_call>\n<function=recording_summary>\n</function>\n</tool_call>'}}]}
        return {'choices': [{'message': {'role': 'assistant', 'content': 'Summary complete.'}}]}


def test_assistant_executes_tools_and_returns_validated_chart(flight_file):
    data = load_flight(flight_file)
    result = McapAssistant(FakeBackend()).answer(data, data.episodes[0], 'Summarize and chart it.')

    assert 'ended normally' in result.text
    assert result.chart['series'][0]['values'] == [1., 2.]
    assert result.trace[0]['name'] == 'recording_summary'
    assert result.trace[0]['arguments'] == {}


def test_assistant_executes_xml_tool_markup_without_showing_it(flight_file):
    data = load_flight(flight_file)
    backend = XmlToolBackend()
    result = McapAssistant(backend).answer(data, data.episodes[0], 'Summarize this flight.')

    assert backend.calls == 2
    assert result.text == 'Summary complete.'
    assert '<tool_call>' not in result.text


def test_chart_validation_rejects_mismatched_series():
    with pytest.raises(ValueError, match='match x values'):
        validate_chart({'type': 'line_chart', 'x': [0., 1.], 'series': [{'values': [1.]}]})


def test_assistant_bounds_accumulated_tool_context(flight_file, monkeypatch):
    data = load_flight(flight_file)
    payload = {'name': 'x', 'times': list(range(2000)), 'values': [[float(i)] for i in range(2000)]}
    monkeypatch.setattr(McapAnalysisTools, 'call', lambda self, name, arguments: payload)

    class Backend:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            assert len(json.dumps(messages)) <= MAX_CONTEXT_CHARS
            self.calls += 1
            if self.calls == 1:
                return {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'query', 'function': {'name': 'query_series', 'arguments': '{"name":"x"}'}}]}}]}
            return {'choices': [{'message': {'role': 'assistant', 'content': 'Done.'}}]}

    result = McapAssistant(Backend()).answer(data, data.episodes[0], 'Query it.')
    assert result.text == 'Done.'


def test_llama_backend_requests_cuda_layer_offload(monkeypatch):
    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_chat_completion(self, **kwargs):
            return {'choices': [{'message': {'content': 'ok'}}]}

    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=FakeLlama))
    LlamaCppBackend('model.gguf').complete([], [])

    assert captured['n_gpu_layers'] == ASSISTANT_GPU_LAYERS == -1
