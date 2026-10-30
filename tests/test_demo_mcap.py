from pathlib import Path
import runpy

from flight_lens.reader import load_flight, read_frame

write_demo = runpy.run_path(str(Path(__file__).parents[1] / 'examples' / 'create_demo_mcap.py'))['write_demo']


def test_demo_generator_is_deterministic_and_loadable(tmp_path):
    first = write_demo(tmp_path / 'first.mcap')
    second = write_demo(tmp_path / 'second.mcap')
    assert first.read_bytes() == second.read_bytes()

    data = load_flight(first)
    episode = data.episodes[0]
    assert episode.summary.outcome == 'Success · target_hit'
    assert len(episode.tracking_samples) == 2
    assert episode.tracking_samples[0].bbox == (2, 1, 4, 3)
    assert episode.tracking_samples[0].dx_px == 0.0
    assert episode.tracking_samples[0].dy_px == -0.5
    assert episode.tracking_samples[0].frame is not None
    assert read_frame(data, episode.tracking_samples[0].frame).rgb[:3] == bytes((220, 40, 40))
    assert episode.ttc_filtered.values[:, 0].tolist() == [1.5, 1.0]


def test_committed_demo_fixture_is_available(tmp_path):
    path = Path(__file__).parents[1] / 'examples' / 'demo.mcap'
    local_copy = tmp_path / path.name
    local_copy.write_bytes(path.read_bytes())
    data = load_flight(local_copy)
    assert data.episodes[0].tracking_samples
