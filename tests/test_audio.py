"""Decoding on the file's own clock: where each stream's audio goes.

The files are written here with PyAV, the way recorders write them: Matroska
keeps timestamps to the millisecond, a stream can begin after the others, a
paused recorder leaves a gap in the timestamps, and two recordings joined end
to end make them jump back. What used to happen to all of them is the same:
the audio was laid end to end from 0:00, and everything after a gap or in a
late stream moved earlier.
"""

from __future__ import annotations

import tempfile
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tests  # noqa: F401  (puts app/ on sys.path)

import audio
from audio import (
    SAMPLE_RATE,
    TIMING_TOLERANCE,
    AudioFileError,
    Timing,
    decode_track_audio,
    iter_stream_power,
)

RATE = 48000


def noise(seconds: float, seed: int, channels: int = 1, rate: int = RATE):
    """Steady noise, about -16 dBFS: loud enough that silence beside it is plain."""
    shape = (int(seconds * rate), channels)
    return (0.15 * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32)


def write_mkv(path: Path, streams, rate: int = RATE, frame: int = 1024,
              fmt: str = "matroska", codec: str = "pcm_s16le") -> Path:
    """One stream per entry of `streams`, each a list of (start s, samples) blocks.

    `samples` has a row per sample and a column per channel. Every block is
    cut into `frame`-sample frames stamped from where the block starts, so a
    gap between blocks is a gap in the timestamps, and a block that starts
    late is a stream that began late.
    """
    import av

    frames = []
    with av.open(str(path), "w", format=fmt) as out:
        for blocks in streams:
            channels = blocks[0][1].shape[1]
            stream = out.add_stream(codec, rate=rate,
                                    layout="mono" if channels == 1 else "stereo")
            for start, samples in blocks:
                for k in range(0, len(samples), frame):
                    chunk = np.ascontiguousarray(samples[k:k + frame], np.float32).reshape(1, -1)
                    f = av.AudioFrame.from_ndarray(chunk, format="flt",
                                                   layout=stream.layout.name)
                    f.sample_rate = rate
                    f.time_base = Fraction(1, rate)
                    f.pts = round(start * rate) + k
                    frames.append((f.pts, stream.index, stream, f))
        frames.sort(key=lambda item: item[:2])
        for _pts, _index, stream, f in frames:
            for packet in stream.encode(f):
                out.mux(packet)
        for stream in out.streams.audio:
            for packet in stream.encode(None):
                out.mux(packet)
    return path


def loud_seconds(samples, rate: int = SAMPLE_RATE) -> list[int]:
    """The whole seconds of `samples` that hold sound."""
    seconds = len(samples) // rate
    power = np.square(np.asarray(samples[:seconds * rate], dtype=np.float64))
    return [int(s) for s in np.flatnonzero(power.reshape(seconds, rate).mean(axis=1) > 1e-4)]


def power_of(path: Path, stream_index=None, rate: int = 8000, timing=None):
    return np.concatenate(list(iter_stream_power(path, stream_index, rate, timing)))


class _Files(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def mkv(self, name: str, streams, **kwargs) -> Path:
        return write_mkv(self.folder / name, streams, **kwargs)


class FollowsTimestamps(_Files):
    def test_a_gap_in_the_timestamps_stays_a_gap(self):
        # Frames for seconds 0, 1, 4 and 5: a recorder paused for two seconds.
        path = self.mkv("gap.mkv", [[(0, noise(2, 1)), (4, noise(2, 2))]], frame=RATE)
        timing = Timing()
        decoded = decode_track_audio(path, None, None, timing)
        self.assertEqual(len(decoded), 6 * SAMPLE_RATE)
        self.assertEqual(loud_seconds(decoded), [0, 1, 4, 5])
        self.assertEqual(timing, Timing())          # a gap is no jump back
        power = power_of(path)
        self.assertEqual(len(power), 6 * 8000)
        self.assertEqual(loud_seconds(np.sqrt(power), 8000), [0, 1, 4, 5])

    def test_a_stream_that_began_late_keeps_its_leading_silence(self):
        # Stream 2 began a minute after stream 1, in the same file.
        path = self.mkv("late.mkv", [[(0, noise(90, 3))], [(60, noise(30, 4))]])
        first = decode_track_audio(path, 0, None)
        second = decode_track_audio(path, 1, None)
        self.assertEqual(len(first), len(second))
        self.assertEqual(loud_seconds(first), list(range(90)))
        self.assertEqual(loud_seconds(second), list(range(60, 90)))
        self.assertEqual(loud_seconds(np.sqrt(power_of(path, 1)), 8000), list(range(60, 90)))

    def test_a_file_whose_clock_starts_at_ten_seconds_starts_at_zero(self):
        # The container's own start is 0:00: no ten seconds of silence first.
        path = self.mkv("ten.mkv", [[(10, noise(5, 5))]])
        decoded = decode_track_audio(path, None, None)
        self.assertEqual(len(decoded), 5 * SAMPLE_RATE)
        self.assertEqual(loud_seconds(decoded), [0, 1, 2, 3, 4])

    def test_one_channel_keeps_the_same_time_as_the_whole_stream(self):
        stereo = noise(2, 6, channels=2)
        stereo[:, 1] = 0                            # only the left channel talks
        later = noise(2, 7, channels=2)
        later[:, 0] = 0                             # then only the right
        path = self.mkv("pair.mkv", [[(0, stereo), (5, later)]])
        whole = decode_track_audio(path, None, None)
        left = decode_track_audio(path, None, 0)
        right = decode_track_audio(path, None, 1)
        self.assertEqual(len(whole), len(left))
        self.assertEqual(len(whole), len(right))
        self.assertEqual(loud_seconds(whole), [0, 1, 5, 6])
        self.assertEqual(loud_seconds(left), [0, 1])
        self.assertEqual(loud_seconds(right), [5, 6])

    def test_frame_size_and_rate_move_nothing_beyond_rounding(self):
        # The same session cut into frames of three sizes and read at two
        # rates: where the sound starts again after the gap agrees to within
        # the timing tolerance, whatever the cut.
        found = []
        for frame in (480, 1024, 4800):
            path = self.mkv(f"cut{frame}.mkv", [[(0, noise(3, 8)), (7.3, noise(2, 9))]],
                            frame=frame)
            for rate in (8000, SAMPLE_RATE):
                power = (power_of(path, rate=rate) if rate == 8000
                         else np.square(decode_track_audio(path, None, None)))
                back = int(np.flatnonzero(power[3 * rate:] > 1e-4)[0]) + 3 * rate
                found.append(back / rate)
                self.assertAlmostEqual(len(power) / rate, 9.3, delta=1 / rate)
        self.assertLessEqual(max(found) - min(found), TIMING_TOLERANCE)
        self.assertAlmostEqual(np.mean(found), 7.3, delta=TIMING_TOLERANCE)

    def test_an_ordinary_wav_decodes_exactly_as_before(self):
        # Zero-based, no gaps: sample for sample what laying the resampled
        # audio end to end gave, at 16 kHz mono and as power at 8 kHz.
        import av
        from av.audio.resampler import AudioResampler

        path = self.folder / "plain.wav"
        samples = noise(7.5, 10, channels=2, rate=44100)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(44100)
            out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

        def laid_end_to_end(**kwargs):
            chunks = []
            with av.open(str(path)) as container:
                resampler = None
                for frame in container.decode(audio=0):
                    resampler = resampler or AudioResampler(format="flt", **kwargs)
                    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
            return np.concatenate(chunks)

        np.testing.assert_array_equal(decode_track_audio(path, None, None),
                                      laid_end_to_end(layout="mono", rate=SAMPLE_RATE))
        both = laid_end_to_end(layout="stereo", rate=8000).astype(np.float64).reshape(-1, 2)
        np.testing.assert_array_equal(power_of(path), np.sum(both * both, axis=1))

    def test_joined_recordings_keep_every_sample_and_say_where(self):
        # Two MPEG-TS recordings joined end to end: the second's clock starts
        # again from zero. All of both is kept, the second after the first.
        first, second = self.folder / "one.ts", self.folder / "two.ts"
        write_mkv(first, [[(0, noise(5, 11))]], frame=1152, fmt="mpegts", codec="mp2")
        write_mkv(second, [[(0, noise(3, 12))]], frame=1152, fmt="mpegts", codec="mp2")
        joined = self.folder / "joined.ts"
        joined.write_bytes(first.read_bytes() + second.read_bytes())
        timing = Timing()
        decoded = decode_track_audio(joined, None, None, timing)
        self.assertAlmostEqual(len(decoded) / SAMPLE_RATE, 8.0, delta=0.05)
        self.assertEqual(loud_seconds(decoded), list(range(8)))
        ((at, back),) = timing.jumps
        self.assertAlmostEqual(at, 5.0, delta=0.05)
        self.assertAlmostEqual(back, 5.0, delta=0.05)
        self.assertIn("jump back at 0:05", timing.summary())

    def test_a_long_gap_is_handed_out_a_little_at_a_time(self):
        # An hour between two bursts: never an hour of samples in one piece.
        path = self.mkv("hour.mkv", [[(0, noise(1, 13)), (3601, noise(1, 14))]])
        chunks = list(iter_stream_power(path, None, 8000))
        self.assertLessEqual(max(len(c) for c in chunks), audio.SILENCE_CHUNK_SECONDS * 8000)
        self.assertEqual(sum(len(c) for c in chunks), 3602 * 8000)

    def test_a_timestamp_past_the_limit_is_an_error_not_an_allocation(self):
        path = self.mkv("far.mkv", [[(0, noise(1, 15)), (120, noise(1, 16))]])
        with patch("audio.MAX_TIMELINE_SECONDS", 60), \
                patch("numpy.zeros", wraps=np.zeros) as zeros:
            with self.assertRaises(AudioFileError) as caught:
                decode_track_audio(path, None, None)
            with self.assertRaises(AudioFileError):
                list(iter_stream_power(path, None, 8000))
        self.assertIn("far.mkv", str(caught.exception))
        self.assertIn("timing looks damaged", str(caught.exception))
        self.assertTrue(all(np.prod(c.args[0]) <= 2 * 60 * SAMPLE_RATE
                            for c in zeros.call_args_list))

    def test_a_missing_stream_is_an_audio_file_error(self):
        path = self.mkv("one.mkv", [[(0, noise(1, 17))]])
        with self.assertRaises(AudioFileError) as caught:
            decode_track_audio(path, 3, None)
        self.assertIn("has no audio stream 4", str(caught.exception))


def _blocks(*runs):
    """(position, samples) blocks of `length` ones, as _timed_blocks gives them."""
    return [(position, np.ones((length, 1), np.float32)) for position, length in runs]


class Placement(unittest.TestCase):
    """The decision for each run of audio, on positions in samples at 1 kHz."""

    def place(self, *runs):
        timing = Timing()
        placed = list(audio._placed(_blocks(*runs), 1000, Path("f.wav"), timing))
        return [(silence, len(samples)) for silence, samples in placed], timing

    def test_within_the_tolerance_it_follows_on(self):
        tolerance = round(TIMING_TOLERANCE * 1000)
        placed, timing = self.place((0, 100), (100 + tolerance, 50), (150 - tolerance, 50))
        self.assertEqual(placed, [(0, 100), (0, 50), (0, 50)])
        self.assertEqual(timing, Timing())

    def test_later_is_a_gap_filled_with_silence(self):
        placed, timing = self.place((0, 100), (500, 50))
        self.assertEqual(placed, [(0, 100), (400, 50)])
        self.assertEqual(timing.jumps, [])

    def test_earlier_plays_on_and_is_noted(self):
        # The clock went back 0.8 s at 1.0 s: read as restarted there, so
        # the 0.6 s gap after it is still a 0.6 s gap.
        placed, timing = self.place((0, 1000), (200, 500), (1300, 100))
        self.assertEqual(placed, [(0, 1000), (0, 500), (600, 100)])
        self.assertEqual(timing.jumps, [(1.0, 0.8)])

    def test_before_zero_at_the_start_plays_from_zero_and_is_noted(self):
        # Normalised against the file's start already: what is left before
        # it is kept, never dropped, and said.
        placed, timing = self.place((-300, 1000))
        self.assertEqual(placed, [(0, 1000)])
        self.assertEqual(timing.jumps, [(0.0, 0.3)])

    def test_a_block_with_no_position_follows_on(self):
        placed, timing = self.place((None, 100), (None, 100), (900, 10))
        self.assertEqual(placed, [(0, 100), (0, 100), (700, 10)])
        self.assertEqual(timing, Timing())

    def test_an_absurd_position_stops_before_making_its_silence(self):
        limit = audio.MAX_TIMELINE_SECONDS * 1000
        at_limit, _timing = self.place((0, 10), (limit, 10))
        self.assertEqual(at_limit, [(0, 10), (limit - 10, 10)])
        placed = audio._placed(_blocks((0, 10), (limit + 1, 10)), 1000,
                               Path("f.wav"), Timing())
        self.assertEqual(next(placed)[0], 0)
        with self.assertRaises(AudioFileError):
            next(placed)


class _Packet:
    def __init__(self, frames=(), error=None):
        self.frames, self.error = list(frames), error

    def decode(self):
        if self.error is not None:
            raise self.error
        return self.frames


def _frame(pts, samples: int = 480, rate: int = RATE):
    import av

    frame = av.AudioFrame.from_ndarray(np.full((1, samples), 0.1, np.float32),
                                       format="flt", layout="mono")
    frame.sample_rate = rate
    frame.pts = pts
    if pts is not None:
        frame.time_base = Fraction(1, rate)
    return frame


class _Container:
    """Just enough of an open container for _frames and _timed_blocks."""

    def __init__(self, packets, start_time=0):
        self.packets, self.start_time = packets, start_time

    def demux(self, *_streams):
        return iter(self.packets)


class Frames(unittest.TestCase):
    def test_a_damaged_packet_is_skipped_and_counted(self):
        import av

        damage = av.error.InvalidDataError(1094995529, "Invalid data")
        container = _Container([_Packet([_frame(0)]), _Packet(error=damage),
                                _Packet([_frame(5280)])])
        timing = Timing()
        frames = list(audio._frames(container, None, Path("f.mka"), timing))
        self.assertEqual([f.pts for f in frames], [0, 5280])
        self.assertEqual(timing.damaged, 1)
        self.assertIn("1 damaged packet skipped", timing.summary())
        # What came after it is still placed by its timestamp: 0.1 s of
        # silence where the damaged packet was.
        stream = SimpleNamespace(time_base=Fraction(1, RATE))
        blocks = audio._timed_blocks(_Container(container.packets), stream, Path("f.mka"),
                                     RATE, "mono", None, 0, Timing())
        placed = list(audio._placed(blocks, RATE, Path("f.mka"), Timing()))
        self.assertEqual([(silence, len(s)) for silence, s in placed],
                         [(0, 480), (4800, 480)])

    def test_a_file_where_nothing_decodes_is_an_audio_file_error(self):
        import av

        damage = av.error.InvalidDataError(1094995529, "Invalid data")
        container = _Container([_Packet(error=damage), _Packet(error=damage)])
        with self.assertRaises(AudioFileError) as caught:
            list(audio._frames(container, None, Path("broken.mka"), Timing()))
        self.assertIn("broken.mka: nothing in it would decode", str(caught.exception))

    def test_frames_without_timestamps_follow_on_until_one_has_one(self):
        # Two frames with no timestamp, then the clock says 0.5 s: the first
        # two follow on from 0:00, and the rest goes where its clock says.
        packets = [_Packet([_frame(None)]), _Packet([_frame(None)]),
                   _Packet([_frame(RATE // 2)]), _Packet([_frame(RATE // 2 + 480)])]
        self.assertEqual(self.layout(packets),
                         [(0, 480), (480, 480), (RATE // 2, 480), (RATE // 2 + 480, 480)])
        # And in a file with no 0:00 of its own, the first timestamp starts
        # the clock where the audio before it ended.
        with patch("audio._source_origin", return_value=None):
            self.assertEqual(self.layout(packets),
                             [(0, 480), (480, 480), (960, 480), (1440, 480)])

    def layout(self, packets):
        """(where it starts, length) of each block, in samples, once placed."""
        stream = SimpleNamespace(time_base=Fraction(1, RATE))
        blocks = audio._timed_blocks(_Container(packets), stream, Path("f.mka"),
                                     RATE, "mono", None, 0, Timing())
        layout, end = [], 0
        for silence, samples in audio._placed(blocks, RATE, Path("f.mka"), Timing()):
            end += silence
            layout.append((end, len(samples)))
            end += len(samples)
        return layout


class Summary(unittest.TestCase):
    def test_nothing_to_say(self):
        self.assertEqual(Timing().summary(), "")

    def test_jumps_and_damage(self):
        text = Timing(jumps=[(3723.4, 12.3), (5000.0, 2.0)], damaged=3).summary()
        self.assertIn("jump back in 2 places, first at 1:02:03", text)
        self.assertIn("off by up to 14.3 s", text)
        self.assertIn("3 damaged packets skipped", text)


if __name__ == "__main__":
    unittest.main()
