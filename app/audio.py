"""Probing and decoding of audio files, including multitrack containers.

A single file can hold several speakers two ways:
  - multiple audio streams in one container (OBS/REAPER/Craig exports), or
  - one stream with multiple channels (multichannel WAV/FLAC).
Both are expanded into one Track per speaker.
"""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import numpy as np

from models import Track, default_speaker_name

SAMPLE_RATE = 16000


def _stream_channels(stream) -> int:
    channels = getattr(stream, "channels", None)
    if channels:
        return channels
    try:
        return len(stream.layout.channels)
    except Exception:
        return 1


def probe_streams(path: Path) -> list[dict]:
    """Audio streams in a file: [{'channels': int, 'duration': float}, ...]."""
    import av

    with av.open(str(path)) as container:
        fallback = container.duration / av.time_base if container.duration else 0.0
        streams = []
        for stream in container.streams.audio:
            if stream.duration and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = fallback
            streams.append({"channels": _stream_channels(stream), "duration": duration})
        return streams


def channel_suffixes(count: int) -> list[str]:
    if count == 2:
        return [" L", " R"]
    return [f" Ch{n + 1}" for n in range(count)]


def tracks_for_file(path: Path, split_stereo: bool = False) -> list[Track]:
    """Expand a file into per-speaker Tracks.

    Multiple audio streams -> one track per stream. A stream with more than
    two channels -> one track per channel. Stereo is kept as one speaker
    unless split_stereo is set (right-click a track to split it).
    """
    base = default_speaker_name(path)
    try:
        streams = probe_streams(path)
    except Exception:
        streams = []
    if not streams:
        return [Track(path=path, speaker=base)]

    multi_stream = len(streams) > 1
    tracks = []
    for si, stream in enumerate(streams):
        name = f"{base} T{si + 1}" if multi_stream else base
        channels = stream["channels"]
        if channels > 2 or (channels == 2 and split_stereo):
            for ch, suffix in enumerate(channel_suffixes(channels)):
                tracks.append(Track(path=path, speaker=name + suffix,
                                    duration=stream["duration"],
                                    stream_index=si, channel=ch))
        else:
            tracks.append(Track(path=path, speaker=name, duration=stream["duration"],
                                stream_index=si if multi_stream else None))
    return tracks


def decode_track_audio(path: Path, stream_index: int | None, channel: int | None) -> np.ndarray:
    """Decode one stream (and optionally one channel of it) to mono float32 16 kHz."""
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    to_mono = AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    to_planar = None
    chunks: list[np.ndarray] = []

    def collect(frames) -> None:
        for out in frames:
            arr = out.to_ndarray().reshape(-1)
            if arr.size:
                chunks.append(arr)

    with av.open(str(path)) as container:
        # Not `stream_index or 0`: that reads index 0 and "no index given" as
        # the same thing. They mean the same today, but the file can also have
        # changed since it was probed, which would raise a bare IndexError
        # into the worker's generic handler.
        index = stream_index if stream_index is not None else 0
        streams = container.streams.audio
        if index >= len(streams):
            raise ValueError(f"{path.name} has no audio stream {index + 1} "
                             f"({len(streams)} found) — has the file changed?")
        stream = streams[index]
        for frame in container.decode(stream):
            if channel is None:
                collect(to_mono.resample(frame))
                continue
            # Convert to planar float so one channel can be picked out, then
            # feed that channel through the mono/16k resampler.
            if to_planar is None:
                to_planar = AudioResampler(format="fltp", layout=frame.layout,
                                           rate=frame.sample_rate)
            for planar in to_planar.resample(frame):
                data = planar.to_ndarray()
                row = np.ascontiguousarray(data[min(channel, data.shape[0] - 1)][None, :])
                mono = av.AudioFrame.from_ndarray(row, format="flt", layout="mono")
                mono.sample_rate = planar.sample_rate
                collect(to_mono.resample(mono))
        if to_planar is not None:
            for planar in to_planar.resample(None):
                data = planar.to_ndarray()
                row = np.ascontiguousarray(data[min(channel, data.shape[0] - 1)][None, :])
                mono = av.AudioFrame.from_ndarray(row, format="flt", layout="mono")
                mono.sample_rate = planar.sample_rate
                collect(to_mono.resample(mono))
        collect(to_mono.resample(None))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def iter_stream_power(path: Path, stream_index: int | None, rate: int) -> Iterator[np.ndarray]:
    """Decode one stream at `rate` as chunks of power summed over its channels.

    For callers that measure loudness once, in passing — track alignment,
    which reads whole files: nothing is held beyond one decoded chunk, where
    a four-hour track at 16 kHz is close to a gigabyte. Every channel counts
    in full. A mono downmix would not do: it gives an LFE-labelled channel
    (the 4th of a 5.1 layout, the 3rd of a 2.1) no weight at all, and in a
    multitrack file that channel can be someone's microphone.
    """
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    def power(frames) -> Iterator[np.ndarray]:
        for out in frames:
            # Packed float: one row of interleaved samples.
            samples = out.to_ndarray().reshape(-1, len(out.layout.channels))
            if samples.size:
                # Per sample, before channels are summed: a filter gone
                # unstable in one channel (NaN to the end of the file) must
                # not take the healthy channels beside it down too.
                samples = np.nan_to_num(samples.astype(np.float64),
                                        nan=0.0, posinf=0.0, neginf=0.0)
                yield np.sum(samples * samples, axis=1)

    with av.open(str(path)) as container:
        index = stream_index if stream_index is not None else 0
        streams = container.streams.audio
        if index >= len(streams):
            raise ValueError(f"{path.name} has no audio stream {index + 1} "
                             f"({len(streams)} found) — has the file changed?")
        resampler = None
        for frame in container.decode(streams[index]):
            if resampler is None:
                resampler = AudioResampler(format="flt", layout=frame.layout, rate=rate)
            yield from power(resampler.resample(frame))
        if resampler is not None:
            yield from power(resampler.resample(None))
