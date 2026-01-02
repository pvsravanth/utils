"""
Stereo audio channel activity detector.

This utility scans a WAV file in short frames and reports which channel
(left/right) is more active, useful for quick speaker/role heuristics such as
patient/clinician tracks. It skips frames below a noise threshold and applies a
simple linear gain before energy comparison.

Dependencies
------------
pip install numpy

Quickstart
----------
```python
from Python.audio_channel_activity import get_channel_activity

activity = get_channel_activity("call.wav", frame_duration_ms=500, noise_threshold=5000)
for ts, speaker in activity:
    print(f"{ts:.2f}s -> {speaker}")
```
"""

from __future__ import annotations

import wave
from typing import List, Tuple

import numpy as np

# Map WAV sample width (bytes) to numpy dtype.
_SAMPLE_WIDTH_TO_DTYPE = {1: np.int8, 2: np.int16, 4: np.int32}


def get_channel_activity(
    filename: str,
    frame_duration_ms: int = 500,
    noise_threshold: int = 5000,
    volume_gain: float = 0.1,
) -> List[Tuple[float, str | None]]:
    """
    Analyze a stereo WAV file and mark which channel is more active per frame.

    Args:
        filename: Path to the .wav file (must be stereo).
        frame_duration_ms: Frame duration in milliseconds for analysis windows.
        noise_threshold: Minimum absolute energy to consider a frame speech-like.
        volume_gain: Linear gain applied to samples before energy comparison.

    Returns:
        List of (timestamp_seconds, label) where label is "spk0" (left channel),
        "spk1" (right channel), or None if energies match.
    """

    with wave.open(filename, "rb") as wf:
        n_channels = wf.getnchannels()
        if n_channels != 2:
            raise ValueError("get_channel_activity only supports stereo WAV files.")

        sample_width = wf.getsampwidth()
        dtype = _SAMPLE_WIDTH_TO_DTYPE.get(sample_width)
        if dtype is None:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")

        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()

        frames_per_window = int(frame_rate * frame_duration_ms / 1000)
        activity_timestamps: List[Tuple[float, str | None]] = []

        for i in range(0, n_frames, frames_per_window):
            wf.setpos(i)
            frames = wf.readframes(frames_per_window)
            if not frames:
                break

            # Convert frame bytes to float32 to avoid overflow when applying gain.
            data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            data *= volume_gain

            # Split interleaved stereo channels.
            left_channel = data[0::2]
            right_channel = data[1::2]

            # Compute simple energy measure (sum of absolute values).
            left_energy = float(np.sum(np.abs(left_channel)))
            right_energy = float(np.sum(np.abs(right_channel)))

            # Skip low-energy frames.
            if left_energy < noise_threshold and right_energy < noise_threshold:
                continue

            timestamp = i / frame_rate
            if left_energy > right_energy:
                activity_timestamps.append((timestamp, "spk0"))
            elif right_energy > left_energy:
                activity_timestamps.append((timestamp, "spk1"))
            else:
                activity_timestamps.append((timestamp, None))

    return activity_timestamps
