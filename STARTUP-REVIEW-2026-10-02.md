# Startup and efficiency fixes — 2026-10-02

Implemented in the source app. Launch it with `TransViber TransScriber.bat`.

The window previously called CTranslate2's CUDA check synchronously.
CTranslate2 imports its converter modules, which also load PyTorch. Opening
the window therefore loaded hundreds of MiB of inference libraries even
before any audio was added. Audio probing also imported NumPy unnecessarily.

CUDA detection now runs after the window opens, on a worker, using the
[NVIDIA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html).
Inference packages and NumPy stay lazy. Backend model-loading errors still
use the existing CPU fallback.

## Measurements and verification

Offscreen launches on this machine, with temporary notes/settings:

| Measurement | Original HEAD | Updated source |
| --- | ---: | ---: |
| Python process to ready window | 2.843 s | 0.165 s |
| Resident RAM at ready window | 675.7 MiB | 52.6 MiB |
| Inference imports at startup | torch, ctranslate2, numpy | none |

Updated GPU detection completed at 0.184 s. These are warm launch probes,
not a measurement after reboot or of Explorer starting Python. The reported
1–2 minute delay was not reproduced. The launcher now logs the click time,
and windowed Python logs startup stages and exceptions to `app.log`, so a
remaining delay before Python starts can be distinguished from app loading.

The offline regression suite passes: 146 tests. A real CUDA smoke test
with the bundled `distil-large-v3.5-ct2` correctly transcribed a synthetic
recording: “The dragon attacked our party. Alice and Bob ran away. We will
meet tomorrow at noon.” No personal recording was needed.

## Other fixes

- Keep the ASR model for repeat transcription runs. Release it on a
  model/device change, AI Notes, or close. Avoid importing PyTorch during
  cleanup; avoid an extra full decoded-audio copy.
- Retain CUDA DLL-directory handles for the life of the app.
- Keep the window alive until all background threads actually stop, with
  an interruptible statistics-poll wait and cancellation checks during
  notes-engine startup and after Qwen work. End stalled notes streams on
  cancellation and treat the resulting connection error as cancellation.
- Clear the transcript render state as well as the tracks. Switching to
  the Notes tab can no longer resurrect a cleared transcript.
- Avoid resaving unchanged notes during startup. Preserve an unreadable
  notes file rather than overwriting it with an empty editor; show save
  failures in the status bar.
- Clear old track-error tooltips on retry, and show a useful launcher error
  if the Python environment is missing.

First model loads/downloads can still take time. Active downloads and model
construction may finish before cancellation can take effect; close now waits
safely instead of destroying their threads. Audio probing still happens on
the UI thread, so large batches or slow network storage remain a possible
future responsiveness improvement. The existing standalone installers and
portable builds are snapshots and have not been rebuilt in this session.
