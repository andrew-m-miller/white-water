# Raw measurement transcripts

Verbatim shell output from the box, kept because `host-notes.md` records conclusions and
conclusions get revised. When a later reading contradicts an earlier one, the argument is
settled by going back to what the machine actually printed — not by re-running a command on
a box whose software has moved on since.

Named `YYYY-MM-DD-<subject>.txt`. Each should be referenced from the entry in
`host-notes.md` that it supports.

| File | Subject |
|---|---|
| `2026-08-20-mocha-investigation.txt` | Mocha Pro 2026.5 ML stack on `flame6`, Rocky 9.5. Supports *Measured — Mocha Pro's ML architecture*. |
| `2026-08-20-nvidia-smi-compute-apps.csv` | Per-process VRAM with Flame open and Mocha's ML matte running. Same section. |
| `2026-08-20-flame-loaded-libraries.txt` | Libraries mapped by the live Flame 2026.2.1 process. Supports *Measured — Flame already runs ONNX Runtime*. |
| `2026-08-20-hostprobe-flame-2026.2.txt` | Phase 0 probe, first run. Item 1 skipped by a probe bug; kept because items 2-5 were valid. |
| `2026-08-20-hostprobe-flame-2026.2-complete.txt` | Phase 0 probe, all five items. The authoritative run for *Measured — Phase 0 probe run in Flame*. |
| `2026-08-20-hostprobe-anamorphic-no-insert.txt` | PAR-2 clip with Insert disconnected. Source for the anamorphic and disconnected-clip findings; its tile verdict is wrong, see correction 5. |
| `2026-08-20-ortprobe-isolation.txt` | ORT isolation probe, two sessions. The second is authoritative; the first's DEEPBIND result was an artifact of a shared library handle. |
| `2026-08-21-ortprobe-sea-raft-m-flame.txt` | Verbatim Flame 2026.2 SEA-RAFT M CPU/CUDA probe. Both paths pass identity and direction under the private ORT 1.29 runtime. |
| `2026-08-21-ortprobe-cuda-warnings.txt` | Verbatim ORT CUDA-provider warnings from the SEA-RAFT M run: nine inserted Memcpy nodes and CPU-assigned shape operations; performance diagnostics, not correctness failures. |
| `2026-08-21-ort-cuda-closure-flame.txt` | Initial ORT CUDA payload closure from the unpacked artifact's shell environment; four NVIDIA dependencies remained unresolved because this was not Flame's loader path. |
| `2026-08-21-ort-cuda-closure-flame-env.txt` | Installed-bundle closure using the generic shell environment. Confirms the same four CUDA SONAMEs remain unresolved without Flame's private search directories. |
| `2026-08-21-ort-cuda-closure-flame-live-map.txt` | Authoritative installed-bundle closure using the live Flame 2026.2.1 loader search path. Resolves cuBLAS, cuBLASLt, CUDA runtime and cuRAND to Flame; unresolved dependencies: none. |
| `2026-08-21-ortprobe-gpu-mem-limit-flame.txt` | Verbatim Flame 2026.2 controlled CUDA-arena-limit run. A 64 MiB `gpu_mem_limit` produced an explicit `BFCArena` allocation failure during session creation, returned device VRAM to the pre-test level, and passed fresh-session numerical CPU recovery. |
| `2026-08-21-ortprobe-highres-control-flame.txt` | Clean-launch, no-environment-variable control for the high-resolution artifact. The ordinary extended CPU/CUDA, lifecycle, cancellation, fallback and controlled-recovery gates pass. |
| `2026-08-21-ortprobe-highres-uhd-flame.txt` | GPU-only UHD 2160×3840 qualification with a 16 GiB ORT arena limit. The warm run reaches a classified bounded-allocation stop; the required measurement-result gate passes. |
| `2026-08-21-ortprobe-highres-dci4k-flame.txt` | GPU-only DCI 4K 2160×4096 qualification with a 16 GiB ORT arena limit. The warm run reaches a classified bounded-allocation stop; the required measurement-result gate passes. |
| `2026-08-21-ortprobe-highres-alexa35-flame.txt` | GPU-only Alexa 35 open-gate qualification. The valid session pads 3164×4608 to 3168×4608 and reaches a classified bounded-allocation stop; the appended file also retains an earlier invalid configuration session. |
