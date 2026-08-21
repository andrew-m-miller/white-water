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
| `2026-08-21-ort-cuda-closure-flame.txt` | ORT CUDA payload and loader-resolved closure measured from the Flame box shell; four NVIDIA dependencies remained unresolved in that shell environment. |
