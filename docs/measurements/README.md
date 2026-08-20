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
