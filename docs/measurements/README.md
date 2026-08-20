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
| `2026-08-20-hostprobe-flame-2026.2.txt` | Phase 0 probe, two sessions in Flame 2026.2. Supports *Measured — Phase 0 probe run in Flame*. |
