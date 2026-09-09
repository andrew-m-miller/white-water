# SPIKE — NeuFlow v2 CUDA qualification (Path A)

**Status: spike for review. Nothing here is wired into CI, and the recording changes in section 4
are NOT applied** — applying them before a real on-box PASS would reintroduce exactly the
fail-closed violation Codex flagged (F2: a NeuFlow CUDA row is not schedulable until a genuine CUDA
result exists).

## 1. What this is

A torch-free way to establish the NeuFlow CUDA provider qualification the frozen protocol requires
(`docs/phase2.5-protocol-v2.md`: *provider support is potential capability, not qualification
evidence… no NeuFlow CUDA row is schedulable until a returned report candidate explicitly carries
`measurement_providers: ["cuda"]`*), **without building or legally re-signing a new CUDA validation
runtime**.

The NeuFlow torch↔ONNX numerical parity is already frozen from the CPU validation
(`models/neuflow-v2.json`). A CUDA qualification therefore only has to prove the **same** graph:

1. runs on the CUDA execution provider and **first-selects it** (no silent CPU fallback);
2. produces finite output of the exact fixed shape/dtype; and
3. still passes the frozen correctness gates (identity EPE, both signed flow directions), **and**
   agrees with the CPU execution provider within a tight tolerance (default `1e-2` px).

The reference is the **CPU ONNX session**, not PyTorch — so the qualifier needs only
`onnxruntime` (CUDA) + `numpy`, both already present in the **P25-6/P25-7 measurement runtime**,
which is already built and legally signed (inventory `6b431b40…`, PR #38). No new pack, no new
Andrew sign-off.

## 2. Deliverables in this spike

- `models/qualify_neuflow_cuda.py` — the qualifier. Reuses the CPU validation's own acceptance
  helpers (`_validate_provider_selection`, `_validate_fixed_io`) so CUDA is held to the same rules.
  Sessions and the available-provider list are injectable for testing.
- `models/qualify_neuflow_cuda_tests.py` — dependency-light (numpy + fakes, **no GPU/torch/ORT**).
  Covers: CUDA first-selected happy path, silent-CPU-fallback rejection, CUDA-vs-CPU tolerance
  breach, non-finite output, wrong flow direction, and CUDA-unavailable. `PASS` locally.
  When this graduates from spike, register it in `CMakeLists.txt`:
  `add_test(NAME models::neuflow_cuda_qualifier COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/models/qualify_neuflow_cuda_tests.py")`

## 3. Running it on the box (per RUN-P25-7 step 5 CUDA preflight)

Inside the extracted P25-7 (or P25-6) runtime, after the CUDA loader preflight
(`LD_LIBRARY_PATH` = carried ORT dir first, then the Flame CUDA dir; no sudo):

```
python3 models/qualify_neuflow_cuda.py \
  --manifest models/neuflow-v2.json \
  --onnx     <path to the CPU-validated neuflow-v2-mixed-opset17-432x768.onnx> \
  --record   models/neuflow-v2.json
```

It prints the CUDA `provider_validation` + evidence and (with `--record`) writes it to
`validation.observed.cuda_provider_qualification`. It exits non-zero on any failed gate and records
nothing. It does **not** touch `measurement_providers` or `status`.

## 4. Downstream recording changes — GATED on a passing run (shown, not applied)

Once the box returns a PASS, these are the exact edits to flip NeuFlow CUDA on. They are the
reverse of the F2 revert.

**a. `bakeoff/p25-7/inputs/candidate-entries.json` — neuflow-v2:**
```diff
-    "measurement_providers": ["cpu"],
+    "measurement_providers": ["cpu", "cuda"],
```
and the `exclusion_reason.message` drops the "GPU measurement is deferred" clause and states the
CUDA provider is qualified by the torch-free CUDA qualifier (`cuda_provider_qualification` evidence
in `models/neuflow-v2.json`, provider first-selected, CUDA-vs-CPU max abs `<value from run>`).

**b. Re-add three selection files** (identical to the CPU neuflow screens, providers swapped):
`selection-screen-neuflow-production-log-cuda.json`,
`selection-screen-neuflow-production-scene-linear-cuda.json`,
`selection-screen-neuflow-synthetic-lattice-cuda.json`, each with
```json
"providers": [{"token": "cuda", "host_loads": ["idle", "live_flame"], "gpu_mem_limit_mib": 15000}]
```
(NeuFlow's cap stays `mp0_331776`; these ride `screen` rigor — one tier below the finalist's
`final`, an accepted asymmetry for the fast-alternative comparison.)

**c. Docs** (`RUN-P25-7.txt`, `PENDING-ONBOX-VALIDATION.md` §3 neuflow bullet): revert the
"deferred pending a CUDA qualification run" language to "qualified via the torch-free CUDA
qualifier; GPU numbers captured at `screen` rigor." Restore the `-cuda` files in the screen list
and the step-5 note.

**Validation after applying:** `build_matrix` returns to **269 cells** (221 + 48 NeuFlow CUDA), all
selections plan, neuflow-v2 measurable on cpu+cuda.

## 5. Effort / risk

- **Effort:** the qualifier + test exist (this spike); remaining work is one on-box CUDA run and the
  small section-4 recording edits. No new runtime, no legal sign-off.
- **Risk:** low — the graph is standard-domain (the operator gate enforces it), so ORT CUDA runs it;
  the qualifier fails closed on fallback, drift, non-finite, or wrong direction.
- **Decision for Andrew:** accept the qualification *definition* — ONNX-on-CUDA provider selection +
  finite output + frozen direction/identity gates + CUDA-vs-CPU tolerance, with torch parity taken as
  already-frozen on CPU. If a fresh torch-on-CUDA parity is required instead, that is Path B (a new
  signed CUDA validation runtime) and this spike does not apply.

## 6. Not covered

WAFT is out of scope: it is fixed 128×192 (no shared-lattice cell) and its exporter couples
`device=cuda ⟺ provider=cuda` (torch-CUDA), so the torch-free shortcut does not carry over.
