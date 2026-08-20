# Host notes

What each OFX host actually supports, measured rather than assumed. Every claim in here
should come from a probe report, not from documentation or memory — vendor docs are
incomplete and, for the features this project depends on, wrong in ways that would cost us
months.

Two kinds of entry live here and they must never be confused:

- **Measured** — a probe report says so, and the report names the host build and date.
- **Inherited** — measured by warp-drive on the same hardware and the same Flame builds.
  Trustworthy, but about a plugin that asked different questions. Re-measured only where
  White Water depends on something warp-drive did not.
- **Open** — not measured by anybody. Phase 0 exists to close these.

---

## Inherited from warp-drive

Source: `warp-drive/docs/host-notes.md`, commit `887a123`. Flame 2026.2 Linux measured
2026-08-06; Flame 2027 macOS measured 2026-08-12. Host id `com.autodesk.flame`, OFX API
1.4, 72 CPUs. Probed in Filter context (1920x1080, PAR 1) and General context (4608x3164,
PAR 2, project size 9216x3164).

### Capabilities

| Capability | Result |
|---|---|
| Overlay interact | Yes — `Describe`, `CreateInstance`, `GainFocus`, `Draw`, `PenMotion`, `LoseFocus` all delivered |
| `OfxDrawSuite` v1 | Available — overlays need no OpenGL linkage |
| `clipGetImage` outside render | Works at every offset tested (0, ±1, ±5, ±24) |
| `OfxProgressSuite` v1 and v2 | Available |
| `OfxTimeLineSuite` v1 | Available |
| Message suite V1 (`sendMessage`) | Present; raises a Flame error dialog |
| Message suite V2 (`setPersistentMessage`) | Present in Flame 2027; surfaces as Flame Console text, not a node badge |
| Parametric params | **Rejected** (`kOfxStatErrUnsupported`) |
| Per-parameter custom interact | `SupportsCustomInteract = 0` |
| String animation | Not supported |
| Choice / StrChoice / Boolean animation | Supported; StrChoice present in 2026.2 |
| Multi-resolution | `SupportsMultiResolution = 0` |
| Tiles | `SupportsTiles = 1` |
| Pixel depths | Byte, Short, Float **and Half** |
| Components | RGBA, RGB, Alpha |
| Premultiplication | **UnPremultiplied** |
| GPU render suites | OpenGL `"true"`; **CUDA, Metal, OpenCL all `"false"`** |
| Multiple clip depths / PARs | Both 0 |
| Contexts | Filter, General, Generator, Transition |
| `SequentialRenderStatus` | 1 on first render, `IsInteractive = 0` |

### Quirks that shape this plugin

- **GPU properties are strings, not ints.** Reading `kOfxImageEffectPropOpenGLRenderSupported`
  as an int returns `kOfxStatErrUnknown`. Any code testing these must read them as strings.
  This is also how we know Flame offers **no** CUDA/Metal/OpenCL OFX render suite, which is
  the fact that forces inference to run in-process on its own device context.
- **`kOfxImageEffectHostPropNativeOrigin` has dimension 0** — the property exists with no
  value, so a plugin cannot learn the origin from Flame. Bottom-left confirmed empirically
  by an asymmetric overlay screenshot.
- **Flame hands OFX 0-based time.** A batch starting at frame 1001 arrives as OFX time 0.
  Reported from long facility experience, including the same behaviour in Mocha — a host
  convention, not one plugin's bug. **Not instrumented by any probe; treat as reported.**
  This is why the reference frame gets a `Set Ref` button that writes `args.time` rather
  than trusting an artist to type a number.
- **Anamorphic PAR**: image bounds are in real pixels, `ProjectSize` is in square-pixel
  terms. A 4608x3164 PAR-2 source reports a 9216x3164 project size.
- **The panel is a flat list.** Parameters flow left to right in columns each headed
  "Controls". Group parameters do not visibly nest their children. Pages are unsupported.
- **Labels are truncated at roughly 12 characters.** Every `probe OfxParamTypeX` label
  rendered as `probe OfxParamT`.
- **`paramDefine` accepting a type is not proof the host renders it.** Group, Page and
  Custom were all accepted; Autodesk documents Pages as unsupported.
- **`kOfxDrawPrimitiveRectangle` renders filled**, not as an outline. Use
  `kOfxDrawPrimitiveLineLoop`.
- **Overlays get no keyboard.** `KeyDown`/`KeyUp`/`KeyRepeat` were never received across
  four sessions with keys pressed. `kOfxInteractPropPixelScale` reads `[1,1]` at every
  zoom. `kOfxInteractPropPenViewportPosition` is not a viewport position — do not use it.
- Both the timeline soft effect and the Batch node work.

### Failure modes

- **A throw out of `describe`/`describeInContext` means the host shows no plugin at all and
  logs nothing** — the worst failure mode available. Set risky properties through the
  property set with the non-throwing flag rather than through a setter that throws.
- **`-fvisibility=hidden` hides the OFX entry points** unless they carry an explicit
  `visibility("default")`. A version script cannot resurrect a symbol hidden at compile
  time. The support library archive is therefore built at default visibility.
- **A wrong-glibc Linux build shows no plugin at all, with no error anywhere.**
- The support library fetches `kOfxMemorySuite` unconditionally at load and
  `kOfxInteractSuite` whenever the host advertises overlays; a host missing either fails
  every such plugin with `kOfxStatErrMissingHostFeature`.

### Installation and diagnostics

- Linux scan dir `/usr/OFX/Plugins`; macOS `/Library/OFX/Plugins`; `OFX_PLUGIN_PATH`
  overrides at runtime.
- **Flame captures plugin stderr in `/opt/Autodesk/log/`.** On a machine nobody can attach
  a debugger to, that is the only diagnostic channel — so anything worth knowing gets
  printed there at `load()`.
- The facility Flame box is airgapped. Artifacts are built by manual CI and carried over
  with a SHA256.

---

## Measured — Mocha Pro's ML architecture

Flame box `flame6`, Rocky Linux 9.5, **2026-08-20**. Mocha Pro 2026.5 installed at
`/usr/OFX/Plugins/BorisFX/MochaPro2026.5/`. Filesystem inspection only; no process was
observed running.

### Mocha Pro runs its ML inference out of process

| Evidence | Reading |
|---|---|
| `mochaPro2026.ofx` is **2.5 MB** and links **no** ML runtime (`ldd` with `LD_LIBRARY_PATH` cleared). Same for `mochaVR2026.ofx` and Silhouette's `mochaPro11.ofx`. | The OFX plugin is a shim. |
| The entire ML stack lives in `Resources/mochaui/lib/ML/` — **outside** the `.ofx.bundle`, beside `Resources/mochaui/bin/mochaui`, an executable. | Inference belongs to the standalone application. |
| `Resources/mochaui/bin/` also contains `HostTestServer`, `ChannelTestServer`, `EchoTestServer`, `ServerNodeTestServer`. | A client/server IPC architecture, with test harnesses for it — the same shape as warp-drive's `src/ipc/` frame channel. |

**Confirmed at runtime**, with Flame open and Mocha's ML matte working
(`docs/measurements/2026-08-20-nvidia-smi-compute-apps.csv`):

| pid | process | VRAM |
|---|---|---|
| 2159577 | `/opt/Autodesk/flame_2026.2.1/bin/flame` | 1,523 MiB |
| 2187840 | `…/MochaPro2026.5/Resources/mochaui/bin/mochaui` | **4,903 MiB** |
| 79812 | `./rgsender` (remote graphics) | 204 MiB |

Two separate processes hold CUDA allocations on the same GPU at the same time, and the ML
one holds three times what the host does.

### What this retires, and what it leaves

**Retired: device contention.** A second process demonstrably gets a large CUDA allocation
alongside a running Flame, on this box. That was the open worry behind Phase 0 item 3 — that
Flame owning the GPU would leave nothing to work with — and it is answered. It also means
GPU inference on a Flame box is not merely plausible but *demonstrated by a shipping
product* on this exact hardware.

**Still open, and now narrower.** Item 3 asked two questions bundled together. Only the
linker question remains — see the next section, which answers a large part of it.

**Do not over-read the architecture.** Mocha is a standalone application with an OFX front
end, so its process split is probably product history rather than a technical verdict on
in-process inference. But the practical consequence stands either way: **the project now has
a path that is known to work.** If in-process fails, out-of-process is not a fallback we
would be inventing under pressure — it is a shape a vendor ships, and warp-drive already has
the machinery (`src/ofx/EditorProcess.cpp`, `src/ipc/`).

**Budget number worth keeping:** 4.9 GB is what ML matting actually costs in VFX practice.
Our own budget should be planned against that order of magnitude, not against a paper figure.
Total GPU capacity was not captured — run this to get the headroom:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
```

### ONNX Runtime is the right call, and their abstraction matches ours

```
Resources/mochaui/lib/ML/
  libonnxruntime.so                    23 MB
  libonnxruntime_providers_cuda.so    775 MB
  libonnxruntime_providers_shared.so   16 KB
  libMLBackend_ort_cpu.so             1.6 MB
  libMLBackend_ort_cuda.so            2.7 MB
  libMLPlugin.so                      3.0 MB
  libcudart.so.12, libcublas.so.12, libcublasLt.so.12, libcudnn*.so.9.10.2
```

Boris FX ships **exactly** the runtime this project chose. Better, `libMLBackend_ort_cpu` /
`libMLBackend_ort_cuda` behind `libMLPlugin` is the same pluggable-backend design as our
`FlowEstimator` plus the `Device: Auto / GPU / CPU` parameter — arrived at independently.

**They dropped TensorRT.** The older Sapphire-bundled Mocha carries TensorRT 8.5.3
(`libnvinfer.so.8`, `libnvonnxparser.so.8`) on CUDA 11 / cuDNN 8. Mocha Pro 2026.5 has **no
TensorRT at all** — plain ORT CUDA EP on CUDA 12 / cuDNN 9.10.2. That is a vendor with a
shipping product moving away from the TensorRT EP, which supports the plan's choice of the
plain CUDA EP over it.

### The payload is 3 GB, not 80 MB

**Measured: 21 files, 3.11 GB** in the ML directory alone; 4.12 GB across the whole
MochaPro2026.5 tree counting only shared libraries. The bulk is `libcublasLt.so.12` (752 MB),
`libonnxruntime_providers_cuda.so` (775 MB) and `libcudnn_engines_precompiled.so` (547 MB).

The plan estimates the bundle growing "by roughly 20-80 MB" for models. That is right for
*weights* and wrong for the thing that actually dominates, which is the CUDA/cuDNN/ORT
runtime. **Three orders of magnitude out.** Consequences:

- Not shippable inside a git-managed `.ofx.bundle`, and awkward even as a CI artifact.
- Boris FX's answer is worth copying: install a sibling tree next to the bundle
  (`/usr/OFX/Plugins/BorisFX/MochaPro2026.5/Resources/...`) rather than inside it. The
  bundle stays a bundle; the runtime is an installed component.
- A CPU-only build is small and should probably be the default artifact, with GPU support a
  separate, larger download.
- Worth checking before accepting the number: **ONNX Runtime 1.22 made cuDNN and cuFFT
  optional at runtime for the CUDA EP and stopped linking nvrtc**, specifically to cut this
  footprint. Mocha's build may predate that. A current ORT could be dramatically smaller.

*No `.onnx` files appear above — but the search matched only `*.so*`, `*.ofx` and `*.dat`,
so weights were never in scope. That is not evidence of absence.*

---

## Measured — Flame already runs ONNX Runtime, in its own process

Flame 2026.2.1 on `flame6`, **2026-08-20**, read from `/proc/<pid>/maps` of the live
process (needs sudo — Flame runs as another user). Raw:
`docs/measurements/2026-08-20-flame-loaded-libraries.txt`.

```
/opt/Autodesk/lib64/2026.2.1/libonnxruntime.so.1.22.0
/opt/Autodesk/lib64/2026.2.1/libcudart.so.12.8.57
/opt/Autodesk/lib64/2026.2.1/libcublas.so.12.8.3.14
/opt/Autodesk/lib64/2026.2.1/libcublasLt.so.12.8.3.14
/opt/Autodesk/lib64/2026.2.1/libcudnn.so.9.9.0
/opt/Autodesk/lib64/2026.2.1/libnvinfer.so.10.8.0
/opt/Autodesk/lib64/2026.2.1/libnvinfer_plugin.so.10.8.0
/opt/Autodesk/lib64/2026.2.1/libnvonnxparser.so.10.8.0
/opt/Autodesk/lib64/2026.2.1/libtbb.so.12.15
/usr/lib64/libcuda.so.580.95.05          (driver 580.95.05)
```

### The good half

**ONNX Runtime demonstrably initialises inside Flame's address space, because Flame does it
itself.** That is the single most important thing item 3 was trying to establish, and it is
now answered without us building anything. Flame is also on **CUDA 12.8 / cuDNN 9.9 /
TensorRT 10.8**, so a plugin built against CUDA 12 matches the host's major version rather
than fighting it.

This also corrects a guess recorded here earlier — that Flame might be on CUDA 11 while
Mocha is on CUDA 12. It is not. **CUDA major version is not a conflict.** Both Flame and
Mocha are on 12.

### The bad half, and it is sharper than the old worry

**Flame has `libonnxruntime.so.1.22.0` already loaded when our plugin is dlopened.** If we
bundle our own ONNX Runtime, there are two copies of it in one address space, and ORT
exports C symbols (`OrtGetApiBase` and friends) that the dynamic linker resolves globally,
first-loaded-wins. Our calls can bind to **Flame's** ORT while our data structures came from
ours — a version-skew crash with no useful backtrace, on a machine with no debugger.

This is precisely the collision `cmake/ofx.map` cannot prevent. The version script keeps
*our* symbols from being visible to Flame; it does nothing about *our references* binding to
*Flame's* exports. The note in that file anticipated this in the abstract; this is the
concrete instance, with a version number.

Note that Flame's ORT is 1.22.0 — the same release whose CUDA EP made cuDNN and cuFFT
optional at runtime, which is also the lead for shrinking our own 3 GB payload problem.

### Options, in rough order of preference

1. **Out of process.** The collision cannot occur, and it is what Boris FX ships. Costs an
   IPC boundary that warp-drive has already built (`src/ofx/EditorProcess.cpp`, `src/ipc/`).
2. **`dlopen` our ORT with `RTLD_LOCAL | RTLD_DEEPBIND`,** so our copy prefers its own
   symbols over the global ones. The classic fix, and it keeps everything in one process.
   `RTLD_DEEPBIND` has real hazards though — it breaks malloc interposition and can misroute
   exceptions across the boundary — so it needs proving, not assuming.
3. **Link ORT statically** with hidden visibility. Awkward to build, but the collision
   disappears at the source.
4. **Use Flame's ORT.** Tempting — the library is right there and the version is current.
   Undocumented, unsupported, and version-locked to whatever Autodesk ships next release.
   Not for a facility deliverable, though possibly interesting for an experiment.

**This is now the decision item 3 exists to settle**, and it is a much better-posed question
than "does GPU inference work at all". The probe should attempt (2), because if
`RTLD_DEEPBIND` works cleanly then in-process is available and simpler; if it does not, (1)
is already known to work on this hardware.

### Worth capturing next

- Does Flame ship `libonnxruntime_providers_cuda.so` on disk? It is not in the mapped list,
  but a provider not in use at that moment would not be mapped. If it is there, Flame's ML
  features use the CUDA EP and the whole stack is proven in-process.
- `readelf -d` on Flame's `libonnxruntime.so.1.22.0` for its `SONAME` and `RUNPATH` — the
  soname is what our references would bind to.

---

## Measured — Phase 0 probe run in Flame

Flame **2026.2** (`com.autodesk.flame`, OFX API 1.4, 72 CPUs) on `flame6`, Rocky 9.5,
**2026-08-20**. General context Batch node, 3840x2160 PAR 1, 23.976 fps, 184-frame clip,
float RGBA unpremultiplied. Raw:
`docs/measurements/2026-08-20-hostprobe-flame-2026.2-complete.txt` (an earlier partial run
is kept alongside it).

**All five questions are answered. Four are green; item 3 is a real constraint.**

### 1. Two clips in the General context — WORKS ✅

```
Defined optional second clip 'Insert': clipDefine kOfxStatOK, OfxImageClipPropOptional kOfxStatOK
clipGetHandle('Insert'): kOfxStatOK
  OfxImageClipPropConnected  = 1
  OfxImageClipPropOptional   = 1
  PixelDepth "OfxBitDepthFloat", Components "OfxImageComponentRGBA",
  PreMultiplication "OfxImageAlphaUnPremultiplied", PAR 1, FrameRange 0, 183
```

**Flame presents two RGBA clips as four sockets** — Front and Matte per clip — confirming the
forum report, and matching the workflow an artist expects. `kOfxImageClipPropOptional` is
accepted. The Insert clip reports the same format as Source.

**Still unknown: what a genuinely *disconnected* optional clip does.** `Connected` read 1 in
this run, so the disconnected path was never exercised. The render must therefore read
`kOfxImageClipPropConnected` and must not infer connectedness from `clipGetImage` succeeding
— which remains untested rather than disproven. Worth one more run with Insert deliberately
left empty.

### 2. `clipGetImage` during render — WORKS ✅

`6 of 6 offsets returned an image, no deadlock.` Every one of ±1, ±10, ±100 returned full
bounds `[0, 0, 3840, 2160]` from inside the render action, on a host thread.

**The on-demand flow chain is viable.** No mandatory Analyze pass, no disk cache.

**But Flame clamps rather than failing.** The clip is `[0, 183]` and the render was at t=0,
so `t-1`, `t-10` and `t-100` were all outside it — and each returned a valid full-bounds
image. Consequences for the chain:

- Read `kOfxImageEffectPropFrameRange` and clamp deliberately; **never** infer a boundary
  from `clipGetImage` failing, because it does not fail.
- A reference frame near either end would otherwise accumulate flow against a held frame,
  producing zero motion that looks like a correct static track.

### 3. ORT/CUDA symbol capture — the collision is REAL ⚠️

`dlsym(RTLD_DEFAULT, …)` from inside the loaded plugin:

| Symbol | Result | Owner |
|---|---|---|
| `OrtGetApiBase` | **VISIBLE** | `/opt/Autodesk/lib64/2026.2.1/libonnxruntime.so.1` |
| `OrtSessionOptionsAppendExecutionProvider_CUDA` | **VISIBLE** | same |
| `cudaRuntimeGetVersion`, `cudaMalloc` | **VISIBLE** | `libcudart.so.12` |
| `cudnnGetVersion` | **VISIBLE** | `libcudnn.so.9` |
| `cublasCreate_v2` | **VISIBLE** | `libcublas.so.12` |
| `createInferBuilder_INTERNAL` | **VISIBLE** | `libnvinfer.so.10` |
| `TBB_runtime_interface_version` | **VISIBLE** | `libtbb.so.12` |
| `protobuf_shutdown`, `…ShutdownEveryN` | **absent** | — |

Flame loads ONNX Runtime, CUDA, cuDNN, cuBLAS, TensorRT and TBB into the **global** scope.
A bundled copy of any of them has our references captured by Flame's, first-loaded-wins.
**Bundling our own ONNX Runtime and simply linking it is not safe.**

One real piece of good news: **protobuf is not exposed**, so ORT 1.22 keeps it private. That
removes the worst of the classic collisions and leaves the ORT C API itself as the problem.

This is the one open architectural decision. See *The inference-loading decision* below.

### 4. `SupportsTiles = 0` — HONOURED ✅

**47 renders, every one** `window [0 0 3840 2160]` against `rod [0 0 3840 2160]`. Flame
advertises `SupportsTiles = 1` as a host but respects a plugin declaring 0. No tile-assembly
logic needed; whole-frame inference is safe.

### 5. `getFramesNeeded` — CALLED, and `{N}` is fine ✅

**Called 256 times** against 47 renders, declaring
`OfxImageClipPropFrameRange_Source = [0 0]` and `…_Insert = [0 0]` each time, all
`kOfxStatOK` — **and item 2's distant pulls still succeeded**. Declaring the honest minimum
does not restrict what `clipGetImage` will serve during render, so the over-fetching worry
does not arise.

**Note the 5:1 ratio.** Flame calls this action far more often than it renders, so the real
plugin's handler must be cheap: no document work, no cache lookups, no allocation. Same rule
warp-drive applies to `getRegionsOfInterest`.

### Overlay — pen confirmed, keyboard confirmed absent

This run exercised the viewer properly and supersedes the earlier inconclusive result:

| Action | Count | |
|---|---|---|
| `PenMotion` | 904 | |
| `Draw` | 66 | |
| `PenDown` | **10** | with pressure (0.248, 0.423, 0.610, 0.544 …) — a tablet |
| `PenUp` | **10** | |
| `GainFocus` / `LoseFocus` | 1 each | |
| `KeyDown` / `KeyUp` | **0** | never received |

Matches warp-drive on both counts. **A Flame overlay has a pen and no keyboard.** Also
observed: pen canonical coordinates go **negative** (−5.5, −14.6) when the pen leaves the
image area, so an overlay must tolerate out-of-bounds positions rather than assuming they
are inside the frame.

### Other results

- **Half float confirmed** in `SupportedPixelDepths`.
- **1 MB string parameter survives save/reload byte-identical** (checksum matched across a
  setup reload, General context, secret parameter). Re-confirms warp-drive.
- `CudaEnabled` / `MetalEnabled` / `OpenCLEnabled` **absent** at render
  (`kOfxStatErrUnknown`); `OpenGLEnabled = 0`.
- `SequentialRenderStatus = 1`, `InteractiveRenderStatus = 0` on first render.
- `MaxPages = 0`, `PageRowColumnCount = 0, 0` — pages unusable.
- Parametric parameters rejected; all other types accepted, including `Custom`, `Group` and
  `Page` — acceptance is not proof of rendering.

---

## The inference-loading decision

Phase 0 leaves exactly one architectural question open, and it is now well posed.

Flame has ONNX Runtime 1.22.0 in its global symbol scope. Three ways to live with that:

| Option | Cost | Confidence |
|---|---|---|
| **Out of process** | An IPC boundary and a supervised helper. warp-drive has both (`src/ofx/EditorProcess.cpp`, `src/ipc/`); Mocha ships exactly this shape and it is measured working on this box. | **Known to work** |
| **`RTLD_LOCAL` + `RTLD_DEEPBIND`** | Keeps one process, much less machinery. But `DEEPBIND` breaks malloc interposition and can misroute exceptions across the boundary. | **Unmeasured** |
| **Static ORT, hidden visibility** | Collision gone at the source. Awkward to build; ORT's static build is not a well-trodden path. | Unmeasured |

**Next measurement, and the last one before Phase 1:** a second probe bundle that links a
real ONNX Runtime and attempts `RTLD_LOCAL|RTLD_DEEPBIND`, reporting whether
`OrtGetApiBase` resolves to *ours* rather than Flame's, and whether a trivial CUDA session
then initialises. Kept as a separate bundle so a runtime that refuses to load cannot take
the capability probe down with it.

A clean result makes in-process available and much simpler. A dirty one costs nothing,
because out-of-process is already known to work.

## Open

Phase 0's five questions are closed — see *Measured — Phase 0 probe run in Flame*. What
remains:

1. **Does `RTLD_DEEPBIND` isolate a bundled ONNX Runtime from Flame's?** The last blocking
   measurement. See *The inference-loading decision*.
2. **What does a genuinely disconnected optional clip report?** `Connected` read 1 in every
   run so far, so the disconnected path is untested. One run with Insert left empty settles
   it.
3. **Whether render scale is ever anything but 1.** Every observation so far is `[1, 1]`.
4. **Whether `kOfxImagePropRowBytes` can be negative** in Flame, and whether images are ever
   windows into larger allocations. The vendored `HostImage` handles both; nothing has
   confirmed Flame exercises either.
5. **Anamorphic behaviour.** Every measurement here is PAR 1. warp-drive measured PAR 2
   separately; White Water has not.

### Procedure

Build the probe, install it, and open it in a **General-context Batch node**. It must
actually render — view it in the viewer — because items 2, 4 and 5 are driven from the
render action. Then press **Run Probe**, and exercise the viewer with the pen.

```bash
sha256sum -c whitewater-linux-*.tar.gz.sha256
```

```bash
cp -R WhiteWaterHostProbe.ofx.bundle /usr/OFX/Plugins/
```

Report goes to `$WHITEWATER_PROBE_LOG`, else `$TMPDIR/whitewater-hostprobe.txt`, else
`/tmp/whitewater-hostprobe.txt`, and to stderr which Flame captures in `/opt/Autodesk/log/`.

Paste results here under a **Measured** heading with the host build and date, and archive the
raw transcript in `docs/measurements/`.
