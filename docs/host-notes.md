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
| Multiple clip depths / PARs | Both 0 — see the consequence below |
| Contexts | Filter, General, Generator, Transition |
| `SequentialRenderStatus` | 1 on first render, `IsInteractive = 0` |

### Quirks that shape this plugin

- **`SupportsMultipleClipDepths = 0` forbids remapping output depth.** Measured in four
  transcripts. The OFX specification is explicit about what that means — from
  `third_party/openfx/Documentation/sources/Reference/ofxClipPreferences.rst`: if the host
  sets this to 0, all of the effect's clips share one depth and *the plugin may not remap
  them*. So there is no serving float ST output from a byte source, at any point in
  `getClipPreferences`. This is why the ST map is a separate float-only descriptor rather
  than an output mode. **This one is read, not measured** — the host property is measured,
  the rule it triggers is the specification's.
- **`eRenderInstanceSafe` is one render per instance, several instances at once.** Also read
  rather than measured: `third_party/openfx/Support/include/ofxsImageEffect.h:94`, and
  `ofxImageEffect.h:768`. `eRenderFullySafe` is the level that would permit concurrent renders
  on one instance and we do not declare it. Practical effect: per-instance caches need no
  same-instance de-duplication, and process-wide state must be thread-safe regardless of what
  a single Batch appears to do. See `docs/context.md`, correction 6, for why this is recorded
  here as read rather than measured.
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
`PairwiseFlowEstimator` plus the `Device: Auto / GPU / CPU` parameter — arrived at
independently.

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

**Disconnected behaviour measured 2026-08-20**, second run with Insert deliberately empty:

```
OfxImageClipPropConnected = 0
OfxImageClipPropOptional  = 1
clipGetImage: kOfxStatFailed
```

**It fails cleanly — there is no trap here.** An earlier note in this file speculated that
Flame might hand back a black image for an unconnected optional input; it does not.

But note what *is* misleading: the clip still reports `PixelDepth`, `Components`,
`PreMultiplication`, `PixelAspectRatio` and `FrameRange` as though it were connected
(float RGBA, PAR 2, `0, 5562`). **Only `kOfxImageClipPropConnected` is authoritative.**
Format properties on a disconnected clip are stale values, not evidence of a connection.

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

**47 renders at PAR 1**, every one `window [0 0 3840 2160]` against `rod [0 0 3840 2160]`.
Flame advertises `SupportsTiles = 1` as a host but respects a plugin declaring 0. No
tile-assembly logic needed; whole-frame inference is safe.

**A PAR-2 run reported "NOT HONOURED -- 21 of 21 renders were partial". That was a probe
bug, not a host finding.** The render window is in *pixels* and `clipGetRegionOfDefinition`
returns *canonical* (square-pixel) coordinates; the probe compared them directly. On the
anamorphic clip the window was `[0 0 4608 3164]` and the RoD `[0 0 9216 3164]` — the same
region, since 4608 x PAR 2 = 9216. Every one of those renders was a full frame. Fixed by
converting the RoD to pixels (`x_pixel = x_canonical * renderScale.x / par`) before
comparing. **Tiles are honoured at both pixel aspect ratios.**

### 5. `getFramesNeeded` — CALLED, and `{N}` is fine ✅

**Called 256 times** against 47 renders, declaring
`OfxImageClipPropFrameRange_Source = [0 0]` and `…_Insert = [0 0]` each time, all
`kOfxStatOK` — **and item 2's distant pulls still succeeded**. Declaring the honest minimum
does not restrict what `clipGetImage` will serve during render, so the over-fetching worry
does not arise.

**Note the 5:1 ratio.** Flame calls this action far more often than it renders, so the real
plugin's handler must be cheap: no document work, no cache lookups, no allocation. Same rule
warp-drive applies to `getRegionsOfInterest`.

### Anamorphic (PAR 2) — measured

Second run on a 4608x3164 PAR-2 clip, 5563 frames:

| Property | Value |
|---|---|
| `ProjectSize`, `ProjectExtent` | `9216, 3164` — **canonical**, square-pixel |
| `ProjectPixelAspectRatio` | 2 |
| Source clip bounds | `4608 x 3164` — **real pixels** |
| Source `PixelAspectRatio` | 2 |
| `FrameRange` | `0, 5562` |
| Render window | `[0 0 4608 3164]` — pixels |
| Region of definition | `[0 0 9216 3164]` — canonical |

Confirms warp-drive's finding on this project's own probe: **image bounds are in real
pixels, project size and RoD are in square-pixel terms.** Everything else behaved as at
PAR 1 — in-render pulls returned `[0, 0, 4608, 3164]` at all six offsets, tiles honoured,
`getFramesNeeded` called 101 times against ~25 renders.

**Pen coordinates are canonical, not pixel.** On this clip they ranged past 7357, which only
fits the 9216-wide canonical space. Any overlay must divide by PAR to reach pixels — and
must tolerate negative values, which appeared here again (−592).

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

## Measured — ONNX Runtime isolation: in-process inference works

Flame 2026.2 on `flame6`, **2026-08-20**. Raw:
`docs/measurements/2026-08-20-ortprobe-isolation.txt` (second session; the first is kept
because its DEEPBIND result was an artifact and the contrast is instructive).

A separate probe bundle carrying its own ONNX Runtime **1.29.0**, deliberately different from
Flame's 1.22.0 so the reported version discriminates between the two copies. Reached only
through `dlopen`/`dlsym` — never linked, since a `DT_NEEDED` entry would be resolved through
the global scope and bind us to Flame's copy before any of our code ran.

| | handle | `OrtGetApiBase` | resolves into | version |
|---|---|---|---|---|
| Host | — | `0x7f526aa76970` | `/opt/Autodesk/lib64/2026.2.1/libonnxruntime.so.1` | **1.22.0** |
| `RTLD_LOCAL｜RTLD_DEEPBIND` | `0x39b27bd0` | `0x7f3afbdd3070` | our `libonnxruntime-b.so` | **1.29.0** |
| `RTLD_LOCAL` alone | `0x2f37cfc0` | `0x7f3af99d3070` | our `libonnxruntime.so` | **1.29.0** |

Both modes created an environment, built a session from the embedded model, ran it and
returned `[11 22 33 44]`. **Three ONNX Runtimes coexisted in Flame's process at once** — the
host's and two of ours — all functional.

### The decision: in-process, with plain `RTLD_LOCAL`

**Inference runs in-process.** No IPC boundary, no supervised helper, no frame transport.

**Use `RTLD_LOCAL` alone; do not use `RTLD_DEEPBIND`.** Both are measured working, so the
choice is which hazard to decline. `DEEPBIND` breaks malloc interposition and can misroute
exception unwinding across the boundary, where `std::type_info` identity fails and `catch`
and `dynamic_cast` stop working — real failure modes that would surface as inexplicable
crashes inside a host with no debugger. Taking that on to solve a problem that is already
solved would be paying a cost for nothing. `DEEPBIND` stays documented as the escalation if
a future ONNX Runtime or Flame combination ever shows capture.

**Why `RTLD_LOCAL` suffices, which was not the expectation.** `RTLD_LOCAL` governs whether
*our* symbols enter the global scope; it does not reorder how our own library's relocations
resolve, so on paper the host's copy should have been able to capture them. It cannot,
almost certainly because ONNX Runtime is built with hidden visibility: its internals resolve
to local symbols and never consult the global scope, and the exported C API is called only
by us through our own handle. The collision is real in principle and defused in practice by
ORT's own build hygiene — which is a property of *their* build, not a law, and therefore
worth re-checking whenever the bundled runtime version changes.

### What this does not establish

- The initial isolation probe used a 128-byte `Add` model. The real SEA-RAFT M run below
  now exercises a substantial network through both CPU and CUDA, so that narrow result is
  no longer the evidence for the inference decision.
- This isolation result does not establish lifecycle behaviour, VRAM headroom, cancellation,
  provider-init or GPU-OOM fallback, or warmed production-resolution performance. Those were
  measured by the later Phase 0B runs below.
- **Only version 1.29.0 against 1.22.0.** If Flame ever ships the version we bundle, the
  version string stops discriminating — a probe concern rather than a runtime one, since
  matching versions are less dangerous, not more.

## Measured — Phase 0B SEA-RAFT M in Flame

Flame **2026.2** (`com.autodesk.flame`, OFX API 1.4, 72 CPUs) on `flame6`, Rocky 9.5,
**2026-08-21**. The General-context probe ran the pinned, manifest-verified SEA-RAFT M
export at 128×192 through the private ORT **1.29.0** opened with plain `RTLD_LOCAL`; Flame's
global ORT is **1.22.0**. Raw transcripts:
[`2026-08-21-ortprobe-sea-raft-m-flame.txt`](measurements/2026-08-21-ortprobe-sea-raft-m-flame.txt)
and the separately captured provider warnings in
[`2026-08-21-ortprobe-cuda-warnings.txt`](measurements/2026-08-21-ortprobe-cuda-warnings.txt).

### Real-network result — PASS on CPU and CUDA

| Path | Session creation | First run | Identity median EPE | Translation medians | Verdict |
|---|---:|---:|---:|---|---|
| CPU | 941.6 ms | 529.4 ms | 0.0026 px | forward (4.0014, 0.0138), reverse (−4.0034, −0.0134) | Correct |
| CUDA | 934.8 ms | 1164.0 ms | 0.0027 px | forward (4.0017, 0.0137), reverse (−4.0032, −0.0134) | Correct |

The real network preserves both identity and direction in the isolated runtime. These are
session-creation and first-run timings at the tiny probe resolution, not a warmed
production-resolution performance gate: initialization, transfers and provider setup are
included, and the latter has not yet been measured at shipping resolutions.

### Library ownership and provider diagnostics

The bundled `libonnxruntime_providers_cuda.so` and
`libonnxruntime_providers_shared.so` were mapped from the probe's private tree. Before CUDA
session creation, the map already contained Flame's `/lib64/libcuda.so.1` and its
`libcublas.so.12`, `libcublasLt.so.12`, `libcudart.so.12`, `libcudnn.so.9`,
`libnvinfer.so.10` and `libnvinfer_plugin.so.10` under `/opt/Autodesk/lib64/2026.2.1/`.
After provider session/run/teardown, the probe's two private ORT provider libraries and
Flame's cuDNN component libraries were additionally mapped. TensorRT was already mapped;
this report does not establish that the ORT CUDA EP selected or used it. This initial
probe's library-name filter did **not** include `libcurand`, so this report alone did not
establish its owner. The later actual-loader-path closure below resolves that omission and
the other CUDA SONAMEs completely.

The shell warnings are performance diagnostics, not inference failures: ONNX Runtime
inserted nine `Memcpy` nodes in the CUDAExecutionProvider graph and assigned some
shape-related operations to CPU. They may reduce performance or prevent CUDA Graph
execution, but both numerical checks passed. Verbose node assignment is a later optimization
measurement, not a reason to reject the CUDA path.

### Payload closure and staging findings — COMPLETE

The first shell report used the bundle plus `/usr/lib:/usr/local/lib`, not Flame's live
loader search path, and therefore left `libcublas.so.12`, `libcublasLt.so.12`,
`libcudart.so.12` and `libcurand.so.10` unresolved. Re-running the same reporter against the
search path recovered from the live Flame process resolved all four from
`/opt/Autodesk/lib64/2026.2.1/`:

| SONAME | Flame-resolved file | Bytes |
|---|---|---:|
| `libcublas.so.12` | `libcublas.so.12.8.3.14` | 116,384,544 |
| `libcublasLt.so.12` | `libcublasLt.so.12.8.3.14` | 781,053,840 |
| `libcudart.so.12` | `libcudart.so.12.8.57` | 728,800 |
| `libcurand.so.10` | `libcurand.so.10.3.9.55` | 136,745,144 |

The authoritative live-map report ends with **`Unresolved dependencies: <none>`**. It
measures a **646,116,476-byte apparent** diagnostic payload (**646,123,520 allocated
bytes**), including the probe-only **31,958,904-byte** second ORT copy. Removing that
duplicate leaves **614,157,572 bytes** (614.2 MB decimal; about 585.7 MiB). The unique
loader-resolved external transitive closure is **1,138,007,368 bytes**: 1,034,912,328 bytes
of Flame-owned CUDA libraries above plus 103,095,040 bytes of driver/system libraries. These
external bytes are ownership accounting, not files to add to the probe bundle.

This closes Phase 0B CUDA dependency ownership and size accounting for this ORT 1.29 CUDA 12
payload on Flame 2026.2.1 / this Rocky 9.5 host. Raw reports:
[`initial shell path`](measurements/2026-08-21-ort-cuda-closure-flame.txt),
[`installed environment path`](measurements/2026-08-21-ort-cuda-closure-flame-env.txt), and
[`authoritative live Flame loader path`](measurements/2026-08-21-ort-cuda-closure-flame-live-map.txt).

An earlier report that the model was "absent" was a staging permissions failure, not a
missing ONNX file: the file existed with mode **0600**, readable only by its owner, while
the Flame runtime user was distinct and could not read it. It was corrected to **0644** and
the probe then loaded it. Model staging and CI must assert mode **0644** (or an equivalent
ACL that lets the Flame runtime user read the file) so this failure cannot recur.

## Measured — Phase 0B multiresolution and duplicated-node run

Flame **2026.2** on `flame6`, Rocky 9.5, **2026-08-21**. The supplied reports
`whitewater-ortprobe-multisize.txt` and `whitewater-ortprobe-multisize-duplicatednode.txt`
ran the same manifest-verified SEA-RAFT M probe at three real-model resolutions. The second
report is cumulative: it repeats the first node's 176-line report, then records the run from
the duplicated OFX node. The supplied reports are not archived in this repository; the
filenames, host and date identify the source of the measurements below.

### Warmed steady timings

The first and duplicated nodes both passed numerical identity/direction checks. Each timing
is the median of three steady samples after the reported warm run:

| Resolution | CPU node 1 | CPU duplicate | CUDA node 1 | CUDA duplicate |
|---|---:|---:|---:|---:|
| 480×640 | 6651.3 ms | 6705.4 ms | 40.7 ms | 40.9 ms |
| 720×1280 | 21818.7 ms | 21865.2 ms | 114.3 ms | 114.7 ms |
| 1080×1920 | 56299.6 ms | 56480.9 ms | 277.9 ms | 278.4 ms |

### VRAM, lifecycle and fallback observations

Both CUDA runs used a 24,564 MiB device and passed the VRAM check. Node 1 measured a
2,188.5 MiB baseline, 14,709.6 MiB peak/steady usage and a 12,521.1 MiB observed delta;
the duplicate measured a 2,445.6 MiB baseline, 14,771.6 MiB peak and 14,751.6 MiB steady
usage, with a 12,326.0 MiB observed delta. The baseline difference is a whole-device NVML
sample and does not, by itself, establish a leak. The 1080p steady samples left 9,854.4 MiB
free for node 1 and 9,812.4 MiB for the duplicate.

CPU and CUDA repeated create/run/destroy completed **3/3** in each report. Cross-thread
cancellation observed provider termination on both nodes. The injected invalid-device
provider-init failure then ran the CPU fallback and passed its numerical check on both nodes.
The post-run map identifies `libcurand.so.10` as Flame-owned at
`/opt/Autodesk/lib64/2026.2.1/libcurand.so.10`. The reported 3,253.1 MiB mapped-file total
includes Flame's already-loaded CUDA/cuDNN/TensorRT libraries; it is not the distributable
payload size.

The two node handles were distinct (`0x2168c940` and `0x1f030640`), and the second run saw
the probe's live-instance counter at 2. Comparing the cumulative reports supports equivalent
behavior for this duplicated-node run. The probe's own per-instance summary still prints
"node duplication equivalence: NOT TESTED" because it does not automate a cross-report
comparison; this evidence does not answer Flame process/restart lifetime, which remains a
0C question.

## Measured — Phase 0B controlled CUDA arena limit and CPU recovery

Flame **2026.2**, **2026-08-21**. The verbatim report is archived as
`docs/measurements/2026-08-21-ortprobe-gpu-mem-limit-flame.txt`. Baseline private-ORT CPU
and CUDA inference, numerical direction/identity, cross-thread cancellation and repeated
create/run/destroy all passed before the injected failure. The 480×640 CUDA steady median
was **41.2 ms**, consistent with the earlier multiresolution runs; its device-wide NVML
baseline/peak/steady values were 2,176.8/2,921.9/2,921.9 MiB.

The controlled exercise set `OrtCUDAProviderOptions::gpu_mem_limit` to **64 MiB** and
attempted the real SEA-RAFT M model at **480×640**. Session creation failed inside
`BFCArena::AllocateRawInternal`: the arena reported 1,931,264 bytes available for a
2,359,296-byte request. The probe therefore classified the result as
`stage CreateSession | kind allocator-limit`, exactly the explicit arena diagnostic required
by the gate. Device-wide NVML use was **2,395.9 MiB before and after cleanup**. After the
limited CUDA objects were released, a fresh CPU session passed the real model's 128×192
identity/direction checks, and `WHITEWATER_ORT_REQUIRE_GPU_MEM_LIMIT=1` reported **PASS**.

This closes the bounded CUDA-arena-limit/CPU-recovery measurement. `gpu_mem_limit` bounds
ORT's CUDA arena, not every CUDA/cuDNN allocation, so it is not evidence about physical
device-wide exhaustion. The fresh CPU session proves process/session recovery after teardown,
not an automatic production fallback or its artist-visible diagnostic; those remain Phase 4
shipping behavior to implement and verify.

## Measured — Phase 0B bounded high-resolution qualification

Flame **2026.2** on `flame6`, Rocky 9.5, **2026-08-21**, with private ORT **1.29.0** and the
pinned SEA-RAFT M export. A clean launch with none of the high-resolution environment
variables set first exercised the ordinary extended path. CPU/CUDA identity and direction,
cross-thread cancellation, CPU and CUDA lifecycle 3/3, provider-init failure followed by CPU
fallback, and the controlled 64 MiB arena-limit/CPU-recovery gate all passed. Its 480×640
steady medians were 6697.5 ms on CPU and 40.9 ms on CUDA, consistent with the earlier runs.

Three subsequent fresh launches selected exactly one GPU-only target and a **16 GiB ORT CUDA
arena ceiling**. All reached a classified `bounded-allocation-stop` during the warm `Run`:

| Target | Source H×W | Tensor H×W | Warm attempt | Arena available | Rejected request | Cleanup delta |
|---|---:|---:|---:|---:|---:|---:|
| UHD | 2160×3840 | 2160×3840 | 704.8 ms | 3,095,502,080 B (2.88 GiB) | 16,796,160,000 B (15.64 GiB) | +2.0 MiB |
| DCI 4K | 2160×4096 | 2160×4096 | 735.2 ms | 1,423,074,560 B (1.33 GiB) | 19,110,297,600 B (17.80 GiB) | +2.0 MiB |
| Alexa 35 open gate | 3164×4608 | 3168×4608 | 866.7 ms | 4,298,024,192 B (4.00 GiB) | 13,006,946,304 B (12.11 GiB) | +2.0 MiB |

The Alexa path correctly replication-padded four rows at the bottom and reported both source
and tensor dimensions. Its appended transcript retains an initial invalid-configuration
session followed by the corrected valid session.

None of the three produced a completed warm inference or any steady samples. The warm-attempt
durations are therefore time to allocation failure, not performance timings. The allocator
requests occurred at different fused matrix-multiplication nodes and are not comparable total
memory estimates. Likewise, the boundary-sampled NVML peaks only captured the small session
allocations before the rejected request; they are not estimates of high-resolution VRAM
demand. The +2 MiB session-cleanup deltas are clean bounded teardown evidence, not proof of
zero device-wide leakage because Flame and the shared ORT environment remained live.

The separate `HIGHRES QUALIFICATION VERDICT` correctly reports
`BOUNDED ALLOCATION STOP OBSERVED`, while the required *measurement-result* gate reports
**PASS** because a classified bounded stop is one of its deliberate valid outcomes. This
closes the higher-resolution 0B measurement negatively for the current 16 GiB arena
configuration. It neither imposes a plugin source-resolution cap nor predicts the fit of a
future GPU or a larger configurable arena budget. Raw reports:
[`clean control`](measurements/2026-08-21-ortprobe-highres-control-flame.txt),
[`UHD`](measurements/2026-08-21-ortprobe-highres-uhd-flame.txt),
[`DCI 4K`](measurements/2026-08-21-ortprobe-highres-dci4k-flame.txt), and
[`Alexa 35 open gate`](measurements/2026-08-21-ortprobe-highres-alexa35-flame.txt).

## Measured — Phase 0C item 1: Flame ST-map convention

Flame **2026.2** on `flame6`, **2026-08-21** (confirm the exact build string against the
capture session; all concurrent 0C-era runs are 2026.2). Method and assets in
[`tools/stprobe/`](../tools/stprobe/README.md); raw float-EXR renders archived in
[`docs/measurements/2026-08-21-stmap/`](measurements/2026-08-21-stmap/). Re-run
`python3 tools/stprobe/analyze_st_results.py docs/measurements/2026-08-21-stmap/` to reproduce.

Measured against **Flame's own consumers — the native ST Map node and Action's UV map** — not
a round trip through our resampler. A coordinate-encoded float plate (each pixel stores its own
normalized position) was warped by exactly-known UV maps and read back; the output values decode
directly to the source pixel Flame fetched. Nearest primary, Linear identity cross-check, at
PAR 1 and PAR 2.

**The convention (Action and the native ST Map node are identical on all of this):**

| Property | Measured value |
|---|---|
| Normalization | `U = (x + 0.5) / W`, `V = (y + 0.5) / H` — pixel centres at half-integer. Identity fit residual **0.000 px**, offset **−0.500**, slope **512.000/384.000**, both PARs |
| Origin | **Bottom-left** — V increases upward (identity un-flipped; `ST_SHIFTY` sampling `y+8` moves content down). Matches Flame's native origin from 0A |
| Channel layout | **U in R (horizontal), V in G (vertical)**, no swap (cross-channel fit residual ~148 px) |
| Semantics | The map value at an output pixel is the **source location to sample from** (backward map). `HALF` (U=V=0.5) fetches x≈255.5 = `0.5·W − 0.5`, confirming `(i+0.5)/N` and ruling out `x/W` and `x/(W−1)` |
| Basis magnitude | **Real-pixel**, PAR-invariant: the 8-px integer shift is 8 px at PAR 1 **and** PAR 2 |
| Out-of-range | **Differs by node.** Native ST Map node → **black**. Action → **mirror/reflect** at the boundary (−0.1→0.1, 1.1→0.9) |

**What this settles for the ST descriptor.** `stOrigin` default *Bottom Left* is correct and
native. Absolute-UV output must encode `(i+0.5)/N` against the source's **own real-pixel**
dimensions — this is the exact half-pixel convention `docs/plan.md` flagged as the ST risk, now
pinned to 0.000 px. `StMap.{h,cpp}` writes U→R, V→G. The two consumers disagree only outside
`[0, 1]`, so emitting in-range absolute UV is the safe target; both OOR behaviours are recorded
above for when a frame-edge motion pushes past the boundary. The depth half was already closed
(`MultipleClipDepths = 0` → float-only descriptor).

**One deliberately-open edge.** At PAR 2 with a full-frame source, image-bounds, RoD and
project-extent normalization coincide exactly 2:1, so this battery cannot separate them — it
only proves *real-pixel, image-own-dimension* normalization is correct and consistent. Telling
RoD from project extent apart would need an undersized or offset source; it matters only for a
source that does not fill the project, and is out of v1 scope. (The `ST_SHIFTX` global fit reads
a noisy slope for **Action** only — 511.3 rather than 512.0 — because Action's mirror-pad bends
the sub-zero edge columns into the least-squares; the OOR-free identity map is authoritative and
exact. The native node's shift fit is clean because its OOR→black pixels are excluded from the
fit.)

## Open

Phase 0A's five questions and all Phase 0B measurements are closed — see the measured sections
above. The pinned SEA-RAFT M export passes identity and direction through private ORT 1.29 on
CPU and CUDA; exact dependency closure, 480p–1080p timing/VRAM, lifecycle, cancellation,
provider fallback, duplicate-node behavior and bounded-recovery behavior are recorded. The
three full-resolution targets completed the required qualification measurement with bounded
allocation stops under the 16 GiB arena ceiling. That outcome does not choose a shipping
model or practical analysis cap, both of which remain later measured decisions.

### 0B — CLOSED 2026-08-21

There is no outstanding 0B item. Keep its operational checks in regression coverage. Flame
process/restart lifetime remains 0C; automatic production fallback and its artist-visible
diagnostic remain Phase 4.

### 0C — blocks ST map and cache integration

1. **The ST convention Flame's own downstream tool expects. — CLOSED 2026-08-21.** Measured
   `(x + 0.5) / W` half-pixel centres, bottom-left origin, U→R/V→G, real-pixel normalization,
   backward-map semantics; ST Map node blacks out-of-range while Action mirrors. Action and the
   native ST Map node are identical inside `[0, 1]`. See *Measured — Phase 0C item 1* above. The
   RoD-vs-project-extent normalization edge is deliberately left open (needs an undersized/offset
   source; out of v1 scope). The depth half was already answered: `MultipleClipDepths = 0`, so
   the ST descriptor declares float only.
2. **Instance and process lifetime.** Save and reload a Batch setup; switch away from the node
   and back; duplicate it; render foreground versus background/final; reopen Flame. This
   decides whether `Precache` has production value: a RAM-only precache is worthless if final
   render happens in another process. Distinguish "background render is another process" from
   "background render does not apply to this node type" — same practical effect, different
   implications later.

### Unblocking nothing in particular

3. **Whether render scale is ever anything but 1.** Every observation so far is `[1, 1]`,
   at both pixel aspect ratios.
4. **Whether `kOfxImagePropRowBytes` can be negative** in Flame, and whether images are ever
   windows into larger allocations. The vendored `HostImage` handles both; nothing has
   confirmed Flame exercises either.
5. **Re-run the tile check on an anamorphic clip** with the PAR fix in place, to confirm
   what the corrected arithmetic already shows: that those renders were full frames.

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
