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

## Open — Phase 0

Nothing in `docs/plan.md` past Phase 0 is worth building until these are answered on the
actual Rocky 9 Flame box. Items 2 and 3 are the project's real risk: either one failing
changes the architecture, not the schedule.

### 1. General context with two clips

Define a mandatory `Source` and an optional `Insert`, both RGBA.

- Do both appear in the Batch node, and as how many sockets? (A forum report says Flame
  splits an RGBA clip into Front + Matte sockets, which would make two clips four sockets.
  Unverified.)
- What does `clipGetImage` return on the disconnected optional clip — a null handle, an
  error, or a black image?
- Does `kOfxImageClipPropConnected` read correctly on it?

### 2. `clipGetImage` at arbitrary times *during* the render action

**The load-bearing assumption of the whole design.** warp-drive measured out-of-render
frame access from an instance-changed action, which is a different situation: the host was
idle and waiting on us. Pulling a frame from *inside* a render, on a host thread, while the
host holds whatever locks a render holds, is not the same question.

Test offsets ±1, ±10, ±100 and past both ends of the clip. Record the status code, whether
the returned image has plausible content, and whether the host deadlocks or complains.

**If this fails:** the on-demand chain is impossible. The design becomes a mandatory
`Analyze` pass that pulls frames from an instance-changed action (which is measured to
work) and writes a disk-backed cache the render reads.

### 3. ONNX Runtime inside Flame's process

Ship a second, separate bundle that links ONNX Runtime and runs a trivial model on a
button press. Separate so that a runtime which refuses to initialise cannot take the
capability probe down with it.

- Does `Ort::Env` construct at all inside Flame?
- ~~Does the CUDA execution provider find a device, given that Flame already owns one?~~
  **Answered 2026-08-20** — see *Measured — Mocha Pro's ML architecture*. A second process
  holds 4.9 GB alongside a running Flame. Device contention is not the problem.
- Does Flame's own rendering degrade while inference runs?
- Does anything collide at the dynamic-linker level — Flame ships its own CUDA runtime and
  quite possibly its own protobuf. (`cmake/ofx.map` keeps *our* symbols private; it does
  not stop us from binding to *theirs*.)
- On macOS: does the CoreML EP initialise, and how much of a RAFT graph does it actually
  take rather than falling back to CPU?

**If this fails:** inference is CPU-only, which makes RAFT at 4K unusable and promotes RIFE
from the fast option to the only option.

#### Prior evidence: Mocha Pro does ML matting in an OFX plugin

*Reported 2026-08-19 from regular production use, not instrumented.* Boris FX Mocha Pro
ships an OFX plugin whose ML matte feature is believed to run GPU inference, and it is used
routinely on this facility's boxes. That is the strongest evidence available short of
measuring: an ML model doing real work from inside an OFX plugin, on this hardware, in
production.

It raises the prior considerably. It does **not** close this item, for one specific reason
worth stating before anyone treats it as settled.

**Mocha Pro's OFX plugin launches its own separate application process.** That is its
defining architectural quirk — the plugin is a shim and the actual Mocha interface is a
standalone program. So the inference may well be happening in *Mocha's* process, with its
own address space, its own CUDA context and its own runtime, and not inside the host's
process at all. If so it is evidence for a **different** architecture than the one this item
is asking about, and the fact that it works says nothing about whether ONNX Runtime can get
a CUDA device inside Flame's own process while Flame holds one.

Two further unknowns: which backend it uses (CUDA, OpenVINO and CPU are all plausible; Boris
FX has shipped OpenVINO elsewhere), and which host the observation was made in.

#### Cheapest way to answer this: inspect Mocha's bundle before building anything

This is worth doing **first**, on the box, before the ONNX Runtime probe bundle exists. It
costs minutes and it may be decisive.

**Run these under `bash`.** Measured 2026-08-20: the Flame box's interactive shell is
**tcsh**, which is traditional in Flame environments and is not what any of this is written
for. tcsh does not understand `2>` — it reads the `2` as an argument and `>/dev/null` as a
*stdout* redirect, so a command that also pipes has two destinations for stdout and dies
with `Ambiguous output redirect.` A `<` or `>` with no filename after it gives
`Missing name for redirect.` instead. Neither message mentions the shell, which is what
makes this cost twenty minutes rather than one.

```bash
bash
```

Then run the rest as written. No placeholders to substitute — an earlier revision used
`<the .ofx>` as a stand-in, which fails for the second reason above even under bash.

**1. What ships inside the bundle.** The cheapest signal there is: a `libonnxruntime.so` or
`libopenvino.so` sitting in the payload names the backend without running anything.

```bash
find /usr/OFX/Plugins /usr/local/OFX/Plugins /opt -ipath '*mocha*' \( -name '*.so*' -o -name '*.ofx' -o -name '*.dat' \) -printf '%10s  %p\n' 2>/dev/null | sort -k2
```

**2. Does the bundle carry a separate executable?** This is the in-process question,
answered directly — Mocha Pro's plugin is known to launch its own application, and finding
that binary here is what confirms where inference could be happening.

```bash
find /usr/OFX/Plugins /usr/local/OFX/Plugins /opt -ipath '*mocha*' -type f -perm -u+x -not -name '*.so*' -not -name '*.ofx' 2>/dev/null
```

**3. What the OFX binary actually links.** `LD_LIBRARY_PATH` is cleared so `ldd` resolves
only through the image's own `RPATH`/`RUNPATH`, which is what the host will do. Note that a
negative result here is not conclusive: a plugin may `dlopen` a runtime rather than link it.

```bash
find /usr/OFX/Plugins /usr/local/OFX/Plugins /opt -ipath '*mocha*' -name '*.ofx' -exec sh -c 'echo "== $1"; env -u LD_LIBRARY_PATH ldd "$1" 2>/dev/null | grep -iE "cuda|onnx|openvino|torch|cudnn|tensorrt" || echo "   no ML runtime linked directly (may still dlopen one)"' _ {} \; 2>/dev/null
```

**4. Who holds the VRAM.** Then, with Flame open and Mocha's ML matte actually running:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Read it like this.** If `nvidia-smi` shows the *Flame* pid holding the extra VRAM, ML
inference runs in-process in Flame and this item is close to answered in our favour. If it
shows a separate Mocha process, the evidence supports out-of-process inference instead —
which is still useful, because warp-drive already has a proven fork/exec supervisor for
exactly that shape (`src/ofx/EditorProcess.cpp`, with environment scrubbing), and it would
become the fallback architecture rather than a rewrite.

### 4. Does `setSupportsTiles(false)` yield whole-frame render windows?

Flame reports `SupportsTiles = 1` as a host. A plugin declaring `false` is legal OFX and
the host must then hand over the full region of definition. Optical flow is inherently
whole-frame, so this is worth a lot — but if Flame ignores it and tiles anyway, the render
path needs the tile-assembly logic warp-drive has and this plan currently does not.

### 5. `getFramesNeeded` honesty

Declaring only `{N}` while pulling `{R..N}` with `clipGetImage` is out of contract. The
alternative — declaring the true range — risks Flame materialising hundreds of upstream
frames for one output frame.

Measure both: does declaring only `{N}` still let the pulls in item 2 succeed, and what
does Flame actually do when a long range *is* declared?

### Procedure

Build the probe bundle, install it, and in a **Batch node** (repeat on a **timeline soft
effect**, and ideally once on an **anamorphic clip**):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cp -R build/bundles/WhiteWaterHostProbe.ofx.bundle /usr/OFX/Plugins/
```

Apply **White Water → White Water Host Probe**, press **Run Probe**, then move the playhead
and scrub. The report goes to `$WHITEWATER_PROBE_LOG`, else `$TMPDIR/whitewater-hostprobe.txt`,
else `/tmp/whitewater-hostprobe.txt`. Flame's own copy of the plugin's stderr is in
`/opt/Autodesk/log/`.

Paste the results into this file under a **Measured** heading with the host build and date,
and move each closed item out of the Open section.
