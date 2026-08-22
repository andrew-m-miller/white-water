# Phase 1 Flame Linux smoke test — 2026-08-22

Host: Autodesk Flame 2026.2 on Linux. Artifact: EL8/glibc-2.28 GitHub Actions build from
`codex/phase-1` at `bf46351`. The checksum passed and older White Water builds were removed
before installation.

## Descriptor and panel

- Both product descriptors appeared, with the expected sockets.
- Both appeared under an unnamed OFX submenu instead of the `White Water` submenu used by
  WhiteWaterHostProbe and WhiteWaterOrtProbe. Screenshot retained with the test report.
- Labels were readable. Tooltips would improve later discoverability but are not a Phase 1
  requirement.
- Track exposed `Output` and `Insert At` choices.
- ST Map exposed `ST Mode` and `ST Origin` and no Track-only choices.
- Bake-off-owned model/analysis choices were absent as designed.
- Model Dir did not make either descriptor disappear.

## Reference time

Repeated Set Ref presses replaced Ref Frame with the latest value and did not animate it.
Flame passed Batch-relative OFX time: in a batch starting at 1001, frame 1001 arrived as 0 and
frame 1020 as 19. This confirms the previously reported Flame convention and the reason Set Ref
exists.

## Phase 1 fallback routing

- Composite left Source unchanged.
- Warped Insert / Current followed an animated Insert.
- Warped Insert / Reference held the selected reference frame.
- A disconnected Insert produced black on both colour and matte outputs.
- The connected Warped Insert RGB appeared full-frame rather than visually cut by its alpha.
  This is correct: the contract preserves unpremultiplied RGB and carries alpha separately.
  A follow-up with a nontrivial Insert Matte confirmed that the connected matte output reproduces
  the Insert matte.

## ST map and logs

- Default Absolute UV / Bottom Left output reproduced Source through Flame's native ST Map.
- A partial render showed no seam or incorrect normalization.
- The load report said `recognized as Flame family : yes`.
- No recovered describe/describeInContext messages appeared.
- Expected Phase 1 fallback messages appeared on renders.

## Disposition

The empty submenu is a product defect: `kOfxImageEffectPluginPropGrouping` was absent from both
permanent descriptors even though both probes set it to `White Water`. The fix sets the grouping
through a non-throwing property write and adds a raw-host assertion. Reconfirm the submenu label
with the replacement Linux artifact before merging Phase 1.
