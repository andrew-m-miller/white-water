#!/usr/bin/env python3
# White Water Phase 0C item 1 — ST-map convention analyzer.
#
# Consumes the EXRs rendered in Flame (node output of an ST/UV-map applied to
# WW_COORD.exr) and recovers Flame's convention numerically. Because WW_COORD
# stores each pixel's own normalized position, the output R,G ARE the source
# coordinate Flame fetched: decode is exact and hypothesis-free.
#
#   Expected inputs:  out_<node>_<par>_<map>.exr
#     node in {action, stmap}   par in {par1, par2}
#     map  in {identity, shiftx, shifty, half, oor}
#
# For each output we align, per output pixel, the AUTHORED normalized (U,V) the
# ST map stored (from ST_<MAP>.exr) against the FETCHED source pixel decoded
# from the output. A linear fit  x_fetched = a*U + b ,  yb_fetched = c*V + d
# then reads off:
#   a  -> denorm width  (~512 real vs ~1024 canonical at PAR 2 => normalization basis)
#   b  -> half-pixel offset  (-0.5 => (i+0.5)/N ;  0 => i/N or i/(N-1) per a)
#   c  -> sign gives ORIGIN  (+ => bottom-left, - => top-left/V-flip)
#   cross-channel fit -> detects U/V (R/G) swap
#
# Usage:  python3 analyze_st_results.py <dir-of-out_*.exr>
# Requires the OpenEXR Python binding (>=3.x):  python3 -m pip install OpenEXR

import glob
import json
import os
import re
import sys

import numpy as np
import OpenEXR

HERE = os.path.dirname(__file__)
ASSET_DIR = os.path.join(HERE, "assets")

MAP_FILE = {
    "identity": "ST_IDENTITY.exr",
    "shiftx": "ST_SHIFTX.exr",
    "shifty": "ST_SHIFTY.exr",
    "half": "ST_HALF.exr",
    "oor": "ST_OOR.exr",
}


def read_rgb(path):
    px = OpenEXR.File(path).channels()["RGB"].pixels  # (H, W, 3) float32
    return px[..., 0], px[..., 1], px[..., 2]


def load_manifest():
    with open(os.path.join(ASSET_DIR, "manifest.json")) as fh:
        return json.load(fh)


def fit_line(x, y):
    """Least-squares y = a*x + b over flattened valid samples. Returns a,b,resid_px."""
    A = np.vstack([x, np.ones_like(x)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.sqrt(np.mean((a * x + b - y) ** 2)))
    return float(a), float(b), resid


def analyze_one(path, man):
    W, H = man["width"], man["height"]
    name = os.path.basename(path)
    m = re.match(r"out_(\w+?)_(par[12])_(\w+)\.exr$", name)
    if not m:
        print(f"  !! cannot parse name: {name}")
        return None
    node, par, mp = m.group(1), m.group(2), m.group(3)
    if mp not in MAP_FILE:
        print(f"  !! unknown map '{mp}' in {name}")
        return None

    oR, oG, oB = read_rgb(path)
    if oR.shape != (H, W):
        print(f"  !! {name}: output raster {oR.shape} != source ({H},{W}); "
              f"tell Claude — needs coordinate-space alignment, not index alignment.")
        return None

    # decode fetched source pixel (PAR-independent: plate is real-pixel normalized)
    xf = oR * W - 0.5
    ybf = oG * H - 0.5
    sampled = np.abs(oB - 0.5) < 0.05   # B~0.5 => genuinely sampled from the plate

    aU, aV, _ = read_rgb(os.path.join(ASSET_DIR, MAP_FILE[mp]))  # authored U,V

    print(f"\n=== {name}  [node={node} par={par} map={mp}] ===")
    frac_sampled = float(np.mean(sampled))
    print(f"  sampled-from-plate fraction (B~0.5): {frac_sampled:.3f}")

    if mp == "half":
        v = sampled
        print(f"  HALF uniform check: xf range [{xf[v].min():.3f},{xf[v].max():.3f}] "
              f"ybf range [{ybf[v].min():.3f},{ybf[v].max():.3f}]")
        print(f"  => u=v=0.5 fetches x≈{np.median(xf[v]):.3f}, yb≈{np.median(ybf[v]):.3f} "
              f"(0.5*W-0.5={0.5*W-0.5:.1f}, 0.5*(W-1)={0.5*(W-1):.1f}, 0.5*W={0.5*W:.1f})")
        return dict(node=node, par=par, map=mp)

    if mp == "oor":
        third = W // 3
        for lbl, sl in (("left(U=-0.1)", slice(0, third)),
                        ("mid(identity)", slice(third, 2 * third)),
                        ("right(U=1.1)", slice(2 * third, W))):
            bandB = oB[:, sl]
            bandx = xf[:, sl]
            black = float(np.mean(np.abs(oB[:, sl]) < 0.05))
            print(f"  {lbl:16} meanB={float(bandB.mean()):.3f} "
                  f"black-frac={black:.3f} xf[min,med,max]="
                  f"[{bandx.min():.1f},{float(np.median(bandx)):.1f},{bandx.max():.1f}]")
        print("  => clamp: edge xf pins to 0 or W-1; wrap: xf wraps; black: B->0")
        return dict(node=node, par=par, map=mp)

    # linear maps: fit primary and cross channels over sampled pixels
    v = sampled
    if v.sum() < 100:
        print("  !! too few sampled pixels to fit")
        return None
    aUx, bUx, rUx = fit_line(aU[v], xf[v])        # U -> x   (expected primary)
    aVy, bVy, rVy = fit_line(aV[v], ybf[v])        # V -> yb  (expected primary)
    aVx, bVx, rVx = fit_line(aV[v], xf[v])        # V -> x   (cross; swap detector)
    aUy, bUy, rUy = fit_line(aU[v], ybf[v])        # U -> yb  (cross)

    print(f"  U->x : slope(a)={aUx:8.3f}  offset(b)={bUx:8.3f}  resid={rUx:.3f}px")
    print(f"  V->yb: slope(c)={aVy:8.3f}  offset(d)={bVy:8.3f}  resid={rVy:.3f}px")
    print(f"  cross V->x resid={rVx:.3f}px  U->yb resid={rUy:.3f}px  "
          f"(low cross resid => R/G swapped)")

    # interpretation hints
    hints = []
    if abs(aUx) > abs(aVx):
        hints.append("U(R) drives x, V(G) drives yb (no channel swap)")
    else:
        hints.append("CHANNEL SWAP: V(G) drives x — Flame reads G as U")
    hints.append(f"denorm width ~ {aUx:.1f} "
                 f"({'real 512' if abs(aUx-512)<abs(aUx-1024) else 'canonical 1024'} at this PAR)")
    hints.append(f"half-pixel offset ~ {bUx:.2f} "
                 f"({'(i+0.5)/N centers' if abs(bUx+0.5)<0.25 else 'i/N or i/(N-1)'})")
    hints.append(f"origin: {'BOTTOM-LEFT (V up)' if aVy > 0 else 'TOP-LEFT (V flipped)'}")
    for h in hints:
        print(f"    - {h}")
    return dict(node=node, par=par, map=mp, aUx=aUx, bUx=bUx, aVy=aVy,
                swap=abs(aUx) <= abs(aVx))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: analyze_st_results.py <dir-of-out_*.exr>")
        sys.exit(2)
    outdir = sys.argv[1]
    man = load_manifest()
    files = sorted(glob.glob(os.path.join(outdir, "out_*.exr")))
    if not files:
        print(f"no out_*.exr found in {outdir}")
        print("expected e.g. out_action_par1_identity.exr")
        sys.exit(1)

    print(f"analyzing {len(files)} result(s) against assets in {ASSET_DIR}")
    rows = []
    for f in files:
        r = analyze_one(f, man)
        if r:
            rows.append(r)

    # cross-file consistency summary for the linear maps
    print("\n================ SUMMARY ================")
    lin = [r for r in rows if "aUx" in r]
    for r in sorted(lin, key=lambda r: (r["node"], r["par"], r["map"])):
        print(f"  {r['node']:6} {r['par']:4} {r['map']:8}  "
              f"width~{r['aUx']:7.1f}  off~{r['bUx']:+.2f}  "
              f"origin={'BL' if r['aVy']>0 else 'TL'}  "
              f"{'SWAP' if r['swap'] else '    '}")
    print("\nCheck: action vs stmap agree? par1 vs par2 width slope "
          "(512 => real-pixel basis, 1024 => canonical/project-extent basis).")


if __name__ == "__main__":
    main()
