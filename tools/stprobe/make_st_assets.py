#!/usr/bin/env python3
# White Water Phase 0C item 1 — ST-map convention test asset generator.
#
# Generates a coordinate-encoded sample plate, a human-readable landmark plate,
# and a battery of ST (UV) maps with exactly-known float values. Applied inside
# Flame to the coordinate plate and rendered back to float EXR, these let us
# recover Flame's ST-map convention numerically (origin, half-pixel offset,
# normalization basis, channel layout, out-of-range behaviour) without ever
# round-tripping through our own resampler.
#
# Requires the OpenEXR Python binding (>=3.x):  python3 -m pip install OpenEXR
# The airgapped Flame box does NOT need this — assets are pre-generated and
# carried over. Only this generator and the analyzer need OpenEXR, and they run
# on the dev machine.
#
# Ground truth is written to assets/manifest.json so analyze_st_results.py and
# the eventual host-notes writeup stay consistent with what was actually baked.

import json
import os

import numpy as np
import OpenEXR

# --- geometry -----------------------------------------------------------------
# Real-pixel raster. Non-square (landscape) on purpose: any axis swap or
# transpose is then unmistakable. These are REAL pixels; at PAR 2 the same files
# are re-imported and Flame's canonical width becomes 2*W — which is exactly how
# we separate "normalize by real width" from "normalize by project extent".
W, H = 512, 384

SHIFT = 8          # integer pixel shift for the shift maps
OOR_LOW = -0.1     # out-of-range probe values
OOR_HIGH = 1.1

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Pixel coordinate grids in FILE space (EXR row 0 = top scanline).
# col x increases left->right; EXR row r increases top->bottom.
xs = np.arange(W, dtype=np.float32)
rs = np.arange(H, dtype=np.float32)
COL = np.broadcast_to(xs[None, :], (H, W))          # x per pixel
ROW = np.broadcast_to(rs[:, None], (H, W))          # EXR row per pixel
# from-bottom y index (our project convention is bottom-left origin)
YB = (H - 1) - ROW


def write_exr(name, r, g, b):
    """Write an RGB float32 scanline EXR (lossless ZIP)."""
    path = os.path.join(ASSET_DIR, name)
    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    rgb = np.stack(
        [np.ascontiguousarray(c, dtype=np.float32) for c in (r, g, b)], axis=-1
    )
    OpenEXR.File(header, {"RGB": rgb}).write(path)
    return path


def norm_x(x):
    """Normalized column under the bottom-left / half-pixel-center hypothesis."""
    return (x + 0.5) / W


def norm_yb(yb):
    """Normalized from-bottom row under the same hypothesis."""
    return (yb + 0.5) / H


def main():
    os.makedirs(ASSET_DIR, exist_ok=True)

    # --- WW_COORD: the measuring stick -------------------------------------
    # Each pixel stores its OWN normalized position. After Flame fetches from
    # some source location, the output R,G ARE that location's stored coords,
    # so we invert exactly:  x_fetched = R*W - 0.5 ,  yb_fetched = G*H - 0.5 .
    # This decode is how the file was BUILT (ground truth), not a hypothesis
    # about Flame, and it is PAR-independent (always real W,H).
    coord_r = norm_x(COL)
    coord_g = norm_yb(YB)
    coord_b = np.full((H, W), 0.5, np.float32)
    write_exr("WW_COORD.exr", coord_r, coord_g, coord_b)

    # --- WW_STSRC: human-readable landmark plate (optional sanity frame) ----
    r = np.full((H, W), 0.25, np.float32)
    g = np.full((H, W), 0.25, np.float32)
    b = np.full((H, W), 0.25, np.float32)
    # 1px red vertical line at x=100
    r[:, 100], g[:, 100], b[:, 100] = 1.0, 0.0, 0.0
    # 1px green horizontal line at from-bottom y=200  (EXR row H-1-200)
    row200 = (H - 1) - 200
    r[row200, :], g[row200, :], b[row200, :] = 0.0, 1.0, 0.0
    # white 8x8 in TRUE bottom-left corner (from-bottom rows 0..7)
    r[H - 8:H, 0:8], g[H - 8:H, 0:8], b[H - 8:H, 0:8] = 1.0, 1.0, 1.0
    # blue 8x8 in top-right corner (from-bottom rows H-8..H-1 => EXR rows 0..7)
    r[0:8, W - 8:W], g[0:8, W - 8:W], b[0:8, W - 8:W] = 0.0, 0.0, 1.0
    write_exr("WW_STSRC.exr", r, g, b)

    # --- ST maps: U in R, V in G (standard). Values are AUTHORED ground truth.
    ident_u = norm_x(COL)
    ident_v = norm_yb(YB)
    zeros = np.zeros((H, W), np.float32)

    # identity ramp across the full [0,1]x[0,1] domain
    write_exr("ST_IDENTITY.exr", ident_u, ident_v, zeros)

    # sample SHIFT px to the -x  => content moves +x if U=R and x-sign as assumed
    write_exr("ST_SHIFTX.exr", norm_x(COL - SHIFT), ident_v, zeros)

    # sample SHIFT px up in from-bottom terms => content moves +y (bottom-left)
    write_exr("ST_SHIFTY.exr", ident_u, norm_yb(YB + SHIFT), zeros)

    # constant U=V=0.5 => uniform output; decoded R,G reveal denorm of exactly 0.5
    half = np.full((H, W), 0.5, np.float32)
    write_exr("ST_HALF.exr", half, half, zeros)

    # three vertical bands: left U=-0.1, middle identity, right U=1.1; V identity
    third = W // 3
    oor_u = ident_u.copy()
    oor_u[:, 0:third] = OOR_LOW
    oor_u[:, 2 * third:] = OOR_HIGH
    write_exr("ST_OOR.exr", oor_u, ident_v, zeros)

    # --- manifest -----------------------------------------------------------
    manifest = {
        "width": W,
        "height": H,
        "origin_hypothesis": "bottom-left, half-pixel centers ((i+0.5)/N)",
        "coord_plate": {
            "file": "WW_COORD.exr",
            "R": "(x + 0.5) / W   (x = column, 0..W-1)",
            "G": "(yb + 0.5) / H  (yb = row from BOTTOM, 0..H-1)",
            "B": 0.5,
            "decode_fetched_pixel": {
                "x": "R * W - 0.5",
                "yb": "G * H - 0.5",
                "note": "PAR-independent; always real W,H. B!=0.5 => not sampled from plate (black/wrap).",
            },
        },
        "st_channel_layout_authored": "U in R, V in G, B=0",
        "shift_px": SHIFT,
        "oor_values": {"low": OOR_LOW, "high": OOR_HIGH, "bands": "left/mid/right thirds of width"},
        "maps": {
            "ST_IDENTITY": "U=(x+0.5)/W, V=(yb+0.5)/H",
            "ST_SHIFTX": "U=((x-8)+0.5)/W, V=identity",
            "ST_SHIFTY": "U=identity, V=((yb+8)+0.5)/H",
            "ST_HALF": "U=V=0.5",
            "ST_OOR": "U: left=-0.1, mid=identity, right=1.1; V=identity",
        },
        "landmark_plate": {
            "file": "WW_STSRC.exr",
            "bg": 0.25,
            "red_vline_x": 100,
            "green_hline_yb": 200,
            "white_block": "8x8 true bottom-left",
            "blue_block": "8x8 top-right",
        },
        "expected_output_naming": "out_<node>_<par>_<map>.exr  node in {action,stmap} par in {par1,par2} map in {identity,shiftx,shifty,half,oor}",
    }
    with open(os.path.join(ASSET_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"wrote assets to {ASSET_DIR}:")
    for n in sorted(os.listdir(ASSET_DIR)):
        p = os.path.join(ASSET_DIR, n)
        print(f"  {n:20} {os.path.getsize(p):>8} bytes")


if __name__ == "__main__":
    main()
