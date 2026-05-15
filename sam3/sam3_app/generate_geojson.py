#!/usr/bin/env python3
"""
generate_geojson.py — Produce a geo-metadata JSON for a PNG cut of an orthomosaic.

The JSON carries all spatial information that a PNG cannot store:
  • CRS (EPSG code)
  • Pixel size (ground resolution in CRS units)
  • Origin (top-left corner in CRS coordinates)
  • Full affine transform (supports rotated / north-up grids)
  • Image dimensions
  • Source orthomosaic path (optional, for traceability)

Usage examples
--------------

1. Derive from the original GeoTIFF orthomosaic automatically:

   python generate_geojson.py \
       --tiff  /data/ortho_full.tif \
       --png   /data/cuts/tile_003.png \
       --out   /data/cuts/tile_003_geo.json

   This reads the affine transform from the GeoTIFF, converts pixel offset
   of the cut to real-world coordinates, and writes the JSON.

2. Specify everything manually (no GeoTIFF required):

   python generate_geojson.py \
       --png        /data/cuts/tile_003.png \
       --crs        EPSG:32721 \
       --pixel-size 0.05 \
       --origin     358200.0 6542800.0 \
       --out        /data/cuts/tile_003_geo.json

3. If the cut was made with a pixel offset from the original orthomosaic
   (e.g. row 2000, col 4000 in the full raster):

   python generate_geojson.py \
       --tiff       /data/ortho_full.tif \
       --png        /data/cuts/tile_003.png \
       --row-offset 2000 \
       --col-offset 4000 \
       --out        /data/cuts/tile_003_geo.json

Output format (tile_003_geo.json)
----------------------------------
{
  "image_name":   "tile_003.png",
  "width":        1024,
  "height":       1024,
  "crs":          "EPSG:32721",
  "origin_x":     358200.0,      // top-left X in CRS units
  "origin_y":     6542800.0,     // top-left Y in CRS units
  "pixel_size_x": 0.05,          // metres per pixel, X direction
  "pixel_size_y": -0.05,         // metres per pixel, Y direction (negative = north-up)
  "transform": [a, b, c, d, e, f] // full affine: [scale_x, shear_x, origin_x,
                                  //               shear_y, scale_y, origin_y]
  "source_tiff":  "/data/ortho_full.tif"  // optional
}
"""

import argparse
import json
import os
import sys

import PIL.Image


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_tiff_meta(tiff_path: str):
    """Return (crs_string, affine_6tuple) from a GeoTIFF using rasterio."""
    try:
        import rasterio
    except ImportError:
        sys.exit("rasterio is required to read GeoTIFFs.  Install with:\n"
                 "  pip install rasterio")
    with rasterio.open(tiff_path) as ds:
        crs = ds.crs.to_string() if ds.crs else "EPSG:4326"
        t   = ds.transform          # affine.Affine
        # Affine order: (a=scale_x, b=shear_x, c=origin_x,
        #                d=shear_y, e=scale_y, f=origin_y)
        affine6 = [t.a, t.b, t.c, t.d, t.e, t.f]
    return crs, affine6


def offset_affine(affine6, row_offset: int, col_offset: int):
    """Shift an affine transform by (col_offset, row_offset) pixels."""
    a, b, c, d, e, f = affine6
    # New origin = old_origin + col_offset * (a,d) + row_offset * (b,e)
    new_c = c + col_offset * a + row_offset * b
    new_f = f + col_offset * d + row_offset * e
    return [a, b, new_c, d, e, new_f]


def affine_from_manual(origin_x: float, origin_y: float,
                       pixel_size_x: float, pixel_size_y: float):
    """Build a north-up (no rotation) affine from simple params."""
    # a=scale_x  b=shear_x  c=origin_x
    # d=shear_y  e=scale_y  f=origin_y
    return [pixel_size_x, 0.0, origin_x, 0.0, pixel_size_y, origin_y]


def generate_png_tiles_from_tiff(tiff_path, out_dir, tile_size=1024):
    import math
    import rasterio
    from rasterio.windows import Window
    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(tiff_path) as ds:
        crs = ds.crs.to_string()
        base_transform = ds.transform

        # Compute grid dimensions up front so every JSON carries them.
        tiles_per_row = math.ceil(ds.width  / tile_size)
        tiles_per_col = math.ceil(ds.height / tile_size)

        tile_id = 0

        for tile_row, row in enumerate(range(0, ds.height, tile_size)):
            for tile_col, col in enumerate(range(0, ds.width, tile_size)):

                window = Window(
                    col,
                    row,
                    min(tile_size, ds.width - col),
                    min(tile_size, ds.height - row)
                )

                transform = rasterio.windows.transform(window, base_transform)

                data = ds.read(window=window)

                # Convert to HWC
                if data.shape[0] >= 3:
                    img = np.transpose(data[:3], (1, 2, 0))
                else:
                    img = data[0]

                # Normalize
                img = img.astype(np.float32)
                img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = img.astype(np.uint8)

                png_name = f"tile_{tile_id:05d}.png"
                png_path = os.path.join(out_dir, png_name)

                Image.fromarray(img).save(png_path)

                affine6 = [
                    transform.a, transform.b, transform.c,
                    transform.d, transform.e, transform.f
                ]

                height, width = img.shape[:2]

                out = {
                    "image_name":    png_name,
                    "width":         width,
                    "height":        height,
                    "crs":           crs,
                    "origin_x":      transform.c,
                    "origin_y":      transform.f,
                    "pixel_size_x":  transform.a,
                    "pixel_size_y":  transform.e,
                    "transform":     affine6,
                    "source_tiff":   os.path.abspath(tiff_path),
                    "tile_row":      tile_row,
                    "tile_col":      tile_col,
                    "tiles_per_row": tiles_per_row,
                    "tiles_per_col": tiles_per_col,
                    "tile_id":       tile_id,
                }

                json_path = os.path.splitext(png_path)[0] + "_geo.json"
                with open(json_path, "w") as fp:
                    json.dump(out, fp, indent=2)

                print(f"\u2713 Tile {tile_id} (row {tile_row}, col {tile_col}) done")

                tile_id += 1

# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a geo-metadata JSON for a PNG cut of an orthomosaic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--png", help="Path to the PNG cut")
    ap.add_argument("--out",   default=None,  help="Output JSON path (default: <png>.geo.json)")

    # ── Mode A: derive from GeoTIFF
    ap.add_argument("--tiff",       default=None, help="Source GeoTIFF orthomosaic")
    ap.add_argument("--row-offset", type=int, default=0,
                    help="Pixel row of the cut's top-left corner in the full raster (default 0)")
    ap.add_argument("--col-offset", type=int, default=0,
                    help="Pixel column of the cut's top-left corner in the full raster (default 0)")

    # ── Mode B: manual
    ap.add_argument("--crs",        default="EPSG:4326",
                    help="CRS as EPSG string, e.g. EPSG:32721 (default: EPSG:4326)")
    ap.add_argument("--origin",     nargs=2, type=float, metavar=("X", "Y"),
                    default=None,
                    help="Top-left corner in CRS units: --origin 358200.0 6542800.0")
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="Ground resolution in CRS units per pixel (positive number; "
                         "Y direction will be negated automatically for north-up images)")
    
    # 
    ap.add_argument("--tile", action="store_true",
                help="Generate PNG tiles + JSONs from GeoTIFF")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--out-dir", default="tiles_output")

    args = ap.parse_args()

    if args.tile:
        if not args.tiff:
            sys.exit("--tile requires --tiff")

        generate_png_tiles_from_tiff(
            args.tiff,
            args.out_dir,
            args.tile_size
        )
        return

    # Enforce PNG only if NOT in tile mode
    if not args.png:
        ap.error("--png is required unless using --tile mode")

    # ── Read PNG dimensions ──────────────────────────────────────────────────
    if not os.path.isfile(args.png):
        sys.exit(f"PNG not found: {args.png}")
    with PIL.Image.open(args.png) as img:
        width, height = img.size  # PIL: (width, height)

    # ── Build affine ─────────────────────────────────────────────────────────
    source_tiff = None
    if args.tiff:
        if not os.path.isfile(args.tiff):
            sys.exit(f"TIFF not found: {args.tiff}")
        crs, affine6 = load_tiff_meta(args.tiff)
        if args.row_offset or args.col_offset:
            affine6 = offset_affine(affine6, args.row_offset, args.col_offset)
        source_tiff = os.path.abspath(args.tiff)
    else:
        # Manual mode — require --origin and --pixel-size
        missing = []
        if args.origin is None:     missing.append("--origin")
        if args.pixel_size is None: missing.append("--pixel-size")
        if missing:
            ap.error(f"Without --tiff you must supply: {', '.join(missing)}")
        crs = args.crs
        ox, oy = args.origin
        ps = args.pixel_size
        # North-up convention: pixel_size_y is negative (top→bottom = decreasing Y)
        affine6 = affine_from_manual(ox, oy, ps, -ps)

    a, b, c, d, e, f = affine6

    # ── Compose output dict ──────────────────────────────────────────────────
    out = {
        "image_name":   os.path.basename(args.png),
        "width":        width,
        "height":       height,
        "crs":          crs,
        "origin_x":     c,          # top-left X
        "origin_y":     f,          # top-left Y
        "pixel_size_x": a,          # metres (or CRS units) per pixel, X
        "pixel_size_y": e,          # metres per pixel, Y (negative = north-up)
        "transform":    affine6,    # [a, b, c, d, e, f] full affine
    }
    if source_tiff:
        out["source_tiff"] = source_tiff

    # ── Write JSON ───────────────────────────────────────────────────────────
    out_path = args.out or (os.path.splitext(args.png)[0] + "_geo.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(f"✓  Wrote {out_path}")
    print(f"   CRS          : {crs}")
    print(f"   Origin (X,Y) : ({c:.6f}, {f:.6f})")
    print(f"   Pixel size X : {a:.8f}  (positive → west-to-east)")
    print(f"   Pixel size Y : {e:.8f}  (negative = north-up, top row has highest Y)")
    print(f"   Shear  (b,d) : ({b:.6f}, {d:.6f})")
    print(f"   Image size   : {width} × {height} px")
    print()
    print(f"   Top-left  corner : ({c:.4f}, {f:.4f})")
    print(f"   Top-right corner : ({c + width*a:.4f}, {f + width*d:.4f})")
    print(f"   Bot-left  corner : ({c + height*b:.4f}, {f + height*e:.4f})")
    print(f"   Bot-right corner : ({c + width*a + height*b:.4f}, {f + width*d + height*e:.4f})")
    print()
    print("   ⚠  Verify these corners match your GIS before exporting GPKG.")


if __name__ == "__main__":
    main()


#python generate_geojson.py \\n    --tiff "/Users/catherinelouys/Downloads/quebec_trees_dataset_2021-09-02/2021-09-02/zone3 copy/2021-09-02-sbl-z3-rgb-cog.tif" \\n    --tile \\n    --tile-size 1024 \\n    --out-dir /Users/catherinelouys/Downloads/imports
