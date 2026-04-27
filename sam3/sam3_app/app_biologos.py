"""
SAM3 Interactive Segmentation – Flask backend
Run:  python app.py
Then open http://localhost:5000 in your browser.
"""

import base64
import io
import os
import traceback
import datetime
import torch
import cv2
import geopandas as gpd
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import csv
from flask import Flask, jsonify, render_template, request
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure
import matplotlib
_cmap = matplotlib.colormaps["tab10"]
def plt_tab10(i): return _cmap(i)

def _binary_to_geom(binary: np.ndarray):
    """
    Convert a boolean H×W mask to a Shapely geometry (Polygon or MultiPolygon)
    that correctly represents exterior rings AND holes.

    Strategy: use cv2.findContours with RETR_CCOMP which returns a two-level
    hierarchy — level-0 are exterior contours, level-1 are holes inside them.
    We pair each hole with its parent exterior to build proper Shapely Polygons
    with interior rings, then union everything into a single geometry.
    """
    uint8 = binary.astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]  # shape (N, 4): next, prev, firstChild, parent
    polys = []
    for idx, (cnt, hier) in enumerate(zip(contours, hierarchy)):
        parent = hier[3]
        if parent != -1:
            continue  # this is a hole — handled when we process its parent exterior
        if len(cnt) < 4:
            continue
        exterior = [(float(pt[0][0]), float(pt[0][1])) for pt in cnt]
        if len(exterior) < 4:
            continue

        # Collect holes: children of this exterior contour
        holes = []
        child = hier[2]  # firstChild index
        while child != -1:
            hole_cnt = contours[child]
            if len(hole_cnt) >= 4:
                hole = [(float(pt[0][0]), float(pt[0][1])) for pt in hole_cnt]
                if len(hole) >= 4:
                    holes.append(hole)
            child = hierarchy[child][0]  # next sibling

        try:
            p = Polygon(exterior, holes)
            if not p.is_valid:
                p = p.buffer(0)  # fix self-intersections
            if p.is_valid and not p.is_empty:
                polys.append(p)
        except Exception:
            pass

    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return unary_union(polys)



def _parse_affine(geo: dict):
    """
    Parse a geo-metadata dict into affine coefficients (a, b, c, d, e, f) where:
        X = c + col*a + row*b
        Y = f + col*d + row*e

    Priority order:
      1. Named keys: pixel_size_x/y, origin_x/y, rotation_x/y (unambiguous)
      2. "transform_type": "gdal" | "rasterio"  +  "transform" list (explicit)
      3. 9-element list  [a,b,c,d,e,f,0,0,1]  (rasterio list(Affine))
      4. 6-element list  — heuristic: GDAL if |t[0]| > |t[1]|*10, else rasterio
    Returns (a, b, c, d, e, f, crs, convention_str).
    """
    crs = geo.get("crs", "EPSG:4326")
    tf  = geo.get("transform")

    # 1. Named keys — always unambiguous
    if "pixel_size_x" in geo or "origin_x" in geo:
        a = float(geo.get("pixel_size_x", 1.0))
        e = float(geo.get("pixel_size_y", -1.0))
        b = float(geo.get("rotation_x",   0.0))
        d = float(geo.get("rotation_y",   0.0))
        c = float(geo.get("origin_x",     0.0))
        f = float(geo.get("origin_y",     0.0))
        return a, b, c, d, e, f, crs, "named-keys"

    if not tf:
        return 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, crs, "identity"

    t  = [float(v) for v in tf]
    tt = geo.get("transform_type", "").lower().strip()

    # 2. Explicit transform_type
    if tt in ("gdal", "gdal_geotransform") and len(t) >= 6:
        c, a, b, f, d, e = t[:6]
        return a, b, c, d, e, f, crs, "gdal (explicit)"
    if tt in ("rasterio", "affine") and len(t) >= 6:
        a, b, c, d, e, f = t[:6]
        return a, b, c, d, e, f, crs, "rasterio (explicit)"

    # 3. 9-element rasterio list(Affine)
    if len(t) == 9:
        a, b, c = t[0], t[1], t[2]
        d, e, f = t[3], t[4], t[5]
        return a, b, c, d, e, f, crs, "rasterio-9"

    # 4. 6-element heuristic
    if len(t) >= 6:
        abs0 = abs(t[0])
        abs1 = abs(t[1]) if t[1] != 0 else None
        is_gdal = (abs0 > abs1 * 10) if abs1 is not None else (abs0 > 10.0)
        if is_gdal:
            c, a, b, f, d, e = t[:6]
            return a, b, c, d, e, f, crs, "gdal (heuristic)"
        else:
            a, b, c, d, e, f = t[:6]
            return a, b, c, d, e, f, crs, "rasterio (heuristic)"

    return 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, crs, "identity (fallback)"

def _geom_with_transform(geom, px_to_geo):
    """Apply a pixel→geo affine transform to every coordinate in a geometry."""
    from shapely.geometry import mapping, shape
    import copy

    def transform_coords(coords):
        return [px_to_geo(x, y) for x, y in coords]

    def transform_poly(p):
        exterior = transform_coords(list(p.exterior.coords))
        holes    = [transform_coords(list(ring.coords)) for ring in p.interiors]
        return Polygon(exterior, holes)

    if geom.geom_type == 'Polygon':
        return transform_poly(geom)
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([transform_poly(p) for p in geom.geoms])
    return geom


# ── SAM3 ──────────────────────────────────────────────────────────────────────
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path  = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
weights_path = f"{sam3_root}/sam3_app/model/sam3.pt"
print("Loading SAM3 model…")
model     = build_sam3_image_model(bpe_path=bpe_path,load_from_HF = False, checkpoint_path=weights_path)
processor = Sam3Processor(model)
print("Model ready.")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

STATE = {
    "processor_state": None,
    "image_array":     None,
    "image_name":      "sam3_image",
    "image_w":         0,
    "image_h":         0,
    "species_list":    [],
    "mask_registry":   [],
    "prompted_boxes":  [],
    "confidence":      0.5,
    "edited_masks":    {},   # {mask_id: np.ndarray bool H×W}
    "deleted_indices": set(), # SAM tensor indices permanently removed by user
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _array_to_b64(arr: np.ndarray) -> str:
    img = PIL.Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get_mask_binary(i, ps) -> np.ndarray:
    """Return binary mask for index i, preferring edited version."""
    if i in STATE["edited_masks"]:
        return STATE["edited_masks"][i]
    return ps["masks"][i][0].cpu().numpy() > 0.5


def _render_overlay() -> dict:
    ps = STATE["processor_state"]
    if ps is None or "masks" not in ps:
        # If update_mask already populated the registry (e.g. during exportAll
        # without a live SAM state), preserve it and return a no-op response.
        if STATE["mask_registry"]:
            return {"overlay_b64": None, "mask_list": [], "status": "Registry-only mode"}
        return {"overlay_b64": None, "mask_list": [], "status": "No image loaded"}

    masks  = ps.get("masks",  [])
    boxes  = ps.get("boxes",  [])
    scores = ps.get("scores", [])
    H, W   = STATE["image_h"], STATE["image_w"]

    overlay  = np.zeros((H, W, 4), dtype=np.float32)

    # Build spatial index of previous assignments for IoU-based carry-forward.
    # Key: old registry entry with box [x0,y0,x1,y1] and species_id.
    # We match each new mask to the old mask with highest box-IoU and transfer
    # the species if IoU >= 0.3.  This is robust against index shifts.
    prev_assigned = [m for m in STATE["mask_registry"] if m.get("species_id") is not None]

    def _box_iou(a, b):
        """IoU of two [x0,y0,x1,y1] boxes."""
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter == 0:
            return 0.0
        aA = (a[2] - a[0]) * (a[3] - a[1])
        bA = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aA + bA - inter)

    STATE["mask_registry"] = []
    # Dict-keyed by mask_id so skipped masks never corrupt indexing.
    mask_b64_by_id: dict = {}

    # Track claimed pixels so masks never overlap (first mask wins each pixel)
    claimed = np.zeros((H, W), dtype=bool)

    for i, (mask_t, box_t, score) in enumerate(zip(masks, boxes, scores)):
        # Skip any mask the user has permanently deleted this session
        if i in STATE["deleted_indices"]:
            continue
        raw_binary = _get_mask_binary(i, ps)

        if i in STATE["edited_masks"]:
            # Edited mask: honour the user's exact pixel selection.
            # Clip against already-claimed pixels for the visual composite,
            # but mark the *full* edited region as claimed so later unedited
            # masks cannot bleed into pixels the user explicitly painted.
            binary = STATE["edited_masks"][i] & ~claimed
            claimed |= STATE["edited_masks"][i]
        else:
            # Unedited mask: standard first-mask-wins overlap removal.
            binary = raw_binary & ~claimed
            claimed |= binary

        # Use the clipped binary for the polygon so exported geometries never intersect.
        poly = _binary_to_geom(binary)

        color     = np.array(plt_tab10(i % 10)[:3])
        color_int = (color * 255).astype(int).tolist()
        overlay[binary] = [*color, 0.55]

        # Skip empty masks (overlap removal can wipe a mask entirely).
        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue

        # Skip masks with no valid polygon (degenerate shape)
        if poly is None:
            continue

        # Skip near-empty polygons (< 4 px² area) — SAM sometimes hallucinates
        # tiny blobs or bounding-box rectangles outside the prompt region.
        if poly.area < 4:
            continue

        # Per-mask grayscale PNG keyed by mask_id (not list position) so the
        # final lookup is always O(1) and immune to skipped-mask off-by-one errors.
        buf = io.BytesIO()
        PIL.Image.fromarray((binary.astype(np.uint8) * 255), "L").save(buf, "PNG")
        mask_b64_by_id[i] = base64.b64encode(buf.getvalue()).decode()

        bx0, bx1 = int(xs.min()), int(xs.max())
        by0, by1 = int(ys.min()), int(ys.max())

        new_box = [bx0, by0, bx1, by1]

        # Carry forward species assignment from the spatially closest previous mask.
        carried_species = None
        best_iou = 0.0
        for pm in prev_assigned:
            iou = _box_iou(new_box, pm["box"])
            if iou > best_iou:
                best_iou = iou
                carried_species = pm["species_id"]
        if best_iou < 0.3:
            carried_species = None

        STATE["mask_registry"].append({
            "mask_id":    i,
            "polygon":    poly,
            "species_id": carried_species,
            "box":        new_box,
            "score":      float(score),
            "color":      color_int,
            "binary":     binary,
        })

    overlay_b64 = _array_to_b64((overlay * 255).astype(np.uint8))
    mask_list = [
        {
            "mask_id":    m["mask_id"],
            "species_id": m["species_id"],
            "box":        m["box"],
            "score":      m["score"],
            "color":      m["color"],
            "centroid":   [m["polygon"].centroid.x, m["polygon"].centroid.y],
            "mask_b64":   mask_b64_by_id[m["mask_id"]],
        }
        for m in STATE["mask_registry"]
    ]
    return {"overlay_b64": overlay_b64, "mask_list": mask_list,
            "width": W, "height": H, "status": f"Found {len(STATE['mask_registry'])} object(s)"}


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_image():
    try:
        import json as _json
        f         = request.files.get("image")
        json_file = request.files.get("json")
        # Always set geo from the JSON in this request, or clear it.
        # Never inherit geo from a previous image (breaks folder-mode tile transforms).
        if json_file:
            STATE["geo"] = _json.load(json_file)
        else:
            STATE["geo"] = None
        if f is None:
            return jsonify({"error": "No file"}), 400
        STATE["image_name"]      = os.path.splitext(f.filename)[0] or "sam3_image"
        pil_img                  = PIL.Image.open(f.stream).convert("RGB")
        STATE["image_array"]     = np.array(pil_img)
        STATE["image_w"], STATE["image_h"] = pil_img.size
        STATE["processor_state"] = processor.set_image(pil_img)
        STATE["prompted_boxes"]  = []
        STATE["mask_registry"]   = []
        STATE["edited_masks"]    = {}
        STATE["deleted_indices"] = set()
        return jsonify({"image_b64": _array_to_b64(STATE["image_array"]),
                        "width": STATE["image_w"], "height": STATE["image_h"],
                        "name": STATE["image_name"]})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/load_url", methods=["POST"])
def load_url():
    try:
        import requests as req
        url = request.json.get("url", "").strip()
        if not url:
            return jsonify({"error": "No URL"}), 400
        r = req.get(url, timeout=15); r.raise_for_status()
        pil_img                  = PIL.Image.open(io.BytesIO(r.content)).convert("RGB")
        STATE["image_name"]      = "url_image"
        STATE["image_array"]     = np.array(pil_img)
        STATE["image_w"], STATE["image_h"] = pil_img.size
        STATE["processor_state"] = processor.set_image(pil_img)
        STATE["prompted_boxes"]  = []
        STATE["mask_registry"]   = []
        STATE["edited_masks"]    = {}
        STATE["deleted_indices"] = set()
        return jsonify({"image_b64": _array_to_b64(STATE["image_array"]),
                        "width": STATE["image_w"], "height": STATE["image_h"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _clear_nonfrozen_edited_masks(frozen_ids):
    """
    Before a re-segmentation, remove edited_masks entries for masks that are
    NOT frozen.  Frozen masks' edited binaries must survive (they protect user
    brush-edits); non-frozen ones must be cleared so SAM's fresh output is used
    instead of the stale edited binary from the previous prompt.
    """
    frozen_set = set(int(i) for i in (frozen_ids or []))
    stale = [k for k in STATE["edited_masks"] if k not in frozen_set]
    for k in stale:
        del STATE["edited_masks"][k]



@app.route("/api/text_prompt", methods=["POST"])
def text_prompt():
    try:
        prompt = request.json.get("prompt", "").strip()
        if not prompt: return jsonify({"error": "Empty prompt"}), 400
        if STATE["processor_state"] is None: return jsonify({"error": "No image loaded"}), 400
        # Do NOT clear edited_masks here — brush edits must survive re-prompting.
        # Frozen masks on the client protect their pixels; we must keep server-side
        # edited binaries so export always uses the brushed version.
        STATE["processor_state"] = processor.set_text_prompt(prompt, STATE["processor_state"])
        return jsonify(_render_overlay())
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/add_box", methods=["POST"])
def add_box():
    try:
        data = request.json
        if STATE["processor_state"] is None: return jsonify({"error": "No image loaded"}), 400
        STATE["prompted_boxes"].append({"box": data["px_box"], "label": data["label"]})
        # Do NOT clear edited_masks — brush edits must survive adding new boxes.
        # Non-frozen edited_masks ARE cleared so SAM's fresh output can take effect.
        _clear_nonfrozen_edited_masks(data.get("frozen_ids"))
        STATE["processor_state"] = processor.add_geometric_prompt(
            data["box"], data["label"], STATE["processor_state"])
        result = _render_overlay()
        result["prompted_boxes"] = STATE["prompted_boxes"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/clear_prompts", methods=["POST"])
def clear_prompts():
    try:
        if STATE["processor_state"] is not None:
            STATE["processor_state"] = processor.reset_all_prompts(STATE["processor_state"])
            if "masks" in STATE["processor_state"]:
                del STATE["processor_state"]["masks"]
        STATE["prompted_boxes"]  = []
        STATE["mask_registry"]   = []
        STATE["edited_masks"]    = {}
        STATE["deleted_indices"] = set()
        return jsonify({"status": "cleared", "overlay_b64": None, "mask_list": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/set_confidence", methods=["POST"])
def set_confidence():
    try:
        val = float(request.json.get("value", 0.5))
        STATE["confidence"] = val
        if STATE["processor_state"] is None: return jsonify({"status": "ok"})
        # Do NOT clear edited_masks — brush edits must survive confidence changes.
        STATE["processor_state"] = processor.set_confidence_threshold(val, STATE["processor_state"])
        return jsonify(_render_overlay())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_mask", methods=["POST"])
def update_mask():
    """
    Accept a brush/eraser-edited mask from the client.
    Body: { mask_id: int, mask_b64: str, meta: {species_id, score, color} (optional) }
      mask_b64 = grayscale PNG, 255 = mask foreground, 0 = background
    Stores the edited binary and returns a fresh overlay.
    Works even when no SAM processor state exists (e.g. during exportAll).
    """
    try:
        data    = request.json
        mask_id = int(data["mask_id"])
        binary  = np.array(
            PIL.Image.open(io.BytesIO(base64.b64decode(data["mask_b64"]))).convert("L")
        ) > 127

        STATE["edited_masks"][mask_id] = binary

        # Build bounding box and polygon from the binary
        ys, xs = np.where(binary)
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else [0,0,1,1]

        poly = _binary_to_geom(binary) if len(xs) >= 4 else None
        # Drop genuinely degenerate polygons (area < 4 px²); keep old polygon if new one is None
        if poly is not None and poly.area < 4:
            poly = None

        # Try to update an existing registry entry
        found = False
        for m in STATE["mask_registry"]:
            if m["mask_id"] == mask_id:
                m["box"]    = box
                m["binary"] = binary
                # Only replace polygon when we have a valid new one; keep old shape otherwise
                if poly is not None:
                    m["polygon"] = poly
                found = True
                break

        # If no entry exists (e.g. exportAll without SAM state), create one from client meta
        if not found:
            meta = data.get("meta", {})
            color_int = [int(v) for v in meta.get("color", [128, 128, 128])]
            STATE["mask_registry"].append({
                "mask_id":    mask_id,
                "polygon":    poly,
                "species_id": meta.get("species_id"),
                "box":        box,
                "score":      float(meta.get("score", 1.0)),
                "color":      color_int,
                "binary":     binary,
            })

        return jsonify(_render_overlay())
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/delete_mask", methods=["POST"])
def delete_mask():
    """Remove one mask from the processor state by index."""
    try:
        mask_id = int(request.json.get("mask_id"))
        ps = STATE["processor_state"]
        if ps is None or "masks" not in ps:
            return jsonify({"error": "No masks"}), 400
        for key in ("masks", "boxes", "scores"):
            if key in ps and len(ps[key]) > mask_id:
                ps[key] = [v for i, v in enumerate(ps[key]) if i != mask_id]
        STATE["mask_registry"] = [m for m in STATE["mask_registry"] if m["mask_id"] != mask_id]
        if mask_id in STATE["edited_masks"]:
            del STATE["edited_masks"][mask_id]
        # Remember this original SAM index so _render_overlay never re-inserts it
        STATE["deleted_indices"].add(mask_id)
        return jsonify({"status": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/delete_species", methods=["POST"])
def delete_species():
    try:
        sid = request.json.get("id")
        STATE["species_list"] = [s for s in STATE["species_list"] if s["id"] != sid]
        return jsonify({"species_list": STATE["species_list"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/add_species", methods=["POST"])
def add_species():
    try:
        name = request.json.get("name", "").strip()
        if not name: return jsonify({"error": "Empty name"}), 400
        sid = len(STATE["species_list"]) + 1
        STATE["species_list"].append({"id": sid, "name": name})
        return jsonify({"species_list": STATE["species_list"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assign_species", methods=["POST"])
def assign_species():
    try:
        mask_id    = request.json.get("mask_id")
        species_id = request.json.get("species_id")
        for m in STATE["mask_registry"]:
            if m["mask_id"] == mask_id:
                m["species_id"] = species_id; break
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export", methods=["POST"])
def export_masks():
    try:
        folder = request.json.get("folder", "").strip() or None
        if not folder:
            ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder = os.path.join(os.getcwd(), f"sam3_exports_{ts}")
        os.makedirs(folder, exist_ok=True)

        H, W  = STATE["image_h"], STATE["image_w"]
        base  = STATE["image_name"]
        registry = STATE["mask_registry"]
        n_masks  = len(registry)

        if n_masks == 0:
            return jsonify({"error": "No masks to export"}), 400
        if H == 0 or W == 0:
            return jsonify({"error": "No image loaded"}), 400

        # ── Build grayscale value table keyed by mask_id ─────────────────────
        # Grayscale must be consistent across ALL images in a session, so it is
        # computed from the FULL global species list (sent by the client), not
        # just the species that happen to appear in this particular image.
        # This guarantees species A always gets the same gray value in every
        # exported PNG/GPKG regardless of which other species are present.
        sp_by_id  = {s["id"]: s["name"] for s in STATE["species_list"]}
        # Use every known species ID, sorted, so the mapping is deterministic
        all_sids = sorted(s["id"] for s in STATE["species_list"])
        n_species = len(all_sids)
        sid_to_gray = {}
        for rank, sid in enumerate(all_sids):
            gray = int(round(255 * (rank + 1) / max(n_species, 1)))
            sid_to_gray[sid] = gray

        # mask_id → grayscale value (0 = unlabeled)
        mask_gray = {}
        for m in registry:
            mask_gray[m["mask_id"]] = sid_to_gray.get(m["species_id"], 0)

        # ── PNG export: grayscale image, each mask painted with its gray value ─
        gray_arr = np.zeros((H, W), dtype=np.uint8)
        sorted_reg = sorted(registry, key=lambda m: m["mask_id"])
        for m in sorted_reg:
            binary = m.get("binary")
            if binary is not None and binary.shape == (H, W):
                gray_arr[binary] = mask_gray[m["mask_id"]]

        png_path = os.path.join(folder, f"{base}_sam3_only_labeled_masks.png")
        PIL.Image.fromarray(gray_arr, "L").save(png_path)

        # ── Legend: CSV + PNG swatch table ───────────────────────────────────
        legend_rows = []
        csv_path = leg_path = None

        if n_masks:
            for m in sorted_reg:
                legend_rows.append({
                    "mask_id":   m["mask_id"],
                    "grayscale": mask_gray[m["mask_id"]],
                    "species":   sp_by_id.get(m["species_id"], "Unlabeled"),
                    "class_id":  m["species_id"] if m["species_id"] is not None else "",
                    "r":         "",
                    "g":         "",
                    "b":         "",
                })

            csv_path = os.path.join(folder, f"{base}_legend.csv")
            with open(csv_path, "w", newline="") as fp:
                writer = csv.DictWriter(
                    fp, fieldnames=["mask_id","grayscale","species","class_id","r","g","b"])
                writer.writeheader()
                writer.writerows(legend_rows)

            # try:
            #     row_h, font_size = 28, 13
            #     legend_h = row_h * n_masks + 24
            #     leg_img  = PIL.Image.new("RGB", (380, legend_h), (20, 20, 28))
            #     draw     = PIL.ImageDraw.Draw(leg_img)
            #     try:
            #         font = PIL.ImageFont.truetype(
            #             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            #     except Exception:
            #         font = PIL.ImageFont.load_default()
            #     draw.text((8, 5), "colour swatch | gray value | mask id | species",
            #               fill=(120, 125, 154), font=font)
            #     for i, row in enumerate(legend_rows):
            #         y    = 22 + i * row_h
            #         gray = row["grayscale"]
            #         r, g, b = row["r"], row["g"], row["b"]
            #         draw.rectangle([8,  y+3, 48,  y+21], fill=(r, g, b))         # colour
            #         draw.rectangle([54, y+3, 74,  y+21], fill=(gray, gray, gray)) # gray
            #         draw.rectangle([54, y+3, 74,  y+21], outline=(80, 80, 80))
            #         label = f"{gray:>3}   Mask {row['mask_id']:>3}   {row['species']}"
            #         draw.text((82, y+6), label, fill=(232, 234, 240), font=font)
            #     leg_path = os.path.join(folder, f"{base}_legend.png")
            #     leg_img.save(leg_path)
            # except Exception:
            #     leg_path = None

        # ── GPKG export ───────────────────────────────────────────────────────
        geo = STATE.get("geo")

        if geo:
            _a, _b, _c, _d, _e, _f, raw_crs, _conv = _parse_affine(geo)
        else:
            _a, _b, _c, _d, _e, _f = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
            raw_crs = "EPSG:4326"

        def px_to_geo(col, row):
            """Pixel (col=x, row=y) → CRS (X, Y) via affine transform."""
            return (_c + col * _a + row * _b, _f + col * _d + row * _e)

        records = []
        for m in registry:
            species_name = sp_by_id.get(m["species_id"], "Unlabeled")
            poly = m["polygon"]

            # Skip masks whose polygon is None (degenerate/empty after brush edits)
            if poly is None:
                continue

            if geo:
                poly = _geom_with_transform(poly, px_to_geo)

            records.append({
                "mask_id":        m["mask_id"],
                "class_id":       m["species_id"],
                "species":        species_name,
                "grayscale_value": mask_gray[m["mask_id"]],
                "geometry":       poly,
            })

        gpkg_path = None
        if records:
            gpkg_path = os.path.join(folder, f"{base}_sam3_masks.gpkg")
            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=raw_crs)
            gdf.to_file(gpkg_path, driver="GPKG")

        return jsonify({
            "status":  "exported",
            "png":     png_path,
            "gpkg":    gpkg_path,
            "legend":  leg_path,
            "csv":     csv_path,
            "folder":  folder,
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/upload_geo", methods=["POST"])
def upload_geo():
    """Load a geo-metadata JSON independently of any image upload."""
    try:
        import json as _json
        json_file = request.files.get("json")
        if not json_file:
            return jsonify({"error": "No JSON file"}), 400
        STATE["geo"] = _json.load(json_file)
        return jsonify({"status": "loaded", "crs": STATE["geo"].get("crs")})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/upload_geo_clear", methods=["POST"])
def upload_geo_clear():
    STATE["geo"] = None
    return jsonify({"status": "cleared"})


@app.route("/api/geo_info", methods=["GET"])
def geo_info():
    """Return the currently loaded geo metadata + computed corner coordinates."""
    geo = STATE.get("geo")
    if not geo:
        return jsonify({"loaded": False})
    W = STATE["image_w"]
    H = STATE["image_h"]
    a, b, c, d, e, f, crs, convention = _parse_affine(geo)
    return jsonify({
        "loaded":     True,
        "crs":        crs,
        "convention": convention,
        "transform":  [a, b, c, d, e, f],
        "corners": {
            "top_left":     [c,             f            ],
            "top_right":    [c + W*a,       f + W*d      ],
            "bottom_left":  [c + H*b,       f + H*e      ],
            "bottom_right": [c + W*a + H*b, f + W*d + H*e],
        }
    })


@app.route("/api/geo_debug", methods=["GET"])
def geo_debug():
    """
    Return the raw geo JSON, the parsed affine coefficients, and the four
    corner coordinates computed from the current image size.
    Use this to verify the transform is being decoded correctly before exporting.
    """
    geo = STATE.get("geo")
    if not geo:
        return jsonify({"loaded": False})

    W, H = STATE["image_w"], STATE["image_h"]
    _a, _b, _c, _d, _e, _f, crs, convention = _parse_affine(geo)

    def px(col, row):
        return (_c + col*_a + row*_b, _f + col*_d + row*_e)

    return jsonify({
        "loaded":        True,
        "convention":    convention,
        "raw_transform": geo.get("transform"),
        "parsed":        {"a": _a, "b": _b, "c": _c, "d": _d, "e": _e, "f": _f},
        "crs":           crs,
        "image_size":    [W, H],
        "corners": {
            "top_left":     px(0, 0),
            "top_right":    px(W, 0),
            "bottom_left":  px(0, H),
            "bottom_right": px(W, H),
        }
    })



def scan_folder():
    """
    Scan a folder path for PNG/JPG images.
    Returns a list of {name, path, thumb_b64, geo_path, tile_row, tile_col,
    tiles_per_row, tiles_per_col} sorted by filename.
    Grid fields are read from companion geo JSONs when present.
    """
    try:
        import json as _json
        folder = request.json.get("folder", "").strip()
        if not folder or not os.path.isdir(folder):
            return jsonify({"error": f"Not a directory: {folder}"}), 400

        EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in EXTS
        )
        if not files:
            return jsonify({"error": "No images found in folder"}), 400

        items = []
        for fname in files:
            fpath = os.path.join(folder, fname)
            base_no_ext = os.path.splitext(fname)[0]
            geo_path = None
            for suffix in ("_geo.json", ".json"):
                candidate = os.path.join(folder, base_no_ext + suffix)
                if os.path.isfile(candidate):
                    geo_path = candidate
                    break

            # Read grid position fields from the geo JSON if present
            grid = {}
            if geo_path:
                try:
                    with open(geo_path) as fp:
                        gj = _json.load(fp)
                    for k in ("tile_row", "tile_col", "tiles_per_row", "tiles_per_col", "tile_id"):
                        if k in gj:
                            grid[k] = gj[k]
                except Exception:
                    pass

            try:
                img = PIL.Image.open(fpath).convert("RGB")
                img.thumbnail((120, 90))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                thumb = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                thumb = ""
            items.append({"name": fname, "path": fpath, "thumb_b64": thumb,
                          "geo_path": geo_path, **grid})

        return jsonify({"items": items, "folder": folder})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/neighbor_thumb", methods=["POST"])
def neighbor_thumb():
    """
    Return a cropped edge strip of a neighbour tile as a base64 PNG.
    Body: { path, pad_px, position, export_folder }
    position: 'top'|'bottom'|'left'|'right'|'tl'|'tr'|'bl'|'br'
    The crop taken is the edge of the neighbour that faces the current tile.
    """
    try:
        data          = request.json
        img_path      = data.get("path", "")
        pad_px        = int(data.get("pad_px", 80))
        position      = data.get("position", "")
        export_folder = data.get("export_folder", "") or ""

        if not img_path or not os.path.isfile(img_path):
            return jsonify({"error": f"Not found: {img_path}"}), 400

        pil_img = PIL.Image.open(img_path).convert("RGB")
        W, H    = pil_img.size

        # Crop the edge of the neighbour that faces the current tile.
        # pos describes the neighbour's position relative to the current tile,
        # so we take the OPPOSITE edge of the neighbour.
        p = pad_px
        crop_box = {
            "left":   (W-p, 0,   W,   H),
            "right":  (0,   0,   p,   H),
            "top":    (0,   H-p, W,   H),
            "bottom": (0,   0,   W,   p),
            "tl":     (W-p, H-p, W,   H),
            "tr":     (0,   H-p, p,   H),
            "bl":     (W-p, 0,   W,   p),
            "br":     (0,   0,   p,   p),
        }.get(position)

        if not crop_box:
            return jsonify({"error": f"Unknown position: {position}"}), 400

        strip = pil_img.crop(crop_box)
        buf   = io.BytesIO()
        strip.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        # Optionally return exported mask strip too
        mask_b64 = None
        if export_folder and os.path.isdir(export_folder):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(export_folder, f"{base_name}_sam3_only_labeled_masks.png")
            if os.path.isfile(mask_path):
                mask_img   = PIL.Image.open(mask_path).convert("L")
                mask_strip = mask_img.crop(crop_box)
                buf2 = io.BytesIO()
                mask_strip.save(buf2, format="PNG")
                mask_b64 = base64.b64encode(buf2.getvalue()).decode()

        sw, sh = strip.size
        return jsonify({"image_b64": image_b64, "mask_b64": mask_b64,
                        "width": sw, "height": sh})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500



@app.route("/api/restore_state", methods=["POST"])
def restore_state():
    """
    Restore mask registry + species assignments from a client snapshot.
    Used by export-all so we never need to re-segment on the server.
    
    """
    try:
        import json as _json
        data = request.json
        geo  = data.get("geo")
        STATE["geo"] = geo  # may be None

        masks_data = data.get("masks", [])
        STATE["mask_registry"] = []
        H, W = STATE["image_h"], STATE["image_w"]

        for m in masks_data:
            coords = m["polygon_coords"]
            poly   = Polygon(coords) if len(coords) > 3 else None
            if poly is not None and (not poly.is_valid or poly.area < 4):
                poly = poly.buffer(0)  # attempt fix
                if poly.area < 4:
                    poly = None
            # Do NOT fall back to a bounding-box rectangle — that produces fake
            # rectangular masks in the GPKG.  If the polygon is bad, skip this mask.
            if poly is None:
                continue

            # Reconstruct binary from polygon (rasterize)
            binary = np.zeros((H, W), dtype=bool)
            try:
                from skimage.draw import polygon as sk_polygon
                px = [c[1] for c in coords]   # rows
                py = [c[0] for c in coords]   # cols
                rr, cc = sk_polygon(px, py, (H, W))
                binary[rr, cc] = True
            except Exception:
                pass

            STATE["mask_registry"].append({
                "mask_id":    m["mask_id"],
                "polygon":    poly,
                "species_id": m.get("species_id"),
                "box":        m.get("box", [0,0,1,1]),
                "score":      m.get("score", 1.0),
                "color":      m.get("color", [128,128,128]),
                "binary":     binary,
            })

        STATE["species_list"] = data.get("species_list", STATE["species_list"])
        return jsonify({"status": "ok", "n_masks": len(STATE["mask_registry"])})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500



@app.route("/api/load_path", methods=["POST"])
def load_path():
    """Load an image from an absolute filesystem path (used in folder sessions)."""
    try:
        import json as _json
        path     = request.json.get("path", "").strip()
        geo_path = request.json.get("geo_path")   # optional companion JSON
        if not path or not os.path.isfile(path):
            return jsonify({"error": f"File not found: {path}"}), 400

        # Load geo JSON if provided
        if geo_path and os.path.isfile(geo_path):
            with open(geo_path) as fp:
                STATE["geo"] = _json.load(fp)
        else:
            STATE["geo"] = None

        STATE["image_name"]      = os.path.splitext(os.path.basename(path))[0]
        pil_img                  = PIL.Image.open(path).convert("RGB")
        STATE["image_array"]     = np.array(pil_img)
        STATE["image_w"], STATE["image_h"] = pil_img.size
        STATE["processor_state"] = processor.set_image(pil_img)
        STATE["prompted_boxes"]  = []
        STATE["mask_registry"]   = []
        STATE["edited_masks"]    = {}
        STATE["deleted_indices"] = set()
        return jsonify({"image_b64": _array_to_b64(STATE["image_array"]),
                        "width": STATE["image_w"], "height": STATE["image_h"],
                        "name": STATE["image_name"],
                        "geo_loaded": STATE["geo"] is not None,
                        "geo_crs": STATE["geo"].get("crs") if STATE["geo"] else None})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
