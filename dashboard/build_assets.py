"""Generate web-ready 3D scene assets for the ResQNet dashboard.

The original meshes (``sionna/scenes/city_damaged.obj`` /
``city_baseline.obj``) are ~13 MB / ~318k triangles and span roughly
5.3 km x 5.4 km. This script produces cropped ``.glb`` files plus a
``scene_meta.json`` describing exactly what was done.

The originals are never modified. Coordinates are preserved: the cropped
meshes stay in the original Sionna scene frame (metres), so the real UAV
and ground-user coordinates overlay directly.

Honesty note baked into scene_meta.json: the dense modelled urban area
sits ~400-800 m from the origin. The UAV / ground-user operating region
around the origin is mostly open ground with only sparse low structures.
The default crop is wide (+/- 700 m) so the surrounding city is visible
and the viewer can see where the operating region actually sits.

Usage:
    python dashboard/build_assets.py                 # default +/- 700 m crop
    python dashboard/build_assets.py --half-extent 500
    python dashboard/build_assets.py --decimate 60000 # target max triangles
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DAMAGED = PROJECT_ROOT / "sionna" / "scenes" / "city_damaged.obj"
SRC_BASELINE = PROJECT_ROOT / "sionna" / "scenes" / "city_baseline.obj"
ASSETS_DIR = PROJECT_ROOT / "dashboard" / "assets"

# UAVDynamicEnv position clip bounds (ml/dynamic_model.py) - the region the
# PPO agent can actually operate in.
OPERATIONAL_REGION = {"x": [-50.0, 70.0], "y": [-50.0, 70.0], "z": [10.0, 50.0]}


def crop_mesh(mesh: trimesh.Trimesh, half_extent: float) -> trimesh.Trimesh:
    """Keep only faces whose three vertices are within +/- half_extent in X and Y."""
    v = mesh.vertices
    inside = (np.abs(v[:, 0]) <= half_extent) & (np.abs(v[:, 1]) <= half_extent)
    face_mask = inside[mesh.faces].all(axis=1)
    cropped = mesh.submesh([face_mask], append=True)
    if isinstance(cropped, list):  # submesh may return a list
        cropped = trimesh.util.concatenate(cropped)
    return cropped


def maybe_decimate(mesh: trimesh.Trimesh, target_faces: int) -> tuple[trimesh.Trimesh, bool]:
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh, False
    try:
        reduced = mesh.simplify_quadric_decimation(face_count=target_faces)
        return reduced, True
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  [warn] decimation unavailable ({exc}); keeping full crop")
        return mesh, False


def process(src: Path, dst: Path, half_extent: float, target_faces: int) -> dict:
    print(f"- {src.name}")
    mesh = trimesh.load(src, force="mesh")
    full_faces = int(len(mesh.faces))
    cropped = crop_mesh(mesh, half_extent)
    crop_faces = int(len(cropped.faces))
    cropped, decimated = maybe_decimate(cropped, target_faces)
    final_faces = int(len(cropped.faces))

    dst.parent.mkdir(parents=True, exist_ok=True)
    cropped.export(dst)  # trimesh infers GLB from the .glb suffix
    size_kb = dst.stat().st_size / 1024
    print(
        f"    full={full_faces} faces  crop={crop_faces}  "
        f"final={final_faces}  ->  {dst.name} ({size_kb:.0f} KB)"
    )
    return {
        "source": str(src.relative_to(PROJECT_ROOT)),
        "output": dst.name,
        "full_faces": full_faces,
        "cropped_faces": crop_faces,
        "final_faces": final_faces,
        "decimated": decimated,
        "size_kb": round(size_kb, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half-extent", type=float, default=700.0, help="crop half-size in metres (default 700)")
    ap.add_argument("--decimate", type=int, default=0, help="target max triangles (0 = no decimation)")
    args = ap.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ResQNet web asset build  (crop +/- {args.half_extent:.0f} m)")

    full = trimesh.load(SRC_DAMAGED, force="mesh")
    full_bounds = full.bounds.tolist()

    d = process(SRC_DAMAGED, ASSETS_DIR / "city_damaged.glb", args.half_extent, args.decimate)
    b = process(SRC_BASELINE, ASSETS_DIR / "city_baseline.glb", args.half_extent, args.decimate)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coordinate_frame": (
            "Original Sionna scene coordinates, metres. Axes as in the OBJ: "
            "X and Y are ground plane, Z is height/altitude. Meshes are NOT "
            "recentred - real UAV and user coordinates overlay directly."
        ),
        "full_scene_bounds": full_bounds,
        "web_asset_crop": {
            "half_extent_m": args.half_extent,
            "center_xy": [0.0, 0.0],
            "bounds_xy": [
                [-args.half_extent, -args.half_extent],
                [args.half_extent, args.half_extent],
            ],
            "damaged": d,
            "baseline": b,
        },
        "operational_region": {
            **OPERATIONAL_REGION,
            "note": "UAVDynamicEnv position clip bounds (ml/dynamic_model.py).",
        },
        "geometry_honesty_note": (
            "The dense modelled urban area lies roughly 400-800 m from the "
            "origin. Within +/- 200 m of the origin the mesh has only a few "
            "hundred triangles - mostly open ground with sparse structures "
            "under ~9 m. The UAV (~26 m altitude) therefore has near line-of-"
            "sight to the ground users, which is why Sionna reports high "
            "coverage. The wide crop shows where the operating region sits "
            "relative to the city."
        ),
        "damage_labels": {
            "available": False,
            "reason": (
                "sionna/scenes/damage_labels.geojson uses a different, "
                "unaligned coordinate frame (values around 1500, -475). It is "
                "not projected onto this scene - no faked overlay."
            ),
        },
        "channel_model_note": (
            "This geometry drives Sionna RT ray tracing (propagation / radio "
            "analysis). NS-3 uses its own Wi-Fi channel model for network / "
            "MAC / traffic validation and does not consume Sionna's ray-traced "
            "channel loss."
        ),
    }
    (ASSETS_DIR / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"- scene_meta.json -> {ASSETS_DIR / 'scene_meta.json'}")
    print("done.")


if __name__ == "__main__":
    main()
