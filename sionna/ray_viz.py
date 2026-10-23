"""Draw the propagation paths Sionna RT actually traced (not straight UAV-user lines).

Every path is rebuilt from the solver's own output:
  paths.vertices     [max_depth, num_rx, num_tx, num_paths, 3]  interaction points
  paths.interactions [max_depth, num_rx, num_tx, num_paths]     InteractionType per bounce
  paths.valid        [num_rx, num_tx, num_paths]
and drawn as transmitter -> interaction points -> receiver, coloured by kind:

  los           direct line of sight (no interaction)
  reflection_1  one specular reflection
  reflection_2  two or more specular reflections
  through_wall  at least one refraction (transmission through a wall)

Diffraction and diffuse scattering are deliberately not traced (see
TRACE_SETTINGS): the scene has a single radio material (itu_brick) and a zero
scattering coefficient, so those effects would rest on invented values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Same tracer settings as the project's 3D pipeline (radio maps / PathSolverRF),
# with the effects we cannot support with material data switched off explicitly.
TRACE_SETTINGS = dict(max_depth=3, los=True, specular_reflection=True, diffuse_reflection=False,
                      refraction=True, diffraction=False, seed=42)

KINDS = ["los", "reflection_1", "reflection_2", "through_wall"]
KIND_LABEL = {"los": "Direct line of sight", "reflection_1": "1 reflection",
              "reflection_2": "2-3 reflections", "through_wall": "Through a wall"}
# Okabe-Ito colours (colour-blind safe); through-wall drawn dashed as well
KIND_COLOUR = {"los": "#009E73", "reflection_1": "#0072B2", "reflection_2": "#E69F00",
               "through_wall": "#7f7f7f"}
BOUNCE_COLOUR = "#D55E00"
NONE, SPECULAR, REFRACTION = 0, 1, 4    # sionna.rt.constants.InteractionType


def configure_scene(scene, freq_hz: float):
    """Isotropic, vertically polarised single antennas (as everywhere in the project)."""
    from sionna.rt import PlanarArray
    scene.frequency = freq_hz
    for attr in ("tx_array", "rx_array"):
        setattr(scene, attr, PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                         horizontal_spacing=0.5, pattern="iso", polarization="V"))
    return scene


def place_devices(scene, uav_positions, user_positions):
    from sionna.rt import Transmitter, Receiver
    for name in list(scene.transmitters) + list(scene.receivers):
        scene.remove(name)
    # explicit marker sizes: by default Sionna scales them to the whole 5 km scene
    for k, p in enumerate(uav_positions):
        scene.add(Transmitter(name=f"uav_{k+1}", position=[float(v) for v in p], display_radius=3.0,
                              color=(0.8, 0.0, 0.0)))
    for i, u in enumerate(user_positions):
        scene.add(Receiver(name=f"user_{i+1}", position=[float(v) for v in u], display_radius=1.5,
                           color=(0.1, 0.1, 0.1)))


def trace(scene):
    from sionna.rt import PathSolver
    return PathSolver()(scene, **TRACE_SETTINGS)


def extract_rays(paths, uav_positions, user_positions) -> list[dict]:
    """One dict per valid path: uav, user, kind, polyline [M, 3], bounce points, gain_db, delay_ns."""
    V = np.asarray(paths.vertices.numpy(), dtype=float)
    I = np.asarray(paths.interactions.numpy())
    ok = np.asarray(paths.valid.numpy(), dtype=bool)
    ar, ai = paths.a
    gain = np.asarray(ar.numpy()) ** 2 + np.asarray(ai.numpy()) ** 2      # [rx, rx_ant, tx, tx_ant, path]
    gain = gain[:, 0, :, 0, :]
    tau = np.asarray(paths.tau.numpy(), dtype=float)                       # [rx, tx, path]
    if V.ndim != 5 or ok.ndim != 3:
        raise ValueError(f"expected synthetic-array path tensors, got vertices {V.shape}, valid {ok.shape}")

    rays = []
    for i, u in enumerate(user_positions):
        for k, p in enumerate(uav_positions):
            for j in np.flatnonzero(ok[i, k]):
                inter = I[:, i, k, j]
                hit = inter != NONE
                pts = V[hit, i, k, j]
                if (inter & REFRACTION).any():
                    kind = "through_wall"
                elif hit.sum() == 0:
                    kind = "los"
                elif hit.sum() == 1:
                    kind = "reflection_1"
                else:
                    kind = "reflection_2"
                g = float(gain[i, k, j])
                rays.append({
                    "uav": k, "user": i, "kind": kind,
                    "polyline": np.vstack([np.asarray(p, float), pts, np.asarray(u, float)]),
                    "bounces": pts[(inter[hit] & SPECULAR) != 0],
                    "gain_db": 10 * np.log10(g) if g > 0 else -np.inf,
                    "delay_ns": float(tau[i, k, j]) * 1e9,
                })
    return rays


def link_table(rays, n_uavs: int, n_users: int, tx_power_dbm: float = 30.0) -> list[dict]:
    """Per (UAV, user): path counts by kind, total received power, strongest path kind."""
    rows = []
    for k in range(n_uavs):
        for i in range(n_users):
            rs = [r for r in rays if r["uav"] == k and r["user"] == i]
            lin = sum(10 ** (r["gain_db"] / 10) for r in rs if np.isfinite(r["gain_db"]))
            best = max(rs, key=lambda r: r["gain_db"])["kind"] if rs else None
            rows.append({"uav": k + 1, "user": i + 1,
                         **{kind: sum(r["kind"] == kind for r in rs) for kind in KINDS},
                         "rss_dbm": round(tx_power_dbm + 10 * np.log10(lin), 1) if lin > 0 else None,
                         "strongest": KIND_LABEL.get(best, "no path")})
    return rows


def crop_mesh(obj_path: str | Path, x_range, y_range, margin: float = 30.0):
    """Faces of the scene OBJ whose centroid lies in the window (+ margin): (vertices, faces)."""
    import trimesh
    mesh = trimesh.load(str(obj_path), force="mesh")
    c = mesh.triangles_center
    keep = ((c[:, 0] >= x_range[0] - margin) & (c[:, 0] <= x_range[1] + margin)
            & (c[:, 1] >= y_range[0] - margin) & (c[:, 1] <= y_range[1] + margin))
    sub = mesh.submesh([np.flatnonzero(keep)], append=True)
    return np.asarray(sub.vertices), np.asarray(sub.faces)


def plotly_figure(rays, uav_positions, user_positions, mesh_vf, title: str):
    """Interactive 3D view: city block, UAVs, users, and every traced path by kind.
    Legend entries toggle a path kind; the buttons show all UAVs or one UAV's paths."""
    import plotly.graph_objects as go
    verts, faces = mesh_vf
    P, U = np.asarray(uav_positions, float), np.asarray(user_positions, float)
    K = len(P)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color="#c8c8c8", opacity=0.55, flatshading=True, hoverinfo="skip",
                            name="Buildings (city_damaged)", showlegend=True))
    owner = [None]                                         # which UAV each trace belongs to (None = always shown)
    in_legend = set()
    for k in range(K):
        for kind in KINDS:
            xs, ys, zs, txt = [], [], [], []
            for r in (r for r in rays if r["uav"] == k and r["kind"] == kind):
                pl = r["polyline"]
                xs += list(pl[:, 0]) + [None]
                ys += list(pl[:, 1]) + [None]
                zs += list(pl[:, 2]) + [None]
                t = (f"UAV {k+1} → user {r['user']+1}<br>{KIND_LABEL[kind]}<br>"
                     f"path gain {r['gain_db']:.1f} dB, delay {r['delay_ns']:.0f} ns")
                txt += [t] * len(pl) + [None]
            if not xs:
                continue
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", hovertext=txt, hoverinfo="text",
                line=dict(color=KIND_COLOUR[kind], width=4 if kind != "through_wall" else 2,
                          dash="dash" if kind == "through_wall" else "solid"),
                name=KIND_LABEL[kind], legendgroup=kind, showlegend=kind not in in_legend))
            in_legend.add(kind)
            owner.append(k)
    B = np.vstack([r["bounces"] for r in rays if len(r["bounces"])]) if any(len(r["bounces"]) for r in rays) \
        else np.zeros((0, 3))
    fig.add_trace(go.Scatter3d(x=B[:, 0], y=B[:, 1], z=B[:, 2], mode="markers",
                               marker=dict(size=2.5, color=BOUNCE_COLOUR), hoverinfo="skip",
                               name="Reflection points"))
    owner.append(None)
    fig.add_trace(go.Scatter3d(x=U[:, 0], y=U[:, 1], z=U[:, 2], mode="markers+text",
                               marker=dict(size=5, color="#222222", symbol="circle"),
                               text=[f"U{i+1}" for i in range(len(U))], textposition="top center",
                               textfont=dict(size=10), name="Ground users"))
    owner.append(None)
    fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text",
                               marker=dict(size=8, color="#CC0000", symbol="diamond"),
                               text=[f"UAV {k+1}" for k in range(K)], textposition="top center",
                               name="UAVs"))
    owner.append(None)

    def vis(sel):
        return [o is None or sel is None or o == sel for o in owner]
    buttons = [dict(label="All UAVs", method="restyle", args=[{"visible": vis(None)}])]
    buttons += [dict(label=f"UAV {k+1}", method="restyle", args=[{"visible": vis(k)}]) for k in range(K)]
    fig.update_layout(
        title=title, height=780, margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="height (m)"),
        legend=dict(x=0.0, y=0.92, bgcolor="rgba(255,255,255,0.8)"),
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.0, xanchor="left", buttons=buttons)])
    return fig


def sionna_render(scene, paths, centre, filename: str | Path, height: float = 260.0,
                  resolution=(1280, 900), num_samples: int = 256, points=None):
    """Sionna's own rendering of the scene with the traced paths, from an oblique
    camera above the window. With `points` (UAV + user xyz) the camera is centred on
    them and pulled back until their spread fits. Saved to `filename` and returned as
    a matplotlib figure."""
    from sionna.rt import Camera
    cx, cy, k = float(centre[0]), float(centre[1]), 1.0
    if points is not None:
        xy = np.asarray(points, dtype=float)[:, :2]
        cx, cy = (float(v) for v in (xy.min(axis=0) + xy.max(axis=0)) / 2.0)  # mi.Point3f rejects numpy scalars
        k = max(1.0, float(np.ptp(xy, axis=0).max()) / 150.0)
    cam = Camera(position=[cx - 120.0 * k, cy - 230.0 * k, height * k], look_at=[cx, cy, 0.0])
    fig = scene.render(camera=cam, paths=paths, resolution=resolution, num_samples=num_samples,
                       show_devices=True)
    fig.savefig(str(filename), dpi=120, bbox_inches="tight")
    return fig
