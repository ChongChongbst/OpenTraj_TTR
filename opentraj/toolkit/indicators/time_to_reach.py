'''
time to reach pipeline - scene goal picker + TTR computation
======================================================================
Run:
    python time_to_reach.py                      # pick goal on ETH-Univ
    python time_to_reach.py --scene UCY-Zara1
    python time_to_reach.py --scene ETH-Hotel --backend odp  (needs optimized_dp: https://github.com/SFU-MARS/optimized_dp/tree/master)
    python time_to_reach.py --goal 5 5 7 6       # headless: goal rectangle XMIN YMIN XMAX YMAX in metres

Workflow:
    1. A window opens showing the scene photo, the pedestrian trajectories, and obstacle walls.
    2. Draw a rectangle goal region:
        First click -> place one corner of the goal
        Second click -> place the opposite corner of the goal; the goal region is now defined
        Right click -> clear and start over
      Two sliders below let you fine tune the half width and half height of the goal region (world meters).
    3. Press the "Compute TTR" button: TTR is computed and saved as
       output/<scene>-ttr-<n>.npz/.png. The picker stays open — pick another goal
       and press it again as often as you like. Press "Quit" when done.
    Without a display, the picker is served in the browser (WebAgg) at the URL
    printed in the terminal (VS Code Remote-SSH forwards the port automatically).
    .
    .
    .

'''
import argparse, xml.etree.ElementTree as ET
import numpy as np
import os
import matplotlib
# Pick how the interactive goal picker is shown:
#   "window"  — a desktop window (needs $DISPLAY, e.g. local session or ssh -X)
#   "browser" — matplotlib WebAgg: served at http://127.0.0.1:<port> and opened
#               in a web browser (works over VS Code Remote-SSH, which forwards
#               the port automatically)
#   None      — no interactive picker possible; --goal must be given
UI_MODE = None
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    for _backend in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(_backend, force=True)
            import matplotlib.pyplot as _test_plt
            _test_plt.figure(); _test_plt.close()
            UI_MODE = "window"
            break
        except Exception:
            continue
if UI_MODE is None:
    try:
        import tornado  # noqa: F401  (required by the WebAgg backend)
        matplotlib.use("WebAgg", force=True)
        matplotlib.rcParams["webagg.open_in_browser"] = False
        UI_MODE = "browser"
    except Exception:
        matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent.parent / "datasets"
#      tools/         toolkit/  opentraj/  OpenTraj/   datasets/
OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)

NS = "https://github.com/amiryanj/OpenTraj"

SCENE_CFG = {
    "ETH-Univ":  {"path": BASE/"ETH/seq_eth",
                  "tfile": "biwi_eth_10fps.txt", "ncols": 4,
                  "H": "H.txt",     "bg": "bg.png"},
    "ETH-Hotel": {"path": BASE/"ETH/seq_hotel",
                  "tfile": "obsmat.txt",          "ncols": 8,
                  "H": "H.txt",     "bg": "bg.png"},
    "UCY-Zara1": {"path": BASE/"UCY/zara01",
                  "tfile": "obsmat.txt",          "ncols": 8,
                  "H": "H.txt",     "bg": "bg.png"},
    "UCY-Zara2": {"path": BASE/"UCY/zara02",
                  "tfile": "obsmat.txt",          "ncols": 8,
                  "H": "H-old.txt", "bg": "reference.png"},
    "UCY-Univ3": {"path": BASE/"UCY/students03",
                  "tfile": "obsmat.txt",          "ncols": 8,
                  "H": "H-old.txt", "bg": "reference.png"},
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_trajectories(cfg: dict) -> dict:
    """Returns {ped_id: (T,2) array of world (x,y) in metres}."""
    d = np.loadtxt(cfg["path"] / cfg["tfile"])
    peds = {}
    for row in d:
        pid = int(row[1])
        xy  = (float(row[2]), float(row[3])) if cfg["ncols"] == 4 \
              else (float(row[2]), float(row[4]))
        peds.setdefault(pid, []).append(xy)
    return {k: np.array(v) for k, v in peds.items()}


def load_obstacles(cfg: dict):
    """Returns (lines, circles) — both in world metres."""
    mxml = cfg["path"] / "map.xml"
    lines, circles = [], []
    if not mxml.exists():
        return lines, circles
    root = ET.parse(mxml).getroot()
    for el in root.iter(f"{{{NS}}}Line"):
        lines.append(((float(el.get("x1")), float(el.get("y1"))),
                      (float(el.get("x2")), float(el.get("y2")))))
    for el in root.iter(f"{{{NS}}}Circle"):
        circles.append((float(el.get("x")), float(el.get("y")),
                        float(el.get("radius"))))
    return lines, circles


# ── coordinate helpers ────────────────────────────────────────────────────────
# H maps  [row, col, 1]  →  world (x, y)
# H_inv maps world (x, y)  →  [row, col]

def world_to_rowcol(wx, wy, H_inv):
    v = H_inv @ np.array([wx, wy, 1.0]); v /= v[2]
    return v[0], v[1]          # row, col

def rowcol_to_world(row, col, H):
    v = H @ np.array([row, col, 1.0]); v /= v[2]
    return v[0], v[1]          # wx, wy


# ── STEP 1 — interactive goal picker ─────────────────────────────────────────

def pick_goal_interactive(scene_name: str, cfg: dict, on_compute,
                          init_half_w: float = 1.0,
                          init_half_h: float = 0.5) -> list:
    """
    Open a matplotlib window (or browser page) showing the scene.

    Drawing interaction:
      First click   → anchor corner of the goal rectangle
      Second click  → opposite corner → rectangle is fixed
      Right-click   → clear and start over

    Width / height sliders let you fine-tune the rectangle after placing it.
    "Compute TTR" calls on_compute(bbox) -> (out_stem, max_ttr) and keeps the
    picker open for the next goal; "Quit" (or closing the window) ends it.

    Returns
    -------
    list of output stems that were saved (e.g. ["ETH-Univ-ttr-0", ...])
    """
    from matplotlib.widgets import Slider, Button

    H     = np.loadtxt(cfg["path"] / cfg["H"])
    H_inv = np.linalg.inv(H)
    peds  = load_trajectories(cfg)
    lines, circles = load_obstacles(cfg)
    bg    = np.array(Image.open(cfg["path"] / cfg["bg"]))
    img_h, img_w = bg.shape[:2]

    # ── helpers: world ↔ pixel ------------------------------------------------
    def w2px(wx, wy):
        """World → (col, row) in image pixels."""
        row, col = world_to_rowcol(wx, wy, H_inv)
        return col, row

    def px2w(col, row):
        """Image pixel → world (x, y)."""
        return rowcol_to_world(row, col, H)

    def world_len_to_px(wx, wy, dx_world):
        """Approximate: how many pixels does dx_world metres span at (wx,wy)?"""
        c0, _ = w2px(wx, wy)
        c1, _ = w2px(wx + dx_world, wy)
        return abs(c1 - c0)

    # ── figure ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.canvas.manager.set_window_title(f"Pick goal — {scene_name}")
    plt.subplots_adjust(bottom=0.18)

    ax.imshow(bg, origin="upper")
    ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)
    ax.set_title(
        f"{scene_name}   |   "
        f"click 1 = first corner   click 2 = opposite corner   "
        f"right-click = clear   [Compute TTR] = compute & save",
        fontsize=9)
    ax.set_xlabel("pixel col"); ax.set_ylabel("pixel row")

    # trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, traj in enumerate(list(peds.values())[:50]):
        rows, cols = zip(*[world_to_rowcol(wx, wy, H_inv) for wx, wy in traj])
        ax.plot(cols, rows, "-", color=colors[i % 10], lw=0.9, alpha=0.55)
        ax.plot(cols[0], rows[0], "o", color=colors[i % 10], ms=2.5)

    # obstacle walls
    for (x1, y1), (x2, y2) in lines:
        r1, c1 = world_to_rowcol(x1, y1, H_inv)
        r2, c2 = world_to_rowcol(x2, y2, H_inv)
        ax.plot([c1, c2], [r1, r2], "r-", lw=2.5, zorder=5)
    for cx, cy, r in circles:
        rc, cc = world_to_rowcol(cx, cy, H_inv)
        rc2, cc2 = world_to_rowcol(cx + r, cy, H_inv)
        ax.add_patch(mpatches.Circle(
            (cc, rc), abs(cc2 - cc), color="red", fill=False, lw=2.5, zorder=5))

    # ── sliders ---------------------------------------------------------------
    ax_sw = fig.add_axes([0.12, 0.09, 0.50, 0.025])
    ax_sh = fig.add_axes([0.12, 0.05, 0.50, 0.025])
    sl_w = Slider(ax_sw, "Half-width  (m)", 0.1, 5.0,
                  valinit=init_half_w, valstep=0.1, color="gold")
    sl_h = Slider(ax_sh, "Half-height (m)", 0.1, 5.0,
                  valinit=init_half_h, valstep=0.1, color="gold")

    # ── status text -----------------------------------------------------------
    status = ax.text(
        0.01, 0.015, "Click first corner of goal rectangle",
        transform=fig.transFigure, fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#222", ec="none", alpha=0.85))

    # ── goal state ------------------------------------------------------------
    # We store the goal as a centre + half-sizes in world coords.
    # Placing is two-click: first click sets anchor, second sets opposite corner.
    state = {
        "phase":    "idle",   # idle | anchor_set | done
        "anchor":   None,     # (wx, wy) of first click
        "cx":       None,     # world centre x
        "cy":       None,     # world centre y
        "rect_px":  None,     # matplotlib Rectangle patch (pixel space)
        "dot":      None,     # anchor dot artist
    }

    def _bbox_world():
        """Current goal as [xmin, ymin, xmax, ymax] in world metres."""
        if state["cx"] is None:
            return None
        hw, hh = sl_w.val, sl_h.val
        return np.array([state["cx"] - hw, state["cy"] - hh,
                         state["cx"] + hw, state["cy"] + hh])

    def _redraw_rect():
        """Remove and redraw the yellow rectangle using current centre + sliders."""
        if state["rect_px"]:
            state["rect_px"].remove(); state["rect_px"] = None
        if state["cx"] is None:
            return
        hw, hh = sl_w.val, sl_h.val
        # world corners → pixel corners
        c_tl, r_tl = w2px(state["cx"] - hw, state["cy"] + hh)
        c_br, r_br = w2px(state["cx"] + hw, state["cy"] - hh)
        # Rectangle: (left_col, top_row), width_px, height_px
        left   = min(c_tl, c_br)
        top    = min(r_tl, r_br)
        w_px   = abs(c_br - c_tl)
        h_px   = abs(r_br - r_tl)
        state["rect_px"] = mpatches.Rectangle(
            (left, top), w_px, h_px,
            linewidth=2, edgecolor="yellow", facecolor="yellow",
            alpha=0.25, linestyle="--", zorder=9)
        ax.add_patch(state["rect_px"])

    def _update_status():
        b = _bbox_world()
        if state["phase"] == "idle":
            status.set_text("Click first corner of goal rectangle")
        elif state["phase"] == "anchor_set":
            ax_, ay_ = state["anchor"]
            status.set_text(
                f"First corner ({ax_:.2f}, {ay_:.2f}) m — click opposite corner")
        else:
            if b is not None:
                status.set_text(
                    f"Goal  ({b[0]:.2f}, {b[1]:.2f}) → ({b[2]:.2f}, {b[3]:.2f}) m"
                    f"   size {b[2]-b[0]:.2f} × {b[3]-b[1]:.2f} m"
                    f"   —   adjust sliders, then press Compute TTR")

    def _clear():
        for key in ("rect_px", "dot"):
            if state[key]: state[key].remove(); state[key] = None
        state.update(phase="idle", anchor=None, cx=None, cy=None)
        _update_status()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax: return
        if event.button == 3:
            _clear(); return
        if event.button != 1: return

        wx, wy = px2w(event.xdata, event.ydata)

        if state["phase"] == "idle":
            # first click — place anchor dot
            if state["dot"]: state["dot"].remove()
            state["dot"] = ax.plot(event.xdata, event.ydata, "o",
                                   color="yellow", ms=6, zorder=10,
                                   mec="darkorange", mew=1)[0]
            state.update(phase="anchor_set", anchor=(wx, wy))
            _update_status()
            fig.canvas.draw_idle()

        elif state["phase"] in ("anchor_set", "done"):
            # second (or subsequent) click — set opposite corner
            ax_, ay_ = state["anchor"]
            state["cx"] = (wx + ax_) / 2
            state["cy"] = (wy + ay_) / 2
            # derive half-sizes from click span, overriding sliders
            hw = abs(wx - ax_) / 2
            hh = abs(wy - ay_) / 2
            if hw > 0.05: sl_w.set_val(round(hw, 1))
            if hh > 0.05: sl_h.set_val(round(hh, 1))
            if state["dot"]: state["dot"].remove(); state["dot"] = None
            state["phase"] = "done"
            _redraw_rect()
            _update_status()
            fig.canvas.draw_idle()

    def on_slider(_):
        _redraw_rect()
        _update_status()
        fig.canvas.draw_idle()

    # ── buttons: "Compute TTR" (compute + save, picker stays open) and "Quit" ---
    ax_done = fig.add_axes([0.70, 0.075, 0.16, 0.05])
    btn_done = Button(ax_done, "Compute TTR", color="gold", hovercolor="orange")
    ax_quit = fig.add_axes([0.70, 0.02, 0.16, 0.045])
    btn_quit = Button(ax_quit, "Quit", color="0.85", hovercolor="0.7")

    if UI_MODE == "browser":
        import tornado.ioloop
        from matplotlib.backends.backend_webagg import WebAggApplication
        loop = tornado.ioloop.IOLoop.instance()   # same loop WebAgg serves on

    def _mark_saved_goal(b, label):
        """Leave a thin outline + label of a goal that has already been computed."""
        c_tl, r_tl = w2px(b[0], b[3])
        c_br, r_br = w2px(b[2], b[1])
        ax.add_patch(mpatches.Rectangle(
            (min(c_tl, c_br), min(r_tl, r_br)), abs(c_br - c_tl), abs(r_br - r_tl),
            linewidth=1.2, edgecolor="cyan", facecolor="none", zorder=8))
        ax.text(min(c_tl, c_br), min(r_tl, r_br) - 3, label, color="cyan",
                fontsize=8, zorder=8, va="bottom")

    def _finish(b, outcome):
        """Runs on the UI thread once a computation has ended."""
        state["busy"] = False
        if isinstance(outcome, Exception):
            status.set_text(f"Not computed: {outcome}")
        else:
            stem, max_ttr = outcome
            results.append(stem)
            _mark_saved_goal(b, stem.rsplit("-", 1)[-1])
            status.set_text(f"Saved {stem}  (max TTR {max_ttr:.1f} s)   —   "
                            f"pick another goal, or Quit")
        fig.canvas.draw_idle()

    def _run(b):
        try:
            outcome = on_compute(b)
        except ValueError as e:     # expected problems (e.g. bad goal): message only
            outcome = e
        except Exception as e:      # keep the picker alive on unexpected errors
            import traceback; traceback.print_exc()
            outcome = e
        if UI_MODE == "browser":
            loop.add_callback(_finish, b, outcome)
        else:
            _finish(b, outcome)

    def on_done(_):
        if state["busy"]:
            return
        b = _bbox_world()
        if b is None:
            status.set_text("No goal yet — click two corners first")
            fig.canvas.draw_idle()
            return
        state["busy"] = True
        status.set_text("Computing TTR ... (progress in terminal)")
        fig.canvas.draw_idle()
        if UI_MODE == "browser":
            # compute in a worker thread so the page stays responsive
            import threading
            threading.Thread(target=_run, args=(b,), daemon=True).start()
        else:
            fig.canvas.draw(); fig.canvas.flush_events()
            _run(b)

    def on_quit(_):
        if state["busy"]:
            status.set_text("Still computing — wait for it to finish, then Quit")
            fig.canvas.draw_idle()
            return
        status.set_text("Stopped — you can close this page.")
        fig.canvas.draw_idle()
        if UI_MODE == "browser":
            def _stop():
                loop.stop()
                WebAggApplication.started = False
            loop.call_later(0.3, _stop)   # let the last status reach the browser
        else:
            plt.close(fig)

    state["busy"] = False
    results = []

    fig.canvas.mpl_connect("button_press_event", on_click)
    sl_w.on_changed(on_slider)
    sl_h.on_changed(on_slider)
    btn_done.on_clicked(on_done)
    btn_quit.on_clicked(on_quit)

    if UI_MODE == "browser":
        WebAggApplication.initialize()
        print("=" * 70)
        print(f"  Goal picker running in your browser:  "
              f"http://127.0.0.1:{WebAggApplication.port}/")
        print("  (VS Code Remote-SSH forwards this port automatically; otherwise use")
        print(f"   ssh -L {WebAggApplication.port}:127.0.0.1:{WebAggApplication.port} <server>)")
        print("  Pick a goal and press 'Compute TTR' — repeat as often as you like.")
        print("  Press 'Quit' (or Ctrl+C here) when done.")
        print("=" * 70, flush=True)

    plt.show()   # blocks until 'Quit' is pressed (or the window is closed)
    plt.close(fig)
    return results


# ── STEP 2 — rasterise obstacles ─────────────────────────────────────────────

def rasterize_obstacles(lines, circles, x_range, y_range, resolution):
    xmin, xmax = x_range; ymin, ymax = y_range
    Nx = int(np.ceil((xmax - xmin) / resolution)) + 1
    Ny = int(np.ceil((ymax - ymin) / resolution)) + 1
    x_arr = np.linspace(xmin, xmax, Nx)
    y_arr = np.linspace(ymin, ymax, Ny)
    grid  = np.zeros((Nx, Ny), dtype=bool)

    for (x1, y1), (x2, y2) in lines:
        n = max(int(np.hypot(x2-x1, y2-y1) / (resolution * 0.4)) + 1, 2)
        for t in np.linspace(0, 1, n):
            ix = int(np.round((x1 + t*(x2-x1) - xmin) / resolution))
            iy = int(np.round((y1 + t*(y2-y1) - ymin) / resolution))
            if 0 <= ix < Nx and 0 <= iy < Ny:
                grid[ix, iy] = True

    for cx, cy, r in circles:
        for i in range(Nx):
            for j in range(Ny):
                if np.hypot(x_arr[i]-cx, y_arr[j]-cy) <= r:
                    grid[i, j] = True

    return grid, x_arr, y_arr


# ── STEP 3 — SDF ──────────────────────────────────────────────────────────────

def compute_sdf(grid: np.ndarray, resolution: float) -> np.ndarray:
    """SDF > 0 = free space, SDF < 0 = inside obstacle."""
    d_free = distance_transform_edt(~grid) * resolution
    d_obs  = distance_transform_edt( grid) * resolution
    return d_free - d_obs


# ── STEP 4A — TTR via scikit-fmm (default) ───────────────────────────────────

def ttr_via_fmm(sdf, x_arr, y_arr, bbox, speed, resolution):
    """
    bbox : [xmin, ymin, xmax, ymax] in world metres — the rectangular goal region.
    Cells whose centre falls inside the bbox are seeded with TTR = 0.
    """
    import skfmm
    XX, YY = np.meshgrid(x_arr, y_arr, indexing="ij")
    spd    = np.where(sdf >= 0, speed, 0.0)
    inside = ((XX >= bbox[0]) & (XX <= bbox[2]) &
              (YY >= bbox[1]) & (YY <= bbox[3]))
    phi    = np.where(inside, -1.0, 1.0)
    ttr    = np.array(skfmm.travel_time(
                 np.ma.MaskedArray(phi, mask=(spd == 0)),
                 speed=spd, dx=resolution))
    return np.where(spd == 0, np.inf, ttr)


# ── STEP 4B — TTR via optimized_dp (requires conda + heterocl) ───────────────

def ttr_via_odp(sdf, x_arr, y_arr, bbox, speed, resolution, epsilon=1e-4):
    """
    bbox : [xmin, ymin, xmax, ymax] in world metres.
    Uses ShapeRectangle so the target set is an axis-aligned box.
    """
    from odp.Grid import Grid
    from odp.Shapes import ShapeRectangle
    from odp.dynamics import Plane2D
    from odp.solver import TTRSolver

    Nx, Ny = sdf.shape
    g          = Grid(minBounds=np.array([x_arr[0], y_arr[0]]),
                      maxBounds=np.array([x_arr[-1], y_arr[-1]]),
                      dims=2,
                      pts_each_dim=np.array([Nx, Ny]))
    target_set = ShapeRectangle(g,
                                target_min=np.array([bbox[0], bbox[1]]),
                                target_max=np.array([bbox[2], bbox[3]]))
    agent      = Plane2D(vxMin=-speed, vxMax=speed,
                         vyMin=-speed, vyMax=speed, uMode="min")
    ttr = TTRSolver(agent, g, [target_set, sdf], epsilon)
    return np.where(ttr >= 999, np.inf, ttr)


# ── STEP 5 — visualise result ─────────────────────────────────────────────────

def _draw_goal_rect(ax, bbox, **kw):
    """Draw the goal rectangle on a world-coord axes."""
    xmin, ymin, xmax, ymax = bbox
    rect = mpatches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor="yellow", facecolor="yellow",
        alpha=0.30, zorder=6, **kw)
    ax.add_patch(rect)


def next_output_stem(scene: str) -> str:
    """Return '<scene>-ttr-<n>' with the smallest n not already used in OUTPUT."""
    n = 0
    while (OUTPUT / f"{scene}-ttr-{n}.npz").exists() or \
          (OUTPUT / f"{scene}-ttr-{n}.png").exists():
        n += 1
    return f"{scene}-ttr-{n}"


def visualize_result(result: dict):
    scene  = result["scene_name"]
    ttr    = result["ttr"]
    sdf    = result["sdf"]
    grid   = result["obs_grid"]
    lines  = result["obs_lines"]
    peds   = result["trajectories"]
    x_arr  = result["x_arr"]
    y_arr  = result["y_arr"]
    bbox   = result["bbox"]           # [xmin, ymin, xmax, ymax]
    extent = [x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]]

    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    # Object-oriented Figure (not pyplot) so this can run from the picker's
    # worker thread without opening a new window / browser figure.
    from matplotlib.figure import Figure
    fig  = Figure(figsize=(18, 6))
    axes = fig.subplots(1, 3)
    fig.suptitle(
        f"{scene}  —  goal  "
        f"({bbox[0]:.2f}, {bbox[1]:.2f}) → ({bbox[2]:.2f}, {bbox[3]:.2f}) m  "
        f"[{bbox[2]-bbox[0]:.2f} × {bbox[3]-bbox[1]:.2f} m]",
        fontsize=11, fontweight="bold")

    # panel 1 — scene (world coords)
    ax = axes[0]
    ax.set_title("Scene + trajectories")
    ax.imshow(grid.T, origin="lower", extent=extent,
              cmap="gray_r", alpha=0.35, interpolation="nearest")
    for (p1, p2) in lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=2.5)
    for i, traj in enumerate(list(peds.values())[:40]):
        ax.plot(traj[:,0], traj[:,1], "-",
                color=plt.cm.tab10(i % 10), lw=0.8, alpha=0.5)
    _draw_goal_rect(ax, bbox)
    ax.plot(cx, cy, "*", color="yellow", ms=10, mec="darkorange", mew=1, zorder=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # panel 2 — SDF
    ax = axes[1]
    ax.set_title("SDF  (green = free, red = obstacle)")
    im = ax.imshow(sdf.T, origin="lower", extent=extent,
                   cmap="RdYlGn",
                   norm=TwoSlopeNorm(vmin=sdf.min(), vcenter=0, vmax=sdf.max()),
                   interpolation="bilinear")
    ax.contour(x_arr, y_arr, sdf.T, levels=[0], colors="k", linewidths=1.2)
    fig.colorbar(im, ax=ax, label="SDF (m)", shrink=0.85)
    _draw_goal_rect(ax, bbox)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    # panel 3 — TTR
    ax = axes[2]
    ax.set_title("Time-To-Reach (s)")
    ttr_vis = np.where(np.isinf(ttr), np.nan, ttr)
    vmax    = np.nanpercentile(ttr_vis, 95)
    im = ax.imshow(ttr_vis.T, origin="lower", extent=extent,
                   cmap="plasma_r", vmin=0, vmax=vmax,
                   interpolation="bilinear")
    ax.contourf(x_arr, y_arr, sdf.T, levels=[-100, 0],
                colors=["black"], alpha=0.4)
    for (p1, p2) in lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "w-", lw=2)
    fig.colorbar(im, ax=ax, label="TTR (s)", shrink=0.85)
    _draw_goal_rect(ax, bbox)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    fig.tight_layout()
    out_png = OUTPUT / f"{result.get('out_stem', scene + '-ttr')}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved → {out_png}")


# ── STEP 6 — query TTR at trajectory waypoints ───────────────────────────────

def query_ttr(result: dict) -> dict:
    """Returns {ped_id: (T,) TTR values along each trajectory}."""
    interp = RegularGridInterpolator(
        (result["x_arr"], result["y_arr"]), result["ttr"],
        method="linear", bounds_error=False, fill_value=np.inf)
    return {pid: interp(traj) for pid, traj in result["trajectories"].items()}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Interactive goal picker + TTR computation for ETH/UCY",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scene",   default="ETH-Univ", choices=list(SCENE_CFG.keys()))
    p.add_argument("--speed",   type=float, default=1.5,  help="max pedestrian speed m/s")
    p.add_argument("--res",     type=float, default=0.15, help="grid resolution m")
    p.add_argument("--backend", default="fmm", choices=["fmm", "odp"],
                   help="fmm = scikit-fmm (default)  |  odp = optimized_dp (needs conda)")
    p.add_argument("--goal", type=float, nargs=4, default=None,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="goal rectangle in world metres; skips the interactive picker "
                        "(required when no display is available)")
    args = p.parse_args()

    cfg = SCENE_CFG[args.scene]
    if args.goal is None and UI_MODE is None:
        p.error("no display and no browser backend (install `tornado`), so the "
                "goal picker cannot open. Pass --goal XMIN YMIN XMAX YMAX instead.")

    # ── 1. build world grid (goal-independent, done once) ─────────────────────
    peds    = load_trajectories(cfg)
    lines, circles = load_obstacles(cfg)
    all_pts = np.vstack(list(peds.values()))
    margin  = 1.5
    xmin, ymin = all_pts.min(0) - margin
    xmax, ymax = all_pts.max(0) + margin
    # Also cover the whole background image, so any goal that can be clicked
    # in the picker lies inside the TTR grid.
    H_img = np.loadtxt(cfg["path"] / cfg["H"])
    img_h, img_w = np.array(Image.open(cfg["path"] / cfg["bg"])).shape[:2]
    corners = np.array([rowcol_to_world(r, c, H_img)
                        for r in (0, img_h - 1) for c in (0, img_w - 1)])
    xmin, ymin = np.minimum([xmin, ymin], corners.min(0))
    xmax, ymax = np.maximum([xmax, ymax], corners.max(0))

    print(f"World bounds: x=[{xmin:.2f},{xmax:.2f}]  y=[{ymin:.2f},{ymax:.2f}]")

    # ── 2. occupancy grid ─────────────────────────────────────────────────────
    print("Rasterising obstacles ...")
    obs_grid, x_arr, y_arr = rasterize_obstacles(
        lines, circles, (xmin, xmax), (ymin, ymax), args.res)
    print(f"  Grid: {obs_grid.shape}  obstacle cells: {obs_grid.sum()}")

    # ── 3. SDF ────────────────────────────────────────────────────────────────
    print("Computing SDF ...")
    sdf = compute_sdf(obs_grid, args.res)
    print(f"  SDF range: [{sdf.min():.2f}, {sdf.max():.2f}] m")

    def compute_and_save(bbox):
        """TTR for one goal → saved as <scene>-ttr-<n>.npz/.png. Returns (stem, max TTR)."""
        bbox = np.asarray(bbox, dtype=float)
        print(f"\nGoal bbox: ({bbox[0]:.3f}, {bbox[1]:.3f}) → "
              f"({bbox[2]:.3f}, {bbox[3]:.3f}) m  "
              f"size {bbox[2]-bbox[0]:.2f} × {bbox[3]-bbox[1]:.2f} m")

        # goal must cover at least one free grid cell, otherwise TTR is empty
        in_x = (x_arr >= bbox[0]) & (x_arr <= bbox[2])
        in_y = (y_arr >= bbox[1]) & (y_arr <= bbox[3])
        if not (sdf[np.ix_(in_x, in_y)] > 0).any():
            print("  Skipped: goal covers no free grid cell — it is outside the "
                  f"computed area (x=[{x_arr[0]:.1f},{x_arr[-1]:.1f}], "
                  f"y=[{y_arr[0]:.1f},{y_arr[-1]:.1f}] m), inside an obstacle, "
                  "or smaller than --res.")
            raise ValueError("goal covers no free cell (outside area / in obstacle)"
                             " — pick another goal")

        # ── 4. TTR ────────────────────────────────────────────────────────────
        print(f"Computing TTR via '{args.backend}' ...")
        if args.backend == "fmm":
            ttr = ttr_via_fmm(sdf, x_arr, y_arr, bbox, args.speed, args.res)
        else:
            ttr = ttr_via_odp(sdf, x_arr, y_arr, bbox, args.speed, args.res)
        finite = ttr[np.isfinite(ttr)]
        if finite.size == 0:
            print("  Skipped: the solver found no reachable cell for this goal.")
            raise ValueError("no cell can reach this goal — pick another goal")
        print(f"  Done — max TTR: {finite.max():.2f} s  reachable cells: {len(finite)}")

        # ── 5. save: each run gets the next free number <scene>-ttr-<n> ───────
        out_stem = next_output_stem(args.scene)
        result = dict(
            ttr=ttr, sdf=sdf, obs_grid=obs_grid, obs_lines=lines,
            x_arr=x_arr, y_arr=y_arr, trajectories=peds,
            scene_name=args.scene, bbox=bbox, out_stem=out_stem,
        )
        out_npz = OUTPUT / f"{out_stem}.npz"
        np.savez(out_npz, ttr=ttr, sdf=sdf, x_arr=x_arr, y_arr=y_arr, bbox=bbox,
                 speed=args.speed, res=args.res, backend=args.backend)
        print(f"Arrays saved → {out_npz}")
        visualize_result(result)

        # ── 6. query TTR along trajectories ───────────────────────────────────
        ttr_trajs = query_ttr(result)
        print("── TTR at start / end of first 5 pedestrian trajectories ──")
        for pid, vals in list(ttr_trajs.items())[:5]:
            fin = vals[np.isfinite(vals)]
            if len(fin):
                print(f"  Ped {pid:5d}:  "
                      f"start={vals[0]:.2f}s  end={vals[-1]:.2f}s  "
                      f"mean={fin.mean():.2f}s")
        return out_stem, float(finite.max())

    # ── 7. goal from CLI (one run) or interactive picker (many runs) ──────────
    if args.goal is not None:
        gx0, gy0, gx1, gy1 = args.goal
        compute_and_save([min(gx0, gx1), min(gy0, gy1),
                          max(gx0, gx1), max(gy0, gy1)])
        return

    print(f"\nOpening scene picker for {args.scene} ...")
    print("  Click 1     = first corner of goal rectangle")
    print("  Click 2     = opposite corner")
    print("  Sliders     = fine-tune width / height")
    print("  Right-click = clear and start over")
    print("  Compute TTR = compute & save this goal (picker stays open)")
    print("  Quit        = exit\n")
    saved = pick_goal_interactive(args.scene, cfg, compute_and_save)
    print(f"\nFinished. {len(saved)} TTR result(s) saved in {OUTPUT}: "
          f"{', '.join(saved) if saved else '-'}")


if __name__ == "__main__":
    main()
