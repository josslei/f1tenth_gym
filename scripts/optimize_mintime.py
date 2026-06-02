"""Generate a minimum-time optimized raceline from a track centerline.

The script takes a 4-column track CSV (x_m, y_m, w_tr_right_m, w_tr_left_m)
and produces a raceline CSV:

    s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2

Example:
    python scripts/optimize_mintime.py \\
        --track tracks/Spielberg/Spielberg_map.csv \\
        --output outputs/waypoints/berlin_mintime.csv \\
        --save_plot
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

_CONSOLE = Console()

_MODULE = Path(__file__).resolve().parent / "raceline_opt"
sys.path.insert(0, str(_MODULE))

import opt_mintime_traj.src.opt_mintime as _opt_mintime  # noqa: E402
import trajectory_planning_helpers.calc_ax_profile as calc_ax_profile  # noqa: E402
import trajectory_planning_helpers.calc_head_curv_an as calc_head_curv_an  # noqa: E402
import trajectory_planning_helpers.create_raceline as create_raceline  # noqa: E402
from helper_funcs_glob.src.import_track import import_track  # noqa: E402
import helper_funcs_glob.src.prep_track  # noqa: E402, used via dotted path later


def _default_params_path() -> Path:
    return Path("configs/raceline/f110.ini")


def _load_params(params_path: Path) -> dict[str, Any]:
    """Load vehicle and optimization parameters from an INI config."""
    cfg = configparser.ConfigParser()
    if not cfg.read(str(params_path)):
        raise FileNotFoundError(f"Could not read parameter config: {params_path}")
    p: dict[str, Any] = {}

    p["ggv_file"] = json.loads(cfg.get("GENERAL_OPTIONS", "ggv_file"))
    p["ax_max_machines_file"] = json.loads(
        cfg.get("GENERAL_OPTIONS", "ax_max_machines_file")
    )
    p["stepsize_opts"] = json.loads(cfg.get("GENERAL_OPTIONS", "stepsize_opts"))
    p["reg_smooth_opts"] = json.loads(cfg.get("GENERAL_OPTIONS", "reg_smooth_opts"))
    p["veh_params"] = json.loads(cfg.get("GENERAL_OPTIONS", "veh_params"))
    p["vel_calc_opts"] = json.loads(cfg.get("GENERAL_OPTIONS", "vel_calc_opts"))
    p["curv_calc_opts"] = json.loads(cfg.get("GENERAL_OPTIONS", "curv_calc_opts"))
    p["optim_opts"] = json.loads(cfg.get("OPTIMIZATION_OPTIONS", "optim_opts_mintime"))
    p["optim_opts"]["var_friction"] = None
    p["optim_opts"]["warm_start"] = False
    p["vehicle_params_mintime"] = json.loads(
        cfg.get("OPTIMIZATION_OPTIONS", "vehicle_params_mintime")
    )
    p["tire_params_mintime"] = json.loads(
        cfg.get("OPTIMIZATION_OPTIONS", "tire_params_mintime")
    )
    p["pwr_params_mintime"] = json.loads(
        cfg.get("OPTIMIZATION_OPTIONS", "pwr_params_mintime")
    )
    p["vehicle_params_mintime"]["wheelbase"] = (
        p["vehicle_params_mintime"]["wheelbase_front"]
        + p["vehicle_params_mintime"]["wheelbase_rear"]
    )
    return p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--track",
        required=True,
        type=Path,
        help="Track CSV (x_m, y_m, w_tr_right_m, w_tr_left_m)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output raceline CSV",
    )
    p.add_argument(
        "--params",
        type=Path,
        default=_default_params_path(),
        help="Parameter config file (default: configs/raceline/f110.ini)",
    )
    p.add_argument(
        "--save_plot",
        action="store_true",
        help="Save the raceline track visualization next to the output CSV",
    )
    p.add_argument(
        "--num_laps",
        type=int,
        default=1,
        help="Number of laps for powertrain-aware mintime optimization",
    )
    p.add_argument(
        "--mintime_max_iter",
        type=int,
        default=1000,
        help="Maximum IPOPT iterations",
    )
    p.add_argument(
        "--ipopt_tol",
        type=float,
        default=None,
        help="IPOPT convergence tolerance (default: 1e-4)",
    )
    p.add_argument(
        "--stepsize_prep",
        type=float,
        default=None,
        help="Override config preprocessing interpolation spacing in meters",
    )
    p.add_argument(
        "--stepsize_reg",
        type=float,
        default=None,
        help="Override config optimization-node spacing in meters",
    )
    p.add_argument(
        "--stepsize_interp_after_opt",
        type=float,
        default=None,
        help="Override config output trajectory spacing in meters",
    )
    p.add_argument(
        "--step_non_reg",
        type=int,
        default=None,
        help="Override config non-regular point sampling skip count",
    )
    p.add_argument(
        "--width_opt",
        type=float,
        default=None,
        help=(
            "Override optimization safety width in meters. Larger values keep "
            "the optimized car center farther from track boundaries."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    _CONSOLE.log("Loading vehicle parameters...")
    pars = _load_params(args.params)

    if args.stepsize_prep is not None:
        pars["stepsize_opts"]["stepsize_prep"] = args.stepsize_prep
    if args.stepsize_reg is not None:
        pars["stepsize_opts"]["stepsize_reg"] = args.stepsize_reg
    if args.stepsize_interp_after_opt is not None:
        pars["stepsize_opts"]["stepsize_interp_after_opt"] = (
            args.stepsize_interp_after_opt
        )
    if args.step_non_reg is not None:
        pars["optim_opts"]["step_non_reg"] = args.step_non_reg
    if args.width_opt is not None:
        pars["optim_opts"]["width_opt"] = args.width_opt
    if args.ipopt_tol is not None:
        pars["optim_opts"]["ipopt_tol"] = args.ipopt_tol

    t0 = time.perf_counter()

    imp_opts = {
        "flip_imp_track": False,
        "set_new_start": False,
        "new_start": np.array([0.0, 0.0]),
        "min_track_width": None,
        "num_laps": args.num_laps,
    }

    _CONSOLE.log(f"Importing track from [cyan]{args.track}[/]...")
    track = import_track(str(args.track), imp_opts, pars["veh_params"]["width"])

    _CONSOLE.log("Preparing reference track...")
    reftrack_interp, normvec, a_interp, coeffs_x, coeffs_y = (
        helper_funcs_glob.src.prep_track.prep_track(
            reftrack_imp=track,
            reg_smooth_opts=pars["reg_smooth_opts"],
            stepsize_opts=pars["stepsize_opts"],
            debug=False,
            min_width=None,
            profile=True,
            use_sparse_splines=True,
        )
    )
    _CONSOLE.log(
        "Prepared reference line with "
        f"[green]{reftrack_interp.shape[0]}[/] optimization nodes"
    )

    pars["optim_opts"]["max_iter"] = args.mintime_max_iter

    _CONSOLE.log("Building and solving mintime NLP with IPOPT...")
    t_opt_start = time.perf_counter()
    with _CONSOLE.status("Building/solving mintime NLP...", spinner="dots"):
        alpha_opt, v_opt, reftrack_interp, a_interp_tmp, normvec = (
            _opt_mintime.opt_mintime(
                reftrack=reftrack_interp,
                coeffs_x=coeffs_x,
                coeffs_y=coeffs_y,
                normvectors=normvec,
                pars=pars,
                tpamap_path="",
                tpadata_path=None,
                export_path=None,
                print_debug=False,
                plot_debug=False,
            )
        )
    t_opt_elapsed = time.perf_counter() - t_opt_start
    _CONSOLE.log(f"Optimization solved in [green]{t_opt_elapsed:.1f}s[/]")

    _CONSOLE.log("Building race trajectory...")
    (
        raceline_interp,
        _,
        coeffs_x_opt,
        coeffs_y_opt,
        spline_inds,
        t_vals,
        s_points,
        spline_lengths,
        el_lengths,
    ) = create_raceline.create_raceline(
        refline=reftrack_interp[:, :2],
        normvectors=normvec,
        alpha=alpha_opt,
        stepsize_interp=pars["stepsize_opts"]["stepsize_interp_after_opt"],
    )

    psi, kappa = calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x_opt,
        coeffs_y=coeffs_y_opt,
        ind_spls=spline_inds,
        t_spls=t_vals,
    )

    s_splines = np.cumsum(spline_lengths)
    s_splines = np.insert(s_splines, 0, 0.0)
    vx = np.interp(s_points, s_splines[:-1], v_opt)

    vx_closed = np.append(vx, vx[0])
    ax = calc_ax_profile.calc_ax_profile(
        vx_profile=vx_closed,
        el_lengths=el_lengths,
        eq_length_output=False,
    )

    traj = np.column_stack((s_points, raceline_interp, psi, kappa, vx, ax))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        args.output,
        traj,
        delimiter="; ",
        header="s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2",
        comments="# ",
        fmt="%.7f",
    )

    elapsed = time.perf_counter() - t0
    _CONSOLE.log(
        "Exported [green]{}[/] waypoints in [green]{:.1f}s[/] to [cyan]{}[/]".format(
            traj.shape[0], elapsed, args.output
        )
    )

    if args.save_plot:
        _save_plot(
            track,
            reftrack_interp,
            raceline_interp,
            args.output.with_suffix(".png"),
        )


def _save_plot(
    track_raw: np.ndarray,
    reftrack: np.ndarray,
    raceline: np.ndarray,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)

    raw_bound_r = _track_boundaries(track_raw, side="right")
    raw_bound_l = _track_boundaries(track_raw, side="left")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        raw_bound_l[:, 0],
        raw_bound_l[:, 1],
        color="0.25",
        linewidth=1.0,
        label="track boundary",
    )
    ax.plot(
        raw_bound_r[:, 0],
        raw_bound_r[:, 1],
        color="0.25",
        linewidth=1.0,
    )
    ax.plot(
        track_raw[:, 0],
        track_raw[:, 1],
        color="0.65",
        linestyle="--",
        linewidth=0.8,
        label="centerline",
    )
    ax.plot(
        raceline[:, 0],
        raceline[:, 1],
        color="tab:red",
        linewidth=1.4,
        label="raceline",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _CONSOLE.log(f"Saved plot → [cyan]{save_path}[/]")


def _track_boundaries(track: np.ndarray, *, side: str) -> np.ndarray:
    """Compute boundary coordinates from a track CSV [x, y, w_tr_right, w_tr_left]."""
    pts = track[:, :2]
    tangents = np.diff(pts, axis=0)
    tangents = np.vstack([tangents, tangents[0]])  # close loop
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    lengths = np.hypot(normals[:, 0], normals[:, 1])
    lengths = np.where(lengths == 0, 1.0, lengths)
    normals /= lengths[:, np.newaxis]

    if side == "right":
        return pts + track[:, 2:3] * normals
    return pts - track[:, 3:4] * normals


if __name__ == "__main__":
    main()
