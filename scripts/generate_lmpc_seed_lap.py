"""Generate an initial LMPC safe set (D^0) by synthesizing one feasible lap.

The LMPC controller needs a historical safe set before it can drive: the paper
does not expect it to invent the first lap from empty data. Rather than *driving*
a seed lap (which launches from rest, giving a dense low-speed cluster near s=0
that leaves LMPC unable to accelerate), this script *synthesizes* a lap that
follows a reference line at its speed profile, so the safe set is at-speed
everywhere including the start.

Two sources, selected with --source:
  centerline (default) -- follow the LMPC trajectory table's centerline
                          (e_y = e_psi = 0), speed/curvature from the table.
  raceline             -- follow example_waypoints.csv, projected onto the
                          centerline for e_y/e_psi, speed from its vx profile.

The output is NOT a geometric trajectory. Each row is one sample of the LMPC
closed-loop state/input history:

    lap, s, e_y, e_psi, v_x, v_y, omega, lon, delta, k, t

Load it with ``LMPCController.load_initial_lap(<csv>)`` before driving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controllers.lmpc.binding import CenterlineTrack, GymVehicleState

# Matches kVehicleMass / default wheelbase in the native controller.
VEHICLE_MASS = 3.47
WHEELBASE = 0.33
# Floor on the reference speed so time integration and force reconstruction stay
# finite where a profile dips to zero.
MIN_SPEED = 0.5


def synthesize_centerline(table: np.ndarray, max_speed: float) -> dict[str, np.ndarray]:
    """Follow the centerline exactly: e_y = e_psi = 0, speed/curvature from table."""
    s = table[:, 6]
    v = np.clip(table[:, 4], MIN_SPEED, max_speed)
    k = table[:, 5]
    e_y = np.zeros_like(s)
    e_psi = np.zeros_like(s)
    v_y = np.zeros_like(s)
    omega = v * k
    return {
        "s": s,
        "e_y": e_y,
        "e_psi": e_psi,
        "v_x": v,
        "v_y": v_y,
        "omega": omega,
        "k": k,
    }


def synthesize_raceline(table: np.ndarray, raceline_csv: str) -> dict[str, np.ndarray]:
    """Follow the raceline, projected onto the centerline for e_y / e_psi."""
    track = CenterlineTrack(table[:, 0].tolist(), table[:, 1].tolist(), True)
    rl = np.atleast_2d(np.loadtxt(raceline_csv, delimiter=";", skiprows=1))
    # columns: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    s = np.empty(rl.shape[0])
    e_y = np.empty(rl.shape[0])
    e_psi = np.empty(rl.shape[0])
    for i, row in enumerate(rl):
        rs = track.to_racing_state(
            GymVehicleState(row[1], row[2], row[3], row[5], 0.0, 0.0)
        )
        s[i], e_y[i], e_psi[i] = rs.s, rs.e_y, rs.e_psi
    order = np.argsort(s)
    s = s[order]
    e_y = e_y[order]
    e_psi = e_psi[order]
    v = np.maximum(rl[order, 5], MIN_SPEED)
    k = rl[order, 4]
    v_y = np.zeros_like(s)
    omega = v * k
    return {
        "s": s,
        "e_y": e_y,
        "e_psi": e_psi,
        "v_x": v,
        "v_y": v_y,
        "omega": omega,
        "k": k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("centerline", "raceline"), default="centerline"
    )
    parser.add_argument(
        "--trajectory",
        default="outputs/lmpc_trajectories/f110_gym_centerline.txt",
        help="17-column LMPC trajectory table (centerline + speed + curvature).",
    )
    parser.add_argument(
        "--raceline",
        default="maps/custom/f110_gym_10/example_waypoints.csv",
        help="Raceline followed when --source raceline.",
    )
    parser.add_argument("--output", default="outputs/lmpc_seed_laps/f110_gym_seed.csv")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=1.0e9,
        help="Cap the seed speed profile so D^0 stays dynamically feasible on "
        "the followed line (the paper's safe set must be a feasible lap).",
    )
    args = parser.parse_args()

    table = np.atleast_2d(np.loadtxt(args.trajectory, dtype=np.float64))
    if args.source == "centerline":
        lap = synthesize_centerline(table, args.max_speed)
    else:
        lap = synthesize_raceline(table, args.raceline)

    s, e_y, e_psi = lap["s"], lap["e_y"], lap["e_psi"]
    v_x, v_y, omega, k = lap["v_x"], lap["v_y"], lap["omega"], lap["k"]

    # Longitudinal acceleration along the path: dv/dt = v * dv/ds. The simplified
    # single-track model uses one longitudinal input scaled by 1000 internally.
    # u only feeds the error-dynamics regression, so an analytic reconstruction
    # is sufficient.
    dv_ds = np.gradient(v_x, s)
    accel = v_x * dv_ds
    force = VEHICLE_MASS * accel
    lon = force / 1000.0
    delta = np.arctan(WHEELBASE * k)

    # Timestamps by integrating dt = ds / v along the lap.
    ds = np.diff(s)
    dt = ds / v_x[:-1]
    t = np.concatenate([[0.0], np.cumsum(dt)])

    rows = np.column_stack(
        [np.zeros_like(s), s, e_y, e_psi, v_x, v_y, omega, lon, delta, k, t]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = "lap,s,e_y,e_psi,v_x,v_y,omega,lon,delta,k,t"
    np.savetxt(output, rows, delimiter=",", header=header, comments="")
    print(
        f"Wrote {rows.shape[0]} samples ({args.source}) to {output} "
        f"| v_x[{v_x.min():.2f},{v_x.max():.2f}] "
        f"e_y[{e_y.min():.2f},{e_y.max():.2f}]"
    )


if __name__ == "__main__":
    main()
