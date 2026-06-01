import math
import numpy as np
import sys
import time
from typing import Any, cast

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trajectory_planning_helpers as tph


def _calc_closed_splines_sparse(path: np.ndarray) -> tuple:
    """Calculate closed cubic splines using sparse linear algebra."""
    if not np.all(np.isclose(path[0], path[-1])):
        return tph.calc_splines.calc_splines(path=path)  # pyright: ignore[reportAttributeAccessIssue]

    el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    el_lengths = np.append(el_lengths, el_lengths[0])
    no_splines = path.shape[0] - 1
    scaling = el_lengths[:-1] / el_lengths[1:]

    row_count = no_splines * 4
    matrix = sp.lil_matrix((row_count, row_count), dtype=np.float64)
    b_x = np.zeros(row_count, dtype=np.float64)
    b_y = np.zeros(row_count, dtype=np.float64)

    for i in range(no_splines):
        row = i * 4
        col = row

        if i < no_splines - 1:
            scale = scaling[i]
            scale_sq = math.pow(scale, 2)

            matrix[row, col] = 1.0

            matrix[row + 1, col] = 1.0
            matrix[row + 1, col + 1] = 1.0
            matrix[row + 1, col + 2] = 1.0
            matrix[row + 1, col + 3] = 1.0

            matrix[row + 2, col + 1] = 1.0
            matrix[row + 2, col + 2] = 2.0
            matrix[row + 2, col + 3] = 3.0
            matrix[row + 2, col + 5] = -scale

            matrix[row + 3, col + 2] = 2.0
            matrix[row + 3, col + 3] = 6.0
            matrix[row + 3, col + 6] = -2.0 * scale_sq
        else:
            matrix[row, col] = 1.0

            matrix[row + 1, col] = 1.0
            matrix[row + 1, col + 1] = 1.0
            matrix[row + 1, col + 2] = 1.0
            matrix[row + 1, col + 3] = 1.0

        b_x[row] = path[i, 0]
        b_x[row + 1] = path[i + 1, 0]
        b_y[row] = path[i, 1]
        b_y[row + 1] = path[i + 1, 1]

    last_scale = scaling[-1]
    matrix[-2, 1] = last_scale
    matrix[-2, -3] = -1.0
    matrix[-2, -2] = -2.0
    matrix[-2, -1] = -3.0

    matrix[-1, 2] = 2.0 * math.pow(last_scale, 2)
    matrix[-1, -2] = -2.0
    matrix[-1, -1] = -6.0

    matrix_csr = matrix.tocsr()
    x_les = np.asarray(cast(Any, spla.spsolve(matrix_csr, b_x)), dtype=np.float64)
    y_les = np.asarray(cast(Any, spla.spsolve(matrix_csr, b_y)), dtype=np.float64)

    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

    return coeffs_x, coeffs_y, matrix_csr, normvec_normalized


def prep_track(
    reftrack_imp: np.ndarray,
    reg_smooth_opts: dict,
    stepsize_opts: dict,
    debug: bool = True,
    min_width: float | None = None,
    profile: bool = False,
    use_sparse_splines: bool = False,
) -> tuple:
    """
    Created by:
    Alexander Heilmeier

    Documentation:
    This function prepares the inserted reference track for optimization.

    Inputs:
    reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reg_smooth_opts:            parameters for the spline approximation
    stepsize_opts:              dict containing the stepsizes before spline approximation and after spline interpolation
    debug:                      boolean showing if debug messages should be printed
    min_width:                  [m] minimum enforced track width (None to deactivate)
    profile:                    boolean printing timings for expensive preprocessing stages
    use_sparse_splines:         boolean using sparse linear algebra for closed spline equations

    Outputs:
    reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
    a_interp:                   LES coefficients when calculating the splines
    coeffs_x_interp:            spline coefficients of the x-component
    coeffs_y_interp:            spline coefficients of the y-component
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # smoothing and interpolating reference track
    start_time = time.perf_counter()
    reftrack_interp = tph.spline_approximation.spline_approximation(  # pyright: ignore[reportAttributeAccessIssue]
        track=reftrack_imp,
        k_reg=reg_smooth_opts["k_reg"],
        s_reg=reg_smooth_opts["s_reg"],
        stepsize_prep=stepsize_opts["stepsize_prep"],
        stepsize_reg=stepsize_opts["stepsize_reg"],
        debug=debug,
    )
    if profile:
        elapsed = time.perf_counter() - start_time
        print(
            "prep_track: spline approximation "
            f"created {reftrack_interp.shape[0]} points in {elapsed:.3f} s",
            flush=True,
        )

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    start_time = time.perf_counter()
    if use_sparse_splines:
        coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = (
            _calc_closed_splines_sparse(refpath_interp_cl)
        )
    else:
        coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = (
            tph.calc_splines.calc_splines(path=refpath_interp_cl)  # pyright: ignore[reportAttributeAccessIssue]
        )
    if profile:
        elapsed = time.perf_counter() - start_time
        print(
            "prep_track: calc_splines "
            f"solved {reftrack_interp.shape[0]} splines in {elapsed:.3f} s",
            flush=True,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.perf_counter()
    normals_crossing = tph.check_normals_crossing.check_normals_crossing(  # pyright: ignore[reportAttributeAccessIssue]
        track=reftrack_interp, normvec_normalized=normvec_normalized_interp, horizon=10
    )
    if profile:
        elapsed = time.perf_counter() - start_time
        print(
            f"prep_track: normal crossing check finished in {elapsed:.3f} s",
            flush=True,
        )

    if normals_crossing:
        import matplotlib.pyplot as plt

        bound_1_tmp = reftrack_interp[
            :, :2
        ] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], axis=1)
        bound_2_tmp = reftrack_interp[
            :, :2
        ] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], axis=1)

        plt.figure()

        plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], "k-")
        for i in range(bound_1_tmp.shape[0]):
            temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
            plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

        plt.grid()
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.title("Error: at least one pair of normals is crossed!")

        plt.show()

        raise IOError(
            "At least two spline normals are crossed, check input or increase smoothing factor!"
        )

    # ------------------------------------------------------------------------------------------------------------------
    # ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    manipulated_track_width = False

    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

            if cur_width < min_width:
                manipulated_track_width = True

                # inflate to both sides equally
                reftrack_interp[i, 2] += (min_width - cur_width) / 2
                reftrack_interp[i, 3] += (min_width - cur_width) / 2

    if manipulated_track_width:
        print(
            "WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
            " order to match the requirements!",
            file=sys.stderr,
        )

    return (
        reftrack_interp,
        normvec_normalized_interp,
        a_interp,
        coeffs_x_interp,
        coeffs_y_interp,
    )


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
