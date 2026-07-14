# LMPC Design Notes

Pinned conventions for the from-scratch rebuild, derived directly from
`xue_et_al_lmpc.pdf` (Xue et al., "Learning Model Predictive Control with
Error Dynamics Regression for Autonomous Racing") and `ref/Racing-LMPC-ROS2/lmpc.tex`.
This is the source of truth for symbol ordering and index conventions — code
should mirror it exactly (named constants/enums, not magic indices).

Status: conventions pinned. The "dummy A^e/B^e/C^e" first pass (§8) is wired
end-to-end. The QP uses normalized state/control variables, normalized
safe-set cost-to-go, and state-aware safe-set neighbors over
`[vx, epsi, s, ey]`. The terminal state is a hard convex combination of all
`K` selected points from every stored lap. Error-dynamics regression (§5/§6)
is not implemented -- this pass genuinely skips 3a/3b as designed.

The QP control convention is `[a, delta]`. The Python wrapper converts solved
acceleration into Gym's target-velocity action by inverting Gym's proportional
velocity controller. Control effort and rate penalties are applied directly
to scaled controls; Hessian regularization and `solve_limited()` remain in the
QP path. Closed-loop state (2026-07-13, after the fixes documented at the
end of this file): **full standing-start laps complete** (44.52s on D^0
alone) under the lap-as-iteration scheme, and each completed lap is fed
back into the safe set (`add_lap`). The remaining blocker for actual
iteration-over-iteration improvement is the §5/§6 error-dynamics
regression: without it the min-time QP's sprint-and-brake plans outrun the
nominal model's cornering accuracy a little above the demonstrated speeds
(see "Lap-as-iteration" below).

## 1. State and control vectors

```
x = [vx, vy, omega, epsi, s, ey]^T   in R^6   (velocity block first, then pose)
u = [a, delta]^T                      in R^2   (long. accel, steering angle)
```

Canonical indices (to be mirrored as named constants in code):

```
IDX_VX = 0, IDX_VY = 1, IDX_OMEGA = 2, IDX_EPSI = 3, IDX_S = 4, IDX_EY = 5
IDX_A = 0, IDX_DELTA = 1
```

This is the paper's own ordering (Section II), not the old port's
`[s, ey, epsi, vx, vy, omega]` convention borrowed from upstream's
`single_track_planar_model`. Adopting it directly keeps every index in this
document, and in eq. (7)'s regression slices, a literal match to the paper —
no silent reindexing between docs and code.

## 2. Safe set: target set + terminal cost-to-go

Given data from previous laps `D^{j-1} = {(x^i_0,u^i_0),...,(x^i_{T^i},u^i_{T^i})}_{i<j}`:

- For a query state `x`, take the `K` nearest neighbors from each of the
  previous `P` laps under the weighted distance `(x^i_k - x)^T D (x^i_k - x)`,
  `D ⪰ 0`. Stack them into `X^j(x; D^j) ∈ R^{n×KP}`.
- Local convex terminal (target) set:
  `X_N^j(x; D^j) = { x̄ | ∃ λ ∈ R^{KP}, 0 ≤ λ ≤ 1, 1^T λ = 1, X^j(x;D^j) λ = x̄ }`.
- Cost-to-go for each stored sample `x^i_k` is `T^i - k` (steps remaining to
  the finish line in that lap). Collect these into `J_N^j(x; D^j) ∈ R^{KP}`.
- Local terminal cost:
  `Q_N^j(x̄, x; D^j) = min_λ J_N^j(x;D^j)^T λ` s.t. the same `0≤λ≤1, 1^Tλ=1,
  X^j(x;D^j)λ = x̄` constraints.

`D` (neighbor distance metric) and `K`/`P` are ours to choose — the paper
only requires `D ⪰ 0`. Pinned below from upstream (`ref/Racing-LMPC-ROS2`,
BARC config — the 1/10-scale reference, apples-to-apples with our vehicle).

### Pinned: K, P (from `barc_lmpc.param.yaml`)

```
K = 32   # neighbors taken per lap        (upstream: num_ss_pts_per_lap)
P = 3    # number of prior laps searched  (upstream: max_lap_stored)
```

`K·P = 96` is the resulting safe-set size (the `λ` dimension in §3/§4). These
are real, wired-up values from upstream's own working controller (unlike `Q`
below) — copied directly, no unit conversion needed since BARC is the same
class of vehicle we are.

**Config requirement:** whatever field names the actual `LmpcConfig` /
YAML use for these two, they must be comented explicitly as "K: neighbors
per lap" and "P: laps kept in the safe set" at the point of definition —
not just `num_ss_pts_per_lap`/`max_lap_stored` copied verbatim with no
annotation. This is so a future reader (or future us) can map the config
knob back to the paper's symbol without re-deriving it from upstream again.

### Safe-set query metric

The terminal query uses normalized distance over `[vx, epsi, s, ey]`.
Including speed and heading error avoids selecting points that are nearby on
the track but cannot represent the predicted terminal dynamic state. The
normalization uses the same per-state scales as the QP.

## 3. Online FHOCP (paper eq. 4a–4f)

At time `k` of lap `j`, using `D^{j-1}`:

```
J_k^j(x_k, u_{k-1}, z̄_k; D^{j-1}) =

  min_{x,u,λ} Σ_{t=0}^{N-1} [ 1_F(x_t) + c_u ||u_t||^2
                                  + c_d_u ||u_t-u_{t-1}||^2 ]
                   + (J_N^{j-1}/J_scale)^T λ

  s.t.  x_0 = x_k,  u_{-1} = u_{k-1}
        x_{t+1} = A(z̄_{k+t}; D^{j-1}) x_t + B(z̄_{k+t}; D^{j-1}) u_t + C(z̄_{k+t}; D^{j-1}),
                                                              t = 0,...,N-1
        x_t ∈ X,  u_t ∈ U,                                  t = 0,...,N
        X^{j-1}(x̄_{k+N}; D^{j-1}) λ = x_N
        0 ≤ λ ≤ 1,  1^T λ = 1
```

`z̄_k = {z̄_k, ..., z̄_{k+N}}` is the linearization sequence, normally the
previous MPC solution (receding-horizon shift). `X = {x | -W/2 ≤ ey ≤ W/2}`,
`U = {u | u_l ≤ u ≤ u_u}`. `1_F` is the min-time indicator — `0` once the
vehicle has crossed the finish line, `1` otherwise; in practice this is a
constant `+1` per stage since within a single horizon the vehicle essentially
never finishes mid-plan.

## 4. Translation into a solver-facing QP

Decision vector: `w = (x_0,...,x_N, u_0,...,u_{N-1}, λ)`,
`n=6, m=2, q=KP`.

```
Φ(w) = Σ [ 1_F(x_t) + c_u ||u_t||^2 + c_d_u ||u_t-u_{t-1}||^2 ]
       + (J/J_scale)^T λ

g_eq(w) = 0:
    x_0 - x_k = 0
    x_{t+1} - A_t x_t - B_t u_t - C_t = 0,   t = 0,...,N-1
    X_ss λ - x_N = 0
    1^T λ - 1 = 0

g_ineq(w) ≤ 0:
    x_t ∈ X,  u_t ∈ U,  0 ≤ λ ≤ 1
```

**`A_t, B_t, C_t` are parameters, not decision variables** — computed once
per control step (nominal Jacobian + learned error correction, §6 below)
*before* the solve, held fixed during it. Because the dynamics are affine and
fixed at solve time, and the cost is quadratic, this is a genuine convex QP,
not a general NLP. Solve with a conic backend through CasADi's `Opti` stack
(`opti.solver('qrqp', ...)` for correctness-first bring-up) — **not**
`sqpmethod`. `sqpmethod` targets
general NLPs and re-linearizes internally, which is both redundant (we've
already linearized) and was the exact cause of a "0% success" dead end in
the previous C++ port.

**Command interface.** The QP solves `[a, delta]`, while Gym consumes target
velocity and target steering angle. The Python wrapper inverts Gym's
proportional velocity controller so the requested setpoint produces `a` before
Gym applies its acceleration constraints. `c_u`/`c_d_u` are SCALAR (the
paper's own formulation -- a plain L2 norm on the control vector, not a
per-component-weighted `R`/`Rd` matrix): acceleration and steering are
already comparable by the time this cost applies, since the control vector
is normalized to O(1) by `QpScaling::u` before it's used anywhere.

## 5. Error dynamics regression (paper eq. 5–8) — pinned sparsity

The true dynamics are `x+ = f(x,u) + e(x,u)`, `f` nominal, `e` unknown. We
learn a local affine model of `e` about a reference `z̄ = (x̄, ū)`:
`x+ - x̂+ = A^e x + B^e u + C^e`, giving the full ATV model
`A = A^f + A^e`, `B = B^f + B^e`, `C = C^f + C^e`.

**We only ever learn corrections for the velocity states `[vx, vy, omega]`.
`epsi`, `s`, `ey` are never regression targets and never regression
covariates, in any of the three rows** — the paper treats the kinematic/pose
components as already well-modeled by the nominal model. Concretely:

```
row vx:    Ae[0, 0:3]·x[0:3] + Be[0,0]·a     + Ce[0]  =  vx+ - v̂x+
row vy:    Ae[1, 0:3]·x[1:3] + Be[1,1]·delta + Ce[1]  =  vy+ - v̂y+
row omega: Ae[2, 0:3]·x[2:3] + Be[2,1]·delta + Ce[2]  =  omega+ - ω̂z+
```

(indices per §1: `x[0:3]` is the `[vx, vy, omega]` block.)

- Every row's covariates are restricted to `[vx, vy, omega]` — never
  `epsi, s, ey`.
- The `vx` row's only control covariate is `a` (never `delta`).
- The `vy` and `omega` rows' only control covariate is `delta`.
- Each row's regressor `Γ_l = [Ae_row(0:3), Be_scalar, Ce_scalar]` is a
  5-vector.

Assembled matrices (rows/cols 3,4,5 = `epsi, s, ey`) are structurally zero
outside the `3×3` velocity block and the two control columns noted above:

```
Ae = [[p1 p2 p3 0 0 0]      Be = [[p4  0 ]      Ce = [p5]
      [q1 q2 q3 0 0 0]            [0   q4]            [q5]
      [r1 r2 r3 0 0 0]            [0   r4]            [r5]
      [ 0  0  0 0 0 0]            [0   0 ]            [0 ]
      [ 0  0  0 0 0 0]            [0   0 ]            [0 ]
      [ 0  0  0 0 0 0]]           [0   0 ]]           [0 ]
```

Rows 3–5 (`epsi, s, ey`) of `Ae, Be, Ce` are identically zero: those state
derivatives are always taken purely from the nominal model, never learned.

## 6. Combined regression solve (shared Gram matrix)

The three rows share one `M`-nearest-neighbor query of `D^{j-1}` around
`z̄` (one query, one weighted-distance kernel `w(m) = K(‖z̄ - z_m‖²_Q)`,
Epanechnikov, bandwidth `h`) — reused identically for all three rows, per
the paper's literal text ("we query `D^{j-1}` to find the `M` nearest
neighbors ... of `z̄`," a single query).

**Pinned: `Q = I`.** Matches both the paper's own stated convention
(`lmpc.tex`: "In the LMPC paper, the Euclidean norm is used, so `Q = I`")
and upstream's code (`safe_set.cpp`'s unused `RegQuery` path computes plain
`dists = sqrt(sum((z_data - query.x)^2))`, no weighting). One caveat worth
keeping in mind, not a reason to deviate: upstream's `RegQuery`/regression
path is never actually constructed or called anywhere in that repository —
`racing_mpc.cpp`'s real `A/B/C` come straight from the nominal model's
Jacobian with no learned correction, ever. So `Q = I` is inherited as the
paper's own stated default, not as a value upstream validated in a working
controller — `h` (bandwidth) and `ε` (ridge) still have no upstream
reference value at all (§ open items) and will need to be picked and tuned
by us from scratch.

Because the weight is shared, the expensive part — the `O(M·d²)` weighted
reduction over neighbor samples — can be computed **once** instead of three
times:

1. Build pooled covariates `Z_full ∈ R^{M×6}`: columns `[vx, vy, omega, a,
   delta, 1]`. Pooled targets `Y_full ∈ R^{M×3}`: columns `[vx-residual,
   vy-residual, omega-residual]`.
2. One weighted matrix product: `P = Z_full^T @ W @ [Z_full | Y_full]`
   (`W = diag(w(1),...,w(M))`) — a single BLAS call producing the `6×6` Gram
   block `G` and the `6×3` cross-term block `C`.
3. Per row, slice `G`'s `5×5` submatrix (drop the row/column of the control
   the row doesn't use) and the matching `5×1` slice of `C`, add ridge
   `λI`, and solve the resulting `5×5` system independently.

This reproduces eq. (7)/(8) exactly (same per-row sparsity as the paper) —
it does **not** use `ref/Racing-LMPC-ROS2/lmpc.tex`'s fully dense
`Θ(Z^TWZ + λI) = Y^TWZ` closed form as-is, since that form has no mechanism
to exclude `delta` from the `vx` row or `a` from the `vy`/`omega` rows and
would introduce spurious cross-terms eq. (7) explicitly rules out. The
speedup instead comes from sharing the one `O(M)` data pass across all three
rows; the per-row `5×5` solves are `O(d³)` and negligible regardless of how
they're structured.

## 7. Solver candidates (not yet chosen)

The per-step QP (§4) is solver-agnostic by construction (`opti.solver(name,
...)` takes a plain string), so swapping backends never requires touching the
QP-building code — only the acceptance/status-check logic needs to be
written against `opti.stats()` rather than backend-specific status strings,
so a swap doesn't silently break success detection. Candidates, checked
against the vendored `thirdparty/casadi/CMakeLists.txt`:

- **qrqp** — CasADi's built-in dense active-set QP solver. Always available,
  no plugin/build work. Correctness-first default for initial bring-up.
- **qpOASES** — dense active-set solver, source vendored *inside* CasADi
  itself (`external_packages/qpOASES`); enabling it is just
  `WITH_LAPACK=ON` + `WITH_QPOASES=ON`, no external fetch, cheapest of the
  three to build. Its real advantage (fast warm-started re-solves) only
  shows up on a **condensed** QP (states eliminated via the affine
  recursion, leaving only the ~150 control variables over the horizon, not
  the full ~650-variable multi-shooting problem). Fed the multi-shooting QP
  directly, it's an unstructured dense solver on a large sparse-structured
  matrix — no guarantee it is competitive. Testing it fairly requires building
  the condensed `H`/`g` first, which is real (if fairly mechanical)
  implementation work, not just a solver-string swap. Worth doing anyway:
  the affine recursion needed to compute `A_t,B_t,C_t` already produces most
  of the intermediate products condensing needs.
- **HPIPM** — interior-point method built around the block-tridiagonal
  structure of exactly this problem shape (small state dim, long horizon,
  repeated warm-started solves); does the equivalent of condensing
  internally via a Riccati recursion, without hand-built condensing or its
  dense O(N³)-ish cost. The default choice in real-time MPC tooling
  (acados, etc.) for this problem class, and the more likely genuine win of
  the two — but needs two chained external builds not currently vendored
  anywhere in `thirdparty/` (`BLASFEO` via `WITH_BUILD_BLASFEO`, then
  `HPIPM` on top via `WITH_BUILD_HPIPM`, both `ExternalProject_Add` fetches
  at configure time), with the added risk of two chained external builds.

**Plan:** qrqp for correctness-first bring-up. Once the QP is verified
correct, try qpOASES on a condensed formulation first (cheap to build, and
condensing is useful groundwork regardless of solver). Reach for HPIPM only
if that isn't fast enough, given its higher, separate build cost.

## 8. Main online control-loop flow (per control step `k` of lap `j`)

```
1. Measure x_k, carry u_{k-1}.
2. Get the linearization sequence z̄_{k:k+N} (previous solve's own
   solution, shifted -- or a naive rollout on the very first solve).
3. For each horizon stage t = 0..N-1, independently:
     a. Query D^{j-1} for the M nearest neighbors of z̄_{k+t} (§6) --
        a FRESH query per stage, not once per control step, since eq.
        4c indexes A(z̄_{k+t}; D^{j-1}) by t.
     b. Run the three weighted ridge regressions (§5/§6) -> A^e_t, B^e_t, C^e_t.
     c. Discretize + linearize a DynamicsModel
        (controllers/lmpc/include/dynamics/) around z̄_{k+t}
        -> A^f_t, B^f_t, C^f_t.
     d. A_t = A^f_t + A^e_t,  B_t = B^f_t + B^e_t,  C_t = C^f_t + C^e_t.
4. Query D^{j-1} once more for the terminal point x̄_{k+N} (§2's D/K/P --
   a DIFFERENT query than 3a: position-only, for the safe-set target set
   and cost-to-go, not the regression).
5. Build and solve the QP (§4) over x_{0:N}, u_{0:N-1}, λ, using the
   per-stage A_t,B_t,C_t from step 3 and the terminal data from step 4.
6. Apply `u_0* = [a*, delta*]`; the Python wrapper converts `a*` to Gym's
   target-velocity command.
7. Observe the new state, go to 1.
8. On lap completion, the closed-loop trajectory (recorded every step via
   a generic record_step-style mechanism, not a lap-end-only pass)
   becomes D^j.
```

### First implementation pass: dummy `A^e, B^e, C^e` (steps 3a/3b skipped)

Steps 3a/3b are skipped for the first pass: `A_t = A^f_t, B_t = B^f_t, C_t =
C^f_t` (no learned correction). **Everything else stays real**, including
step 4 (the terminal safe-set query, using the driven D⁰ seed lap already
collected in `outputs/lmpc_seed_laps/`) — this tests whether the base MPC
mechanism (the QP, the receding horizon, the terminal cost-to-go actually
pulling the car forward) works at all, before the learning layer is added.

Steps 3c/3d are **not** skipped and are **not** hoisted out of the per-stage
loop — `A_t, B_t, C_t` still varies stage-to-stage even with no error
correction, since `z̄_{k+t}` does. Structuring step 3 as a per-stage loop
from the start means turning on regression later is additive (fill in
3a/3b inside the existing loop) rather than a restructuring.

**Implementation order, this pass:**
1. Nominal dynamics models — done (`GymDynamics`, `ExtendedKinematicDynamics`).
2. Discretize + linearize utility — done (`include/linearization.hpp`'s
   `Linearizer`: builds the CasADi Jacobian graph once per
   (DynamicsModel, Integrator, dt), evaluates it at any numeric z̄).
3. Safe-set loader — done (`include/safe_set.hpp`'s `SafeSet`: loads one or
   more driven laps, K-NN query under §2's `D`).
4. QP construction (§4) — done (`include/qp_builder.hpp`'s `QpBuilder`:
   builds the multi-shooting Opti('conic') graph once, re-parametrized and
   re-solved every control step).
5. Wired into `LMPCController::update()`/`control()` — done
   (`src/lmpc_controller.cpp`): per-stage linearization loop, terminal
   safe-set query, QP solve, receding-horizon warm-start shift.

**Structural feasibility fixes.** State/control decisions are normalized by
`QpScaling`, and safe-set cost-to-go is divided by D^0's fixed maximum value.
Control effort and rate costs act directly on scaled `U`, making their weights
independent of physical units and the seed lap's length. The terminal neighbor
query uses normalized `[vx, epsi, s, ey]` distance rather than position alone.
It returns exactly `K` points per stored lap, so `q = K*P`; `Lambda` is the
full q-dimensional decision variable with `0<=Lambda<=1`, `1^T Lambda=1`, and
the hard equality `x_N = X_ss Lambda`. No terminal slack or Lambda ridge is
added. Because q grows as completed laps are stored, `QpBuilder` is rebuilt
once after each `add_lap()`, while the cost-to-go scale remains fixed to D^0.

**`SIM_TIMESTEP` switched from an accidental 0.01 to the originally-
intended 0.025 (2026-07-13)** -- `gym/f110_gym/envs/f110_env.py`'s
`F110Env` defaults `timestep=0.01` and no caller was ever passing an
explicit override, so every closed-loop measurement above (the 292/459-step
numbers) was silently running at `dt=0.01`, not the `N=75`-paired `0.025`
this project's own config default (`LmpcConfig::dt`) already assumed.
Fixed by passing `timestep=0.025` explicitly at the two LMPC call sites
(`runs/lmpc_drive.py`, `scripts/lmpc_collect_seed_lap.py`) rather than
editing that vendored default (`CLAUDE.md`: treat `gym/` as a black box).

This single change cascaded into two more real, previously-latent bugs,
both found and fixed, not just the dt mismatch itself:
- **Centerline waypoint spacing (~0.10m) was nearly equal to one control
  step's travel distance (0.0875m at `dt=0.025`, 3.5 m/s)**, causing the
  nearest-waypoint heading reference to jump almost every control step.
  Regenerated the centerline at ~0.02m spacing
  (`scripts/generate_centerline.py --target-spacing 0.02`, same winding
  direction/enclosed area verified against the old file, just a different
  arbitrary start point on the closed loop -- harmless, everything derives
  `xy[0]` dynamically).
- **A hardcoded `SEARCH_WINDOW=200` (an INDEX count, not a physical
  distance) in `controllers/stanley.py`, `controllers/pure_pursuit.py`, and
  `controllers/lmpc/lmpc.py`** silently became physically narrower as
  waypoint density changed (~20m -> ~4m once the centerline above got 5x
  denser), causing `nearest_waypoint_index` to lose track. Fixed by
  deriving each instance's own window from its actual waypoint spacing
  (`SEARCH_WINDOW_METERS=20.0` target, computed once in `__init__`).

**Stanley replaced with Pure Pursuit for D^0 collection.** Even after both
fixes above, Stanley genuinely spun out at one corner at `dt=0.025`
(confirmed against raw simulator pose: yaw diverged ~45 degrees from the
path heading within ~0.3s while steering sat pinned at its bound) --
correction happens 2.5x less often than at the old `dt=0.01`, giving tire
slip time to develop before Stanley's next correction arrives. Retuning
Stanley's own gain did NOT fix it in either direction (lower gain measured
WORSE cross-track error, a classic "too sluggish for the track's curvature"
failure, not a stability fix). Pure Pursuit's geometric "aim at a point
ahead" law has no heading-error feedback term to react sharply in the
first place -- swapped in
(`scripts/lmpc_collect_seed_lap.py`/`controllers/pure_pursuit.py`, same
`DynamicLookaheadDistance` values `runs/waypoint_drive.py` already
validated), and the resulting D^0 completes cleanly (`mean|ey|=0.018m`,
no spin).

## Closed-loop integration fixes (2026-07-13)

Six independent root causes, fixed in one pass, each diagnosed by direct
measurement (the env-gated `LMPC_DEBUG_STAGES`/`LMPC_DEBUG_TERMINAL` dumps
added to `lmpc_controller.cpp`/`qp_builder.cpp`) -- took the closed loop
from "car never moves" to 46.5s (~the full lap) of stable 3.5-4.0 m/s
driving. Each fix EXPOSED the next failure; the order below is causal.
`recom.md` (repo root) has the longer narrative and the remaining-problem
list.

1. **Gym PID inversion needs gym's own 2/10 gain branch** (`lmpc.py`).
   gym's `pid()` uses `kp = 10*a_max/...` only when `current_speed > 0`;
   at a standing start (`0 > 0` is false) it uses `2*...` -- a constant-10
   inversion under-delivers launch acceleration exactly 5x. Current speed
   must be gym's signed `state[3]` (what obs `linear_vels_x` carries),
   never a magnitude. Setpoint clipped to `[v_min, v_max]`.

2. **First solve warm-starts from the recorded D^0 segment, not a
   zero-control rollout** (`SafeSet::trajectory_segment`,
   `LMPCController::seed_warm_start_from_safe_set`). From rest, a `u=0`
   rollout parks the whole horizon at the start line; the terminal query
   then finds D^0's own launch samples (near-max J, zero mismatch), every
   cost term is ~0, and nothing pulls the car forward. `SafeSet` now keeps
   the CSV's `a`/`delta` columns (`SafeSetSample::u`/`has_control`).
   Related: `scripts/lmpc_collect_seed_lap.py` drives an unrecorded
   warm-up lap and records lap 2, keeping launch transients out of D^0.

3. **`x_warm` is re-rolled out from the MEASURED state before every solve;
   only `u_warm` shifts** (`rollout_warm_states_from_current`,
   `shift_warm_start`). Shifting predicted states and patching only column
   0 with the measurement leaves stages 1..N as stale predictions that
   drift from reality with nothing pulling them back -- the accumulated
   inconsistency is what produced qrqp's "Failed to calculate search
   direction". (Post-hoc clamping of slightly-violating solutions was
   tried and reverted, twice across two architectures: x_traj is solved
   self-consistently against the UNCLAMPED u_traj, so clamping corrupts
   the next warm start worse than failing does.)

4. **`GymDynamics` models BOTH of gym's regimes, C^1-blended over
   v in [0.5, 1.0]** (kinematic weight exactly 1 below 0.5). The
   tire-force branch's `1/v..1/v^2` terms have O(10^3) eigenvalues below
   ~1 m/s, and the Euler rollout at `dt=0.025` is violently unstable there
   under any nonzero steering (measured: omega 0 -> 19 -> -190 rad/s in
   two stages, feeding garbage linearization references to the QP). Gym's
   own `|v| < 0.5` kinematic switch is a REGIME boundary (the tire model
   is invalid at low slip velocity, not merely stiff); the nominal model
   must respect it, not smooth over it with a tiny epsilon floor.

5. **Per-stage steering-rate constraint
   `|delta_t - delta_{t-1}| <= sv_max*dt`** (`LmpcConfig::sv_max = 3.2`,
   `QpBounds::ddelta_max`). Gym's steering actuator is rate-limited to
   ~0.08 rad per 0.025s step; without the constraint the plan flips full
   lock (~0.84 rad) in one step, 10x beyond executable, chattering the
   real simulator into its documented low-speed steering divergence (raw
   sim omega hit -420 rad/s). `LaunchSteeringGuard` is ALSO reinstated in
   `runs/lmpc_drive.py` (zero steer below 2.0 m/s, latch at 3.0, same as
   the seed collector): gym diverges at low speed even under small
   rate-limited steering -- a plant defect no QP-side change reaches.

6. **Safe-set locality metric is NOT the QP variable scaling**
   (`LMPCController::safe_set_query_scale`). Normalizing the query
   metric's `s` by track length (166m) makes position nearly free (2m of
   s = 0.012), so "nearest" is decided by noise in the other coordinates
   -- measured: the terminal query returned samples 2m BEHIND the
   terminal reference, the cost-to-go stopped pulling forward, and the
   car decelerated 3.8 -> 1.5 m/s until the QP failed. The query now uses
   `s` in raw meters (§2's `D = diag(0,0,0,0,1,1)` spirit, with the
   normalized `vx/epsi/ey` terms as mild tie-breakers); the QP keeps its
   own conditioning scale. Two different concepts -- keep them separate.

Robustness on top: a solve failure discards the warm start, reseeds
`u_warm` from the D^0 segment at the current state, and retries once
(`solve_once`); qrqp's known small-box-violation false-convergence is
accepted up to 2% of each bound's range WITHOUT mutating the solution
(gym's actuator clamps at the physical limit regardless).

## Lap-as-iteration and the start/finish seam (2026-07-13, second pass)

The seam blocker (terminal reference running past D^0's data end; across
the line the query landed on lap-start samples with `J ~= max`, the
cost-to-go pointed backward, the car braked 3.7 -> 2.3 m/s and the QP
died at the start pose) is resolved by two coupled decisions:

1. **Lap-as-iteration, not a wrapped-lap copy.** Crossing the line ends
   iteration `j` exactly as the paper defines it: the runner
   (`runs/lmpc_drive.py`) records each pre/post-transition plant state,
   computes `J_k = T - k` at the crossing, calls
   `LMPCController::add_lap`, resets the sim + controller, and relaunches
   from rest. `s` stays non-periodic; `J` keeps its single-task
   steps-to-finish meaning; the reset initial state matches D^0's own
   standing-start initial condition. The rejected alternative -- virtual
   forward candidates `(s + L, J - T_lap)` -- fixes `J`'s continuity but
   not the STATE's: the copied lap-start samples still have `vx ~= 0`, so
   a flying finish would still brake toward a standing start. Continuous
   multi-lap driving needs flying-lap data plus unwrapped-s/periodic-kappa
   /continuing-task-J redefinitions -- out of scope.
   Crashed/truncated laps are never added (the safe set's meaning rests
   on every stored trajectory reaching the finish); `SafeSet` keeps at
   most `kMaxLaps = 3` laps, evicting the oldest.

   One completed lap with T transitions stores T+1 states `x_0..x_T`, T
   realized inputs, and T+1 costs with `J_k=T-k`. `x_T` is the first
   post-step finish-crossing state. Both D^0 and online laps reconstruct
   `a_k` from raw scalar-speed finite differences and use the plant's actual
   pre-transition steering state, never commanded steering.
2. **Finish-mode terminal set for the last horizons.** Even within one
   lap, the terminal reference passes the stored data's end a few
   horizons before the crossing; clamping the query onto each lap's final
   samples anchors `x_N` BEHIND the reference with `J` already ~0, making
   "park on the data endpoint" the optimum (the measured braking above).
   When `s_ref > SafeSet::data_end_s()`, `solve_once` swaps in a forward
   absorbing finish set: queried anchors keep their dynamic-state rows,
   the `s` row is matched to the reference itself (no backward pull),
   `J = 0`.

Also in the second pass: the `ey` box is now SOFT (per-stage slack,
exact-plus-quadratic penalty `ey_slack_l1/l2` -- a hard box made the QP
instantly infeasible whenever `x_0` or the reachable tube left the
corridor), and the runner has a controlled-brake fallback (~4.8 m/s^2,
holding last steering, LMPC re-attempted every step) for double solve
failures.

**Measured after the second pass:** iteration 0 completes the full lap
(44.52s, 1781 steps) and grows the safe set. Iteration 1 launches faster
but self-terminates mid-lap: with a low control-effort weight (`c_u`,
formerly a separate near-zero `c_a`), a soft terminal anchor, and
J rewarding a further-along `x_N`, each solve rationally plans
"sprint now, brake at horizon end" and the receding horizon re-defers the
brake -- correct min-time behavior IF the model is right, but the
uncorrected nominal model overestimates cornering grip slightly above the
demonstrated 3.5-4 m/s (one-step vx prediction error is ~0; the failures
are lateral: bursts to ~6.1 m/s ending in real slides, epsi -0.6 rad).
This is precisely the mismatch §5/§6's regression corrects, which makes
the regression the next milestone -- `add_lap` already stores the
`(x_k, u_k, x_{k+1})` transitions it needs.

## Open items (not yet pinned)

- **§5/§6 error-dynamics regression -- the blocker for improvement past
  D^0** (measurements above). `h` (kernel bandwidth) and `ε` (ridge
  regularization) have no upstream reference value at all (§6: upstream's
  regression path is never exercised), so both need picking and tuning
  from scratch, unlike `K`/`P`/`Q`/`D` above.
- Gym's speed-dependent acceleration cap (`accl_constraints`'s
  `v_switch = 7.319` scaling) is not modeled -- the QP's constant
  `a_max = 9.51` becomes optimistic once speeds rise well past D^0's 3.5.
  Freeze `a_max,t = a_max * min(1, v_switch/v_ref,t)` per stage at the
  linearization reference to stay convex.
- Realized steering lags commanded by up to one rate-step; the faithful
  formulation adds delta as a state with `u = [a, sv]`. Biggest schema
  change -- last.
