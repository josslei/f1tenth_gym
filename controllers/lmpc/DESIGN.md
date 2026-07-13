# LMPC Design Notes

Pinned conventions for the from-scratch rebuild, derived directly from
`xue_et_al_lmpc.pdf` (Xue et al., "Learning Model Predictive Control with
Error Dynamics Regression for Autonomous Racing") and `ref/Racing-LMPC-ROS2/lmpc.tex`.
This is the source of truth for symbol ordering and index conventions — code
should mirror it exactly (named constants/enums, not magic indices).

Status: conventions pinned. The "dummy A^e/B^e/C^e" first pass (§8) is fully
wired end-to-end: nominal dynamics models (`include/dynamics/`), the
discretize+linearize utility (`include/linearization.hpp`), the D^0 safe-set
loader/K-NN query (`include/safe_set.hpp`), the FHOCP QP builder/solver
(`include/qp_builder.hpp`), the track curvature source (`include/track.hpp`),
and `LMPCController::control()` itself all implemented and verified to run
closed-loop against the real gym simulator (`f110_gym_10` map). At a short
horizon (N=20) it sustains 100+ control steps of forward progress before
qrqp genuinely fails to find a search direction; at the pinned horizon
(N=75) qrqp frequently reports "success" on the very first solve while
returning a control trajectory that violates its own box constraints -- a
known finite-but-garbage-success failure mode of solving an UNSCALED QP,
not a wiring bug (see the "Known limitation" note under Open items). A
bounds/regularity guard in `QpBuilder::solve()` catches this and reports
failure rather than silently applying a bad command. Error-dynamics
regression (§5/§6) is not implemented -- this pass genuinely skips 3a/3b as
designed.

## 1. State and control vectors

```
x = [vx, vy, omega, epsi, s, ey]^T   in R^6   (velocity block first, then pose)
u = [a, delta]^T                      in R^2   (long. accel, front steering angle)
```

Canonical indices (to be mirrored as named constants in code):

```
IDX_VX = 0, IDX_VY = 1, IDX_OMEGA = 2, IDX_EPSI = 3, IDX_S = 4, IDX_EY = 5
IDX_A  = 0, IDX_DELTA = 1
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

### Pinned: D (adapted to our state ordering, §1)

Upstream's actual implementation of the `(x^i_k - x)^T D (x^i_k - x)`
neighbor search (`safe_set.cpp`, `SSTrajectory::query(SSQuery)`) is **not** a
general weighted distance — it builds a 2D nearest-neighbor search (a CGAL
KD-tree) over exactly two state dimensions: `PX, PY` in their ordering
(`XIndex::PX=0, PY=1`), i.e. `s` and `ey`. Every other state dimension
(`yaw, vx, vy, vyaw`) is ignored for this query. In `D ⪰ 0` terms, that's
`D = diag(1, 1, 0, 0, 0, 0)` in *their* index order.

Translated to our own ordering (`x = [vx, vy, omega, epsi, s, ey]`, §1 —
`IDX_S=4, IDX_EY=5`):

```
D = diag(0, 0, 0, 0, 1, 1)   # nonzero only at IDX_S, IDX_EY
```

i.e. `(x^i_k - x)^T D (x^i_k - x) = (s^i_k - s)² + (ey^i_k - ey)²` — the
target-set/terminal-cost neighbor search is nearest-by-track-position only,
same behavior as upstream, expressed against our own index convention. This
reduces the safe-set query to a 2D nearest-neighbor problem (implementable
as a direct search or a KD-tree — an implementation choice, not a design
one, deferred until we're writing the actual query code).

## 3. Online FHOCP (paper eq. 4a–4f)

At time `k` of lap `j`, using `D^{j-1}`:

```
J_k^j(x_k, u_{k-1}, z̄_k; D^{j-1}) =

  min_{x,u,λ}  Σ_{t=0}^{N-1} [ 1_F(x_t) + c_u‖u_t‖² + c_Δu‖u_t - u_{t-1}‖² ]
               + J_N^{j-1}(x̄_{k+N}; D^{j-1})^T λ

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

Decision vector: `w = (x_0,...,x_N, u_0,...,u_{N-1}, λ) ∈ R^{(N+1)n + Nm + q}`,
`n=6, m=2, q=KP`.

```
Φ(w) = Σ_{t=0}^{N-1} [ 1_F(x_t) + c_u‖u_t‖² + c_Δu‖u_t - u_{t-1}‖² ] + J^T λ

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
(`opti.solver('qrqp', ...)` for correctness-first bring-up, `'osqp'` with
`polish=true` once it's working) — **not** `sqpmethod`. `sqpmethod` targets
general NLPs and re-linearizes internally, which is both redundant (we've
already linearized) and was the exact cause of a "0% success" dead end in
the previous C++ port.

**Command interface, no separate integration step.** `X[:, 1]` in the
solved trajectory already *is* the model-consistent one-step-ahead state —
it's produced by the dynamics equality constraint as part of the solve, at
zero marginal cost. The velocity command handed to `env.step` is simply
`X[IDX_VX, k]` for whichever preview index `k` we choose (`k=1` for the very
next step), not a hand-rolled Euler/RK integration. If a value were ever
needed outside a solve, `v_cmd = vx_0 + a*dt` is a single FMA — still O(1),
never a performance concern.

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
- The `vy` and `omega` rows' only control covariate is `delta` (never `a`).
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
- **OSQP** — sparse ADMM solver. Already known to build (`recom.md`:
  `WITH_OSQP`/`WITH_BUILD_OSQP`, plus a corrupted vendored patch that had to
  be rewritten). Robust to poor conditioning, but convergence to tight
  tolerance is slow — previous port measured mean solve times of
  ~60–120 ms at `N=75`, far over a 25 ms control period.
- **qpOASES** — dense active-set solver, source vendored *inside* CasADi
  itself (`external_packages/qpOASES`); enabling it is just
  `WITH_LAPACK=ON` + `WITH_QPOASES=ON`, no external fetch, cheapest of the
  three to build. Its real advantage (fast warm-started re-solves) only
  shows up on a **condensed** QP (states eliminated via the affine
  recursion, leaving only the ~150 control variables over the horizon, not
  the full ~650-variable multi-shooting problem). Fed the multi-shooting QP
  directly, it's an unstructured dense solver on a large sparse-structured
  matrix — no guarantee it beats OSQP. Testing it fairly requires building
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
  at configure time) — the same class of build risk OSQP already cost us
  once, plausibly worse since it's two chained builds instead of one.

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
6. Apply u_0* = [a*, delta*]. Converted to
   ControlCommand(steering=delta*, velocity=X[IDX_VX, 1]) in the Python
   wrapper -- no separate integration (§4).
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

**Known limitation (not yet fixed): unscaled-QP conditioning.** `qrqp` can
report a solve as "success" while returning a control trajectory that
violates its own box constraints (`QpBuilder::solve()`'s post-solve
regularity/bounds guard catches this and reports failure instead of
applying it) -- most reproducible on the very first solve of a run (rest
state, coarsest linearization) and more frequent at the pinned N=75 than at
a short test horizon (N=20 sustains 100+ closed-loop steps before a
genuine, non-garbage solver failure). This is the exact failure class this
project's memory documents at length from the prior (deleted) native LMPC
port, always eventually traced back to the QP having no variable scaling
(state/control magnitudes spanning very different orders -- `s` up to
~164m, `vx` O(1-20), `ey` O(1) -- left as-is in the Hessian/constraint
matrix). The "Overall variable-scaling convention" open item below is the
anticipated fix; not applied yet since it is a substantial, separately-
scoped body of tuning work, not part of getting the base mechanism wired.

## Open items (not yet pinned)

- Overall variable-scaling convention (`scale_x`, `scale_u`) — should be
  fixed once, up front, given how much of the previous C++ port's pain
  traced back to scaling being retrofitted late. (Note: upstream's own
  `scale_x_`/`scale_u_` are hardcoded in `racing_mpc.cpp`'s constructor,
  identical for BARC and full-scale IAC — `recom.md` already flagged this
  as ill-fitting BARC's own force scale, so these are *not* to be copied
  the way `K`/`P`/`Q` were; ours need deriving from our own vehicle's
  physical limits.)
- `h` (kernel bandwidth) and `ε` (ridge regularization) for the error
  regression — no upstream reference value exists at all (§6: upstream's
  regression path is never exercised), so these need to be picked and
  tuned from scratch, unlike `K`/`P`/`Q`/`D` above.
