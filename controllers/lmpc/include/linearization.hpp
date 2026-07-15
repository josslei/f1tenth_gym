#ifndef LMPC__LINEARIZATION_HPP_
#define LMPC__LINEARIZATION_HPP_

#include <casadi/casadi.hpp>

#include "dynamics/common.hpp"
#include "integrators/common.hpp"

namespace lmpc {

// Per-stage affine dynamics x_{t+1} = A_t x_t + B_t u_t + C_t (DESIGN.md
// SS4/SS8 step 3c) -- the first-order Taylor expansion of the discrete step
// F(x, u) = Integrator(DynamicsModel)(x, u, dt) around a reference point
// z_bar = (x_ref, u_ref):
//
//   F(x, u) ~= F(x_ref, u_ref) + A(x - x_ref) + B(u - u_ref)
//            = A x + B u + [F(x_ref, u_ref) - A x_ref - B u_ref]
//                           \_________________________________/
//                                          C
//
// so C absorbs the linearization residual and makes the result a plain
// affine map in (x, u), which is what the QP (DESIGN.md SS4) needs -- A_t,
// B_t, C_t are parameters there, not decision variables.
struct LinearizedDynamics {
  casadi::DM x_next; // kStateDim x 1 -- F(x_ref, u_ref) itself, the SAME
                     // evaluation the Jacobians below come from; returned
                     // so callers rolling out a nominal trajectory can
                     // reuse this call instead of re-evaluating F()
                     // separately (recom.md: step_and_jacobians() computes
                     // x_next regardless of whether the caller wants the
                     // Jacobians, so a rollout-only pass was throwing this
                     // away and then recomputing the whole thing, Jacobians
                     // included, a second time).
  casadi::DM A;      // kStateDim x kStateDim
  casadi::DM B;      // kStateDim x kControlDim
  casadi::DM C;      // kStateDim x 1
};

// Builds the CasADi expression graph for F and its Jacobians ONCE (in the
// constructor) for a given (DynamicsModel, Integrator, dt) triple, then
// evaluates that same compiled Function repeatedly at different numeric
// reference points -- DESIGN.md SS8 step 3 runs this once per horizon stage
// per control step (N stages x every step), so building the graph freshly
// each call would be wasted symbolic work.
class Linearizer {
public:
  Linearizer(const dynamics::DynamicsModel &model,
             const integrators::Integrator &integrator, double dt) {
    using casadi::SX;

    const SX x = SX::sym("x", dynamics::kStateDim);
    const SX u = SX::sym("u", dynamics::kControlDim);
    const SX u_prev = SX::sym("u_prev", dynamics::kControlDim);
    const SX kappa = SX::sym("kappa", 1);

    const integrators::ContinuousDynamics f =
        [&model](const SX &x_, const SX &u_, const SX &u_prev_,
                 const SX &kappa_) {
          return dynamics::full_state_dynamics(model, x_, u_, u_prev_, kappa_);
        };

    const SX x_next = integrator(f, x, u, u_prev, kappa, dt);
    const SX jac_x = SX::jacobian(x_next, x);
    const SX jac_u = SX::jacobian(x_next, u);

    step_and_jacobians = casadi::Function(
        "step_and_jacobians", {x, u, u_prev, kappa}, {x_next, jac_x, jac_u},
        {"x", "u", "u_prev", "kappa"}, {"x_next", "jac_x", "jac_u"});
  }

  // Evaluates F and its Jacobians at (x_ref, u_ref, u_prev_ref, kappa_ref)
  // and folds them into the affine (A, B, C) form documented above --
  // x_next comes along for free (step_and_jacobians' own compiled Function
  // outputs it regardless of whether the caller wants the Jacobians), so
  // this single call also serves as the one-step rollout: callers building
  // a nominal trajectory (LMPCController::solve_once()) chain x_next from
  // stage stg into x_ref for stage stg+1, rather than rolling out via a
  // SEPARATE step()-only pass first and then re-linearizing (which used to
  // mean N redundant Jacobian evaluations per control step -- recom.md).
  LinearizedDynamics operator()(const casadi::DM &x_ref,
                                const casadi::DM &u_ref,
                                const casadi::DM &u_prev_ref,
                                double kappa_ref) const {
    const casadi::DMDict result =
        step_and_jacobians(casadi::DMDict{{"x", x_ref},
                                          {"u", u_ref},
                                          {"u_prev", u_prev_ref},
                                          {"kappa", kappa_ref}});

    const casadi::DM &x_next = result.at("x_next");
    const casadi::DM &A = result.at("jac_x");
    const casadi::DM &B = result.at("jac_u");
    const casadi::DM C =
        x_next - casadi::DM::mtimes(A, x_ref) - casadi::DM::mtimes(B, u_ref);

    return LinearizedDynamics{x_next, A, B, C};
  }

private:
  casadi::Function step_and_jacobians;
};

} // namespace lmpc

#endif // LMPC__LINEARIZATION_HPP_
