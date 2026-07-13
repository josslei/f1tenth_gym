#ifndef LMPC__INTEGRATORS__RK4_HPP_
#define LMPC__INTEGRATORS__RK4_HPP_

#include "integrators/common.hpp"

namespace lmpc
{
namespace integrators
{

// Classic 4th-order Runge-Kutta: 4 evaluations of f per step, at (x), (x
// half-stepped by k1), (x half-stepped by k2), (x full-stepped by k3),
// combined with the standard 1/6, 2/6, 2/6, 1/6 weights. u, u_prev, kappa
// are held fixed across the sub-steps (standard zero-order-hold assumption
// -- the control is genuinely constant over one discretization interval).
// What upstream (ref/Racing-LMPC-ROS2) actually uses in every real vehicle
// config (BARC, hawaii_gokart, iac_car all set integrator_type: "rk4"),
// unlike Euler which their code supports but no config selects.
class Rk4 final : public Integrator
{
public:
  casadi::SX operator()(
    const ContinuousDynamics & f,
    const casadi::SX & x,
    const casadi::SX & u,
    const casadi::SX & u_prev,
    const casadi::SX & kappa,
    double dt) const override
  {
    const casadi::SX k1 = f(x, u, u_prev, kappa);
    const casadi::SX k2 = f(x + (dt / 2.0) * k1, u, u_prev, kappa);
    const casadi::SX k3 = f(x + (dt / 2.0) * k2, u, u_prev, kappa);
    const casadi::SX k4 = f(x + dt * k3, u, u_prev, kappa);
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  }

  const char * name() const override {return "rk4";}
};

}  // namespace integrators
}  // namespace lmpc

#endif  // LMPC__INTEGRATORS__RK4_HPP_
