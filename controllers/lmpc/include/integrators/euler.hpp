#ifndef LMPC__INTEGRATORS__EULER_HPP_
#define LMPC__INTEGRATORS__EULER_HPP_

#include "integrators/common.hpp"

namespace lmpc
{
namespace integrators
{

// Forward Euler: one evaluation of f per step. Cheapest option; error grows
// with how much f actually changes within dt (DESIGN.md SS8 discussion).
class Euler final : public Integrator
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
    return x + dt * f(x, u, u_prev, kappa);
  }

  const char * name() const override {return "euler";}
};

}  // namespace integrators
}  // namespace lmpc

#endif  // LMPC__INTEGRATORS__EULER_HPP_
