#ifndef LMPC__INTEGRATORS__COMMON_HPP_
#define LMPC__INTEGRATORS__COMMON_HPP_

#include <functional>

#include <casadi/casadi.hpp>

namespace lmpc
{
namespace integrators
{

// A continuous-time dynamics function x_dot = f(x, u, u_prev, kappa) -- e.g.
// a lambda wrapping lmpc::dynamics::full_state_dynamics() for a chosen
// DynamicsModel. Kept as a callable rather than a pre-evaluated SX
// expression because an integrator (RK4 in particular) needs to evaluate it
// repeatedly at DIFFERENT symbolic x points within one step, not just once
// at the start of the interval.
using ContinuousDynamics = std::function<casadi::SX(
    const casadi::SX & x,
    const casadi::SX & u,
    const casadi::SX & u_prev,
    const casadi::SX & kappa)>;

// Turns a continuous x_dot = f(x, u, u_prev, kappa) into a discrete step
// x_{t+1} = F(x, u, u_prev, kappa, dt). Interchangeable (functor), same
// pattern as lmpc::dynamics::DynamicsModel, so the discretization scheme
// can be swapped without touching whatever calls it (the discretize +
// linearize step, controllers/lmpc/DESIGN.md SS8 step 3c).
class Integrator
{
public:
  virtual ~Integrator() = default;

  virtual casadi::SX operator()(
    const ContinuousDynamics & f,
    const casadi::SX & x,
    const casadi::SX & u,
    const casadi::SX & u_prev,
    const casadi::SX & kappa,
    double dt) const = 0;

  virtual const char * name() const = 0;
};

}  // namespace integrators
}  // namespace lmpc

#endif  // LMPC__INTEGRATORS__COMMON_HPP_
