#ifndef F110_ROLLOUT_KERNEL_STEP_HPP_
#define F110_ROLLOUT_KERNEL_STEP_HPP_

#include "f110_rollout_kernel/types.hpp"

namespace f110_rollout_kernel {

F110StepResult step(const F110State &state, const F110Action &action,
                    const F110Params &params = F110Params{},
                    Integrator integrator = Integrator::RK4);

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_STEP_HPP_
