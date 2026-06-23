#ifndef F110_ROLLOUT_KERNEL_DYNAMICS_HPP_
#define F110_ROLLOUT_KERNEL_DYNAMICS_HPP_

#include "types.hpp"

namespace f110_rollout_kernel {

double accl_constraints(double vel, double accl, double v_switch, double a_max,
                        double v_min, double v_max);
double steering_constraint(double steering_angle, double steering_velocity,
                           double s_min, double s_max, double sv_min,
                           double sv_max);
ControlVector pid(double speed, double steer, double current_speed,
                  double current_steer, double max_sv, double max_a,
                  double max_v, double min_v);
StateVector vehicle_dynamics_ks(const StateVector &x,
                                const ControlVector &u_init,
                                const F110Params &params);
StateVector vehicle_dynamics_st(const StateVector &x,
                                const ControlVector &u_init,
                                const F110Params &params);

} // namespace f110_rollout_kernel

#endif // F110_ROLLOUT_KERNEL_DYNAMICS_HPP_
