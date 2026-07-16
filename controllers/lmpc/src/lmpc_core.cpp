/**
 * lmpc_core.cpp — ROS-free extraction of src/LMPC.cpp for use with f1tenth_gym.
 *
 * The controller logic is transcribed VERBATIM from src/LMPC.cpp
 * (mlab-upenn/LearningMPC). What changed (see src_gym/README.md for the full
 * honest list):
 *   - ROS pub/sub/tf plumbing removed; state comes in via set_state(), the
 *     control comes out as step()'s return value instead of an
 *     AckermannDriveStamped publish.
 *   - Parameters arrive as a std::map (loaded from the same Lmpc_params.yaml
 *     on the python side) instead of ros::NodeHandle::getParam.
 *   - The occupancy grid arrives as an int8 array (converted from the map png
 *     with ROS map_server's rule on the python side) instead of a /map topic.
 *   - visualization functions and the RRT-obstacle callbacks (map_callback /
 *     rrt_path_callback) are dropped; the latter are inactive in the standard
 *     racing setup (nothing publishes "path_found").
 *   - first_run_ and time_ are explicitly initialized (uninitialized in the
 *     original C++ — undefined behavior that happened to work).
 *   - OSQP verbosity off; the per-step couts of applyControl dropped.
 *
 * The original headers track.h / occupancy_grid.h / spline.h / CSVReader.h /
 * car_params.h are included UNMODIFIED via shim headers in cpp/ros_shim.
 */

#include <math.h>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <map>
#include <algorithm>

#include <LearningMPC/track.h>
#include <Eigen/Sparse>
#include "OsqpEigen/OsqpEigen.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <LearningMPC/car_params.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

const int nx = 6;
const int nu = 2;

using namespace std;
using namespace Eigen;

struct Sample{
    Matrix<double,nx,1> x;
    Matrix<double,nu,1> u;
    double s;
    int time;
    int iter;
    int cost;
};

// verbatim from LMPC.cpp
void wrap_angle(double& angle, const double angle_ref){
    while(angle - angle_ref > M_PI) {angle -= 2*M_PI;}
    while(angle - angle_ref < -M_PI) {angle += 2*M_PI;}
}

class LMPCCore{
public:
    LMPCCore(const std::map<std::string,double>& params,
             py::array_t<int8_t> grid_data,
             uint32_t width, uint32_t height,
             double resolution, double origin_x, double origin_y,
             const std::string& waypoint_file,
             const std::string& init_data_file,
             double x0, double y0, double yaw0){

        getParameters(params);

        /* init_occupancy_grid() equivalent: grid passed in, then inflated
         * exactly as the original (occupancy_grid::inflate_map with MAP_MARGIN) */
        map_.info.resolution = resolution;
        map_.info.width = width;
        map_.info.height = height;
        map_.info.origin.position.x = origin_x;
        map_.info.origin.position.y = origin_y;
        auto buf = grid_data.request();
        const int8_t* ptr = static_cast<const int8_t*>(buf.ptr);
        map_.data.assign(ptr, ptr + (size_t)width*height);
        occupancy_grid::inflate_map(map_, MAP_MARGIN);

        track_ = new Track(waypoint_file, map_, true);

        // constructor tail of LMPC::LMPC, verbatim (first odom sample)
        s_prev_ = track_->findTheta(x0, y0, 0, true);
        car_pos_ = Vector3d(x0, y0, 0.0);
        yaw_ = yaw0;
        vel_ = 0.0;
        yawdot_ = 0;
        slip_angle_ = 0;
        s_curr_ = s_prev_;

        iter_ = 2;
        use_dyn_ = false;
        first_run_ = true;   // uninitialized in original (UB); intended value
        time_ = 0;           // uninitialized in original (UB); intended value
        last_solve_ok_ = true;
        init_SS_from_data(init_data_file);
    }

    ~LMPCCore(){ delete track_; }

    /* body of odom_callback, verbatim (message unpacking replaced by args;
     * vel_/slip_angle_ computed from vx,vy exactly as from twist.linear) */
    void set_state(double x, double y, double yaw, double vx, double vy, double yawdot){
        s_curr_ = track_->findTheta(x, y, 0, true);
        car_pos_ = Vector3d(x, y, 0.0);
        yaw_ = yaw;
        vel_ = sqrt(pow(vx,2) + pow(vy,2));
        yawdot_ = yawdot;
        slip_angle_ = atan2(vy, vx);

        /** STATE MACHINE: check if dynamic model should be used based on current speed **/
        if ((!use_dyn_) && (vel_ > VEL_THRESHOLD)){
            use_dyn_ = true;
        }
        if(use_dyn_ && (vel_< VEL_THRESHOLD*0.5)){
            use_dyn_ = false;
        }
        if (vel_ > 4.5) {
            R(0,0) = 1.3 * r_accel;
            R(1,1) = 1.8 * r_steer;
        }
    }

    /* body of run(), verbatim; applyControl() replaced by returning (accel,
     * steer) with the same hardcoded steer clamp the original applies before
     * publishing */
    py::tuple step(){
        if (first_run_){
            // initialize QPSolution_ from initial Sample Safe Set (using the 2nd iteration)
            reset_QPSolution(1);
        }

        /******** LMPC MAIN LOOP starts ********/

        /***check if it is new lap***/
        if (s_curr_ - s_prev_ < -track_->length/2){
            iter_++;
            update_cost_to_go(curr_trajectory_);
            //sort(curr_trajectory_.begin(), curr_trajectory_.end(), compare_s);
            SS_.push_back(curr_trajectory_);
            curr_trajectory_.clear();
         //   reset_QPSolution(iter_-1);
            time_ = 0;
        }

        /*** select terminal state candidate and its convex safe set ***/
        Matrix<double,nx,1> terminal_candidate = select_terminal_candidate();
        /** solve MPC and record current state***/
        for (int i=0; i<1; i++){
            solve_MPC(terminal_candidate);
        }
        // applyControl() body (publish removed, clamp kept)
        float accel = QPSolution_((N+1)*nx);
        float steer = QPSolution_((N+1)*nx+1);
        steer = min(steer, 0.41f);
        steer = max(steer, -0.41f);
        u_prev_applied_ << accel, steer;   // u_{-1} anchor for the rate cost

        add_point();
        /*** store info and advance to next time step***/
        terminal_state_pred_ = QPSolution_.segment<nx>(N*nx);
        s_prev_ = s_curr_;
        time_++;
        first_run_ = false;

        return py::make_tuple(accel, steer, last_solve_ok_);
    }

    // diagnostics for the python runner (no effect on control flow)
    double s_curr() const { return s_curr_; }
    int iter() const { return iter_; }
    bool use_dyn() const { return use_dyn_; }
    double track_length() const { return track_->length; }
    double vel() const { return vel_; }
    py::array_t<double> predicted_states(){
        py::array_t<double> out({N+1, (int)nx});
        auto r = out.mutable_unchecked<2>();
        for (int i=0; i<N+1; i++)
            for (int j=0; j<nx; j++)
                r(i,j) = QPSolution_(i*nx+j);
        return out;
    }

private:
    /*Paramaters*/
    CarParams car;
    double Ts;
    int N;
    int K_NEAR;
    double SPEED_MAX;
    double STEER_MAX;
    double ACCELERATION_MAX;
    double DECELERATION_MAX;
    double MAP_MARGIN;
    double VEL_THRESHOLD;
    double WAYPOINT_SPACE;
    // MPC params
    double q_s;
    double q_s_terminal;
    double r_accel;
    double r_steer;
    Matrix<double, nu, nu> R;
    // control-rate cost c_du*||u_k - u_{k-1}||^2 (Xue et al., arXiv:2309.10716,
    // eq. of the LMPC stage cost). NOT in the original LMPC.cpp; enabled only
    // when r_d_* params are present and nonzero. u_{-1} anchors to the last
    // APPLIED input, as in their u_ic constraint.
    Matrix<double, nu, nu> R_d;
    Matrix<double, nu, 1> u_prev_applied_;
    bool rate_cost_on_;
    // optional OSQP numerical settings (0 / -1 = library default)
    int osqp_max_iter_;
    int osqp_scaling_;
    double osqp_eps_prim_inf_;
    double osqp_eps_abs_;
    double osqp_eps_rel_;

    Track* track_;
    //odometry
    Vector3d car_pos_;
    double yaw_;
    double vel_;
    double yawdot_;
    double slip_angle_;
    double s_prev_;
    double s_curr_;

    // use dynamic model or not
    bool use_dyn_;

    //Sample Safe set
    vector<vector<Sample>> SS_;
    vector<Sample> curr_trajectory_;
    int iter_;
    int time_;
    Matrix<double,nx,1> terminal_state_pred_;

    // map info
    nav_msgs::OccupancyGrid map_;

    VectorXd QPSolution_;
    bool first_run_;
    vector<geometry_msgs::Point> border_lines_;
    bool last_solve_ok_;

    /* getParameters, verbatim reads from the same-named yaml keys */
    void getParameters(const std::map<std::string,double>& p){
        N = (int)p.at("N");
        Ts = p.at("Ts");
        K_NEAR = (int)p.at("K_NEAR");
        ACCELERATION_MAX = p.at("ACCELERATION_MAX");
        DECELERATION_MAX = p.at("DECELERATION_MAX");
        SPEED_MAX = p.at("SPEED_MAX");
        STEER_MAX = p.at("STEER_MAX");
        VEL_THRESHOLD = p.at("VEL_THRESHOLD");
        WAYPOINT_SPACE = p.at("WAYPOINT_SPACE");
        r_accel = p.at("r_accel");
        r_steer = p.at("r_steer");
        q_s = p.at("q_s");
        q_s_terminal = p.at("q_s_terminal");
        R.setZero();
        R.diagonal() << r_accel, r_steer;
        // optional control-rate weights (0 / absent = term disabled)
        double rd_a = p.count("r_d_accel") ? p.at("r_d_accel") : 0.0;
        double rd_s = p.count("r_d_steer") ? p.at("r_d_steer") : 0.0;
        R_d.setZero();
        R_d.diagonal() << rd_a, rd_s;
        rate_cost_on_ = (rd_a > 0.0) || (rd_s > 0.0);
        u_prev_applied_.setZero();
        MAP_MARGIN = p.at("MAP_MARGIN");
        // optional OSQP numerical knobs; absent = library defaults (original behavior)
        osqp_max_iter_ = p.count("osqp_max_iter") ? (int)p.at("osqp_max_iter") : 0;
        osqp_scaling_ = p.count("osqp_scaling") ? (int)p.at("osqp_scaling") : -1;
        osqp_eps_prim_inf_ = p.count("osqp_eps_prim_inf") ? p.at("osqp_eps_prim_inf") : 0.0;
        osqp_eps_abs_ = p.count("osqp_eps_abs") ? p.at("osqp_eps_abs") : 0.0;
        osqp_eps_rel_ = p.count("osqp_eps_rel") ? p.at("osqp_eps_rel") : 0.0;

        car.wheelbase = p.at("wheelbase");
        car.friction_coeff = p.at("friction_coeff");
        car.h_cg = p.at("height_cg");
        car.l_r = p.at("l_cg2rear");
        car.l_f = p.at("l_cg2front");
        car.cs_f = p.at("C_S_front");
        car.cs_r = p.at("C_S_rear");
        car.I_z = p.at("moment_inertia");
        car.mass = p.at("mass");
    }

    // verbatim from LMPC.cpp
    void init_SS_from_data(const string data_file) {
        CSVReader reader(data_file);
        // Get the data from CSV File
        std::vector<std::vector<std::string>> dataList = reader.getData();
        SS_.clear();
        // Print the content of row by row on screen
        int time_prev=0;
        int it =0;
        vector<Sample> traj;
        for(std::vector<std::string> vec : dataList){
            Sample sample;
            sample.time = std::stof(vec.at(0));
            // check if it's a new lap
            if (sample.time - time_prev < 0) {
                it++;
                update_cost_to_go(traj);
                SS_.push_back(traj);
                traj.clear();
            }
            sample.x(0) = std::stof(vec.at(1));
            sample.x(1) = std::stof(vec.at(2));
            sample.x(2) = std::stof(vec.at(3));
            sample.x(3) = std::stof(vec.at(4));
            sample.x(4) = 0;
            sample.x(5) = 0;
            sample.u(0) = std::stof(vec.at(5));
            sample.u(1) = std::stof(vec.at(6));
            sample.s = std::stof(vec.at(7));
            sample.iter = it;
            traj.push_back(sample);
            time_prev = sample.time;
        }
        update_cost_to_go(traj);
        SS_.push_back(traj);
    }

    // verbatim from LMPC.cpp
    int reset_QPSolution(int iter){
        QPSolution_ = VectorXd::Zero((N+1)*nx+ N*nu + nx*(N+1) + (2*K_NEAR+1));
        for (int i=0; i<N+1; i++){
            QPSolution_.segment<nx>(i*nx) = SS_[iter][i].x;
            if (i<N) QPSolution_.segment<nu>((N+1)*nx + i*nu) = SS_[iter][i].u;
        }
        return 0;
    }

    // verbatim from LMPC.cpp
    Matrix<double,nx,1> select_terminal_candidate(){
        if (first_run_){
            return SS_.back()[N].x;
        }
        else{
            return terminal_state_pred_;
        }
    }

    // verbatim from LMPC.cpp
    void add_point(){
        Sample point;
        point.x << car_pos_.x(), car_pos_.y(), yaw_, vel_, yawdot_, slip_angle_;

        point.s = s_curr_;
        point.iter = iter_;
        point.time = time_;
        point.u = QPSolution_.segment<nu>((N+1)*nx);
        curr_trajectory_.push_back(point);
    }

    // verbatim from LMPC.cpp
    void select_convex_safe_set(vector<Sample>& convex_safe_set, int iter_start, int iter_end, double s){
        for (int it = iter_start; it<= iter_end; it++){
            int nearest_ind = find_nearest_point(SS_[it], s);
            int start_ind, end_ind;
            int lap_cost = SS_[it][0].cost;

            if (K_NEAR%2 != 0 ) {
                start_ind = nearest_ind - (K_NEAR-1)/2;
                end_ind = nearest_ind + (K_NEAR-1)/2;
            }
            else{
                start_ind = nearest_ind - K_NEAR/2 + 1;
                end_ind = nearest_ind + K_NEAR/2;
            }

            vector<Sample> curr_set;
            if (end_ind > (int)SS_[it].size()-1){ // front portion of set crossed finishing line
                for (int ind=start_ind; ind<(int)SS_[it].size(); ind++){
                    curr_set.push_back(SS_[it][ind]);
                    // modify the cost-to-go for each point before finishing line
                    // to incentivize the car to cross finishing line towards a new lap
                    curr_set[curr_set.size()-1].cost += lap_cost;
                }
                for (int ind=0; ind<end_ind-(int)SS_[it].size()+1; ind ++){
                    curr_set.push_back(SS_[it][ind]);
                }
                if ((int)curr_set.size()!=K_NEAR) throw;  // for debug
            }
            else if (start_ind < 0){  //  set crossed finishing line
                for (int ind=start_ind+(int)SS_[it].size(); ind<(int)SS_[it].size(); ind++){
                    // modify the cost-to-go, same
                    curr_set.push_back(SS_[it][ind]);
                    curr_set[curr_set.size()-1].cost += lap_cost;
                }
                for (int ind=0; ind<end_ind+1; ind ++){
                    curr_set.push_back(SS_[it][ind]);
                }
                if ((int)curr_set.size()!=K_NEAR) throw;  // for debug
            }
            else {  // no overlapping with finishing line
                for (int ind=start_ind; ind<=end_ind; ind++){
                    curr_set.push_back(SS_[it][ind]);
                }
            }
            convex_safe_set.insert(convex_safe_set.end(), curr_set.begin(), curr_set.end());
        }
    }

    // verbatim from LMPC.cpp
    int find_nearest_point(vector<Sample>& trajectory, double s){
        // binary search to find closest point to a given s
        int low = 0; int high = trajectory.size()-1;
        while (low<=high){
            int mid = (low + high)/2;
            if (s == trajectory[mid].s) return mid;
            if (s < trajectory[mid].s) high = mid-1;
            else low = mid+1;
        }
        return abs(trajectory[low].s-s) < (abs(trajectory[high].s-s))? low : high;
    }

    // verbatim from LMPC.cpp
    void update_cost_to_go(vector<Sample>& trajectory){
        trajectory[trajectory.size()-1].cost = 0;
        for (int i=trajectory.size()-2; i>=0; i--){
            trajectory[i].cost = trajectory[i+1].cost + 1;
        }
    }

    // verbatim from LMPC.cpp (unused in the control path; kept for fidelity)
    Vector3d global_to_track(double x, double y, double yaw, double s){
        double x_proj = track_->x_eval(s);
        double y_proj = track_->y_eval(s);
        double e_y = sqrt((x-x_proj)*(x-x_proj) + (y-y_proj)*(y-y_proj));
        double dx_ds = track_->x_eval_d(s);
        double dy_ds = track_->y_eval_d(s);
        e_y = dx_ds*(y-y_proj) - dy_ds*(x-x_proj) >0 ? e_y : -e_y;
        double e_yaw = yaw - atan2(dy_ds, dx_ds);
        while(e_yaw > M_PI) e_yaw -= 2*M_PI;
        while(e_yaw < -M_PI) e_yaw += 2*M_PI;

        return Vector3d(e_y, e_yaw, s);
    }

    // verbatim from LMPC.cpp (unused in the control path; kept for fidelity)
    Vector3d track_to_global(double e_y, double e_yaw, double s){
        double dx_ds = track_->x_eval_d(s);
        double dy_ds = track_->y_eval_d(s);
        Vector2d proj(track_->x_eval(s), track_->y_eval(s));
        Vector2d pos = proj + Vector2d(-dy_ds, dx_ds).normalized()*e_y;
        double yaw = e_yaw + atan2(dy_ds, dx_ds);
        return Vector3d(pos(0), pos(1), yaw);
    }

    // verbatim from LMPC.cpp
    void get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1>& hd,
            Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op, bool use_dyn){

        double yaw = x_op(2);
        double v = x_op(3);
        double accel = u_op(0);
        double steer = u_op(1);
        double yaw_dot = x_op(4);
        double slip_angle = x_op(5);

        VectorXd dynamics(6), h(6);
        Matrix<double, nx, nx> A, M12;
        Matrix<double, nx, nu> B;

        if (!use_dyn) {
            // Kinematic Model
            dynamics(0) = v * cos(yaw);
            dynamics(1) = v * sin(yaw);
            dynamics(2) = v * tan(steer)/car.wheelbase;
            dynamics(3) = accel;
            dynamics(4) = 0;
            dynamics(5) = 0;

            A <<    0.0, 0.0, -v*sin(yaw),  cos(yaw),       0.0,  0.0,
                    0.0, 0.0,  v*cos(yaw),  sin(yaw),       0.0,  0.0,
                    0.0, 0.0,         0.0,   tan(steer)/car.wheelbase,     0.0,  0.0,
                    0.0, 0.0,         0.0,       0.0,       0.0,  0.0,
                    0.0, 0.0,         0.0,       0.0,       0.0,  0.0,
                    0.0, 0.0,         0.0,       0.0,       0.0,  0.0;

            B <<    0.0, 0.0,
                    0.0, 0.0,
                    0.0, v / (cos(steer) * cos(steer) * car.wheelbase),
                    1.0, 0.0,
                    0.0, 0.0,
                    0.0, 0.0;
        }
        else{
            // Single Track Dynamic Model

            double g = 9.81;
            double rear_val = g * car.l_r - accel * car.h_cg;
            double front_val = g * car.l_f + accel * car.h_cg;

            dynamics(0) = v * cos(yaw+slip_angle);
            dynamics(1) = v * sin(yaw+slip_angle);
            dynamics(2) = yaw_dot;
            dynamics(3) = accel;
            dynamics(4) = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase)) *
                          (car.l_f * car.cs_f * steer * (rear_val) +
                           slip_angle * (car.l_r * car.cs_r * (front_val) - car.l_f * car.cs_f * (rear_val)) -
                           (yaw_dot/v) * (pow(car.l_f, 2) * car.cs_f * (rear_val) + pow(car.l_r, 2) * car.cs_r * (front_val)));        // yaw_dot dynamics
            dynamics(5) = (car.friction_coeff / (v * (car.l_r + car.l_f))) *
                          (car.cs_f * steer * rear_val - slip_angle * (car.cs_r * front_val + car.cs_f * rear_val) +
                                  (yaw_dot/v) * (car.cs_r * car.l_r * front_val - car.cs_f * car.l_f * rear_val)) - yaw_dot;        // slip_angle dynamics

            double dfyawdot_dv, dfyawdot_dyawdot, dfyawdot_dslip, dfslip_dv, dfslip_dyawdot, dfslip_dslip;
            double dfyawdot_da, dfyawdot_dsteer, dfslip_da, dfslip_dsteer;

            dfyawdot_dv = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                    * (pow(car.l_f, 2) * car.cs_f * (rear_val) + pow(car.l_r, 2) * car.cs_r * (front_val))
                    * yaw_dot / pow(v, 2);

            dfyawdot_dyawdot = -(car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                               * (pow(car.l_f, 2) * car.cs_f * (rear_val) + pow(car.l_r, 2) * car.cs_r * (front_val))/v;

            dfyawdot_dslip = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                                * (car.l_r * car.cs_r * (front_val) - car.l_f * car.cs_f * (rear_val));

            dfslip_dv = -(car.friction_coeff / (car.l_r + car.l_f)) *
                        (car.cs_f * steer * rear_val - slip_angle * (car.cs_r * front_val + car.cs_f * rear_val))/pow(v,2)
                    -2*(car.friction_coeff / (car.l_r + car.l_f)) * (car.cs_r * car.l_r * front_val - car.cs_f * car.l_f * rear_val) * yaw_dot/pow(v,3);

            dfslip_dyawdot = (car.friction_coeff / (pow(v,2) * (car.l_r + car.l_f))) * (car.cs_r * car.l_r * front_val - car.cs_f * car.l_f * rear_val) - 1;

            dfslip_dslip = -(car.friction_coeff / (v * (car.l_r + car.l_f)))*(car.cs_r * front_val + car.cs_f * rear_val);

            dfyawdot_da = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                    *(-car.l_f*car.cs_f*car.h_cg*steer + car.l_r*car.cs_r*car.h_cg*slip_angle + car.l_f*car.cs_f*car.h_cg*slip_angle
                      - (yaw_dot/v)*(-pow(car.l_f,2)*car.cs_f*car.h_cg) + pow(car.l_r,2)*car.cs_r*car.h_cg);

            dfyawdot_dsteer = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase)) *
                          (car.l_f * car.cs_f * rear_val);

            dfslip_da = (car.friction_coeff / (v * (car.l_r + car.l_f))) *
                    (-car.cs_f*car.h_cg*steer - (car.cs_r*car.h_cg - car.cs_f*car.h_cg)*slip_angle +
                    (car.cs_r*car.h_cg*car.l_r + car.cs_f*car.h_cg*car.l_f)*(yaw_dot/v));

            dfslip_dsteer = (car.friction_coeff / (v * (car.l_r + car.l_f))) *
                    (car.cs_f * rear_val);


            A <<    0.0, 0.0, -v*sin(yaw+slip_angle), cos(yaw+slip_angle),                 0.0,  -v*sin(yaw+slip_angle),
                    0.0, 0.0,  v*cos(yaw+slip_angle), sin(yaw+slip_angle),                 0.0,   v*cos(yaw+slip_angle),
                    0.0, 0.0,                       0.0,                   0.0,                 1.0,                       0.0,
                    0.0, 0.0,                       0.0,                   0.0,                 0.0,                       0.0,
                    0.0, 0.0,                       0.0,           dfyawdot_dv,     dfyawdot_dyawdot,           dfyawdot_dslip,
                    0.0, 0.0,                       0.0,             dfslip_dv,       dfslip_dyawdot,             dfslip_dslip;

            B <<    0.0, 0.0,
                    0.0, 0.0,
                    0.0, 0.0,
                    1.0, 0.0,
                    dfyawdot_da, dfyawdot_dsteer,
                    dfslip_da,     dfslip_dsteer;
        }

        /**  Discretize using Zero-Order Hold **/
        Matrix<double,nx+nx,nx+nx> aux, M;
        aux.setZero();
        aux.block<nx,nx>(0,0) << A;
        aux.block<nx,nx>(0, nx) << Matrix<double,nx,nx>::Identity();
        M = (aux*Ts).exp();
        M12 = M.block<nx,nx>(0,nx);
        h = dynamics - (A*x_op + B*u_op);

        Ad = (A*Ts).exp();
        Bd = M12*B;
        hd = M12*h;

    }

    // verbatim from LMPC.cpp (visualization block kept as border-line
    // computation into border_lines_; only the rviz publish is gone)
    void solve_MPC(const Matrix<double,nx,1>& terminal_candidate){
        vector<Sample> terminal_CSS;
        double s_t = track_->findTheta(terminal_candidate(0), terminal_candidate(1), 0, true);
        select_convex_safe_set(terminal_CSS, iter_-2, iter_-1, s_t);

        /** MPC variables: z = [x0, ..., xN, u0, ..., uN-1, s0, ..., sN, lambda0, ....., lambda(2*K_NEAR), s_t1, s_t2, .. s_t6]*
         *  constraints: dynamics, track bounds, input limits, acceleration limit, slack, lambdas, terminal state, sum of lambda's*/
        SparseMatrix<double> HessianMatrix((N+1)*nx+ N*nu + (N+1) + (2*K_NEAR) +nx, (N+1)*nx+ N*nu + (N+1)+ (2*K_NEAR) +nx);
        SparseMatrix<double> constraintMatrix((N+1)*nx+ 2*(N+1) + N*nu + (N+1) + (N+1) + (2*K_NEAR) + 2*nx+1, (N+1)*nx+ N*nu + (N+1)+ (2*K_NEAR) +nx);

        VectorXd gradient((N+1)*nx+ N*nu + (N+1) + (2*K_NEAR) +nx);

        VectorXd lower((N+1)*nx+ 2*(N+1) + N*nu + (N+1) + (N+1) + (2*K_NEAR) + 2*nx+1);
        VectorXd upper((N+1)*nx+ 2*(N+1) + N*nu + (N+1) + (N+1) + (2*K_NEAR) + 2*nx+1);

        gradient.setZero();
        lower.setZero(); upper.setZero();

        Matrix<double,nx,1> x_k_ref;
        Matrix<double,nu,1> u_k_ref;
        Matrix<double,nx,nx> Ad;
        Matrix<double,nx,nu> Bd;
        Matrix<double,nx,1> x0, hd;
        border_lines_.clear();

        if (use_dyn_)  x0 <<car_pos_.x(), car_pos_.y(), yaw_, vel_, yawdot_, slip_angle_;
        else{ x0 <<car_pos_.x(), car_pos_.y(), yaw_, vel_, 0.0, 0.0; }
        /** make sure there are no discontinuities in yaw**/
        // first check terminal safe_set
        for (int i=0; i<(int)terminal_CSS.size(); i++){
            wrap_angle(terminal_CSS[i].x(2), x0(2));
        }
        // also check for previous QPSolution
        for (int i=0; i<N+1; i++){
            wrap_angle(QPSolution_(i*nx+2), x0(2));
        }

        for (int i=0; i<N+1; i++){        //0 to N

            x_k_ref = QPSolution_.segment<nx>(i*nx);
            u_k_ref = QPSolution_.segment<nu>((N+1)*nx + i*nu);
            double s_ref = track_->findTheta(x_k_ref(0), x_k_ref(1), 0, true);
            get_linearized_dynamics(Ad, Bd, hd, x_k_ref, u_k_ref, use_dyn_);
            /* form Hessian entries*/
            // cost does not depend on x0, only 1 to N
            if (i>0) {
                HessianMatrix.insert((N+1)*nx + N*nu + i, (N+1)*nx + N*nu + i) = q_s;
            }
            if (i<N){
                for (int row=0; row<nu; row++){
                    // diagonal: input-effort weight R plus, when the control-rate
                    // cost is enabled, the tridiagonal contribution of
                    // 0.5*R_d*sum_k ||u_k - u_{k-1}||^2 with u_{-1} = last applied
                    // input (2*R_d for interior stages, R_d for the last stage)
                    double rate_diag = rate_cost_on_ ? ((i < N-1) ? 2.0 : 1.0) * R_d(row, row) : 0.0;
                    HessianMatrix.insert((N+1)*nx + i*nu + row, (N+1)*nx + i*nu + row) = R(row, row) + rate_diag;
                }
                if (rate_cost_on_){
                    if (i == 0){
                        // linear term from (u_0 - u_applied)^2
                        for (int row=0; row<nu; row++){
                            gradient((N+1)*nx + row) += -R_d(row, row) * u_prev_applied_(row);
                        }
                    }
                    else {
                        // symmetric off-diagonal coupling u_i <-> u_{i-1}
                        for (int row=0; row<nu; row++){
                            HessianMatrix.insert((N+1)*nx + i*nu + row, (N+1)*nx + (i-1)*nu + row) = -R_d(row, row);
                            HessianMatrix.insert((N+1)*nx + (i-1)*nu + row, (N+1)*nx + i*nu + row) = -R_d(row, row);
                        }
                    }
                }
            }

            /* form constraint matrix */
            if (i<N){
                // Ad
                for (int row=0; row<nx; row++){
                    for(int col=0; col<nx; col++){
                        constraintMatrix.insert((i+1)*nx+row, i*nx+col) = Ad(row,col);
                    }
                }
                // Bd
                for (int row=0; row<nx; row++){
                    for(int col=0; col<nu; col++){
                        constraintMatrix.insert((i+1)*nx+row, (N+1)*nx+ i*nu+col) = Bd(row,col);
                    }
                }
                lower.segment<nx>((i+1)*nx) = -hd;//-OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY,-OsqpEigen::INFTY;//-hd;
                upper.segment<nx>((i+1)*nx) = -hd; //OsqpEigen::INFTY, OsqpEigen::INFTY,OsqpEigen::INFTY,OsqpEigen::INFTY;//-hd;
            }

            // -I for each x_k+1
            for (int row=0; row<nx; row++) {
                constraintMatrix.insert(i*nx+row, i*nx+row) = -1.0;
            }

            double dx_dtheta = track_->x_eval_d(s_ref);
            double dy_dtheta = track_->y_eval_d(s_ref);

            constraintMatrix.insert((N+1)*nx+ 2*i, i*nx) = -dy_dtheta;      // a*x
            constraintMatrix.insert((N+1)*nx+ 2*i, i*nx+1) = dx_dtheta;     // b*y
            constraintMatrix.insert((N+1)*nx+ 2*i, (N+1)*nx +N*nu +i) = 1.0;   // min(C1,C2) <= a*x + b*y + s_k <= inf

            constraintMatrix.insert((N+1)*nx+ 2*i+1, i*nx) = -dy_dtheta;      // a*x
            constraintMatrix.insert((N+1)*nx+ 2*i+1, i*nx+1) = dx_dtheta;     // b*y
            constraintMatrix.insert((N+1)*nx+ 2*i+1, (N+1)*nx +N*nu +i) = -1.0;   // -inf <= a*x + b*y - s_k <= max(C1,C2)

            //get upper line and lower line
            Vector2d left_tangent_p, right_tangent_p, center_p;
            Vector2d right_line_p1, right_line_p2, left_line_p1, left_line_p2;
            geometry_msgs::Point r_p1, r_p2, l_p1, l_p2;

            center_p << track_->x_eval(s_ref), track_->y_eval(s_ref);
            right_tangent_p = center_p + track_->getRightHalfWidth(s_ref) * Vector2d(dy_dtheta, -dx_dtheta).normalized();
            left_tangent_p  = center_p + track_->getLeftHalfWidth(s_ref) * Vector2d(-dy_dtheta, dx_dtheta).normalized();

            right_line_p1 = right_tangent_p + 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
            right_line_p2 = right_tangent_p - 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
            left_line_p1 = left_tangent_p + 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
            left_line_p2 = left_tangent_p - 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();

            // For visualizing track boundaries
            r_p1.x = right_line_p1(0);  r_p1.y = right_line_p1(1);
            r_p2.x = right_line_p2(0);  r_p2.y = right_line_p2(1);
            l_p1.x = left_line_p1(0);   l_p1.y = left_line_p1(1);
            l_p2.x = left_line_p2(0);   l_p2.y = left_line_p2(1);
            border_lines_.push_back(r_p1);  border_lines_.push_back(r_p2);
            border_lines_.push_back(l_p1); border_lines_.push_back(l_p2);

            double C1 =  - dy_dtheta*right_tangent_p(0) + dx_dtheta*right_tangent_p(1);
            double C2 = - dy_dtheta*left_tangent_p(0) + dx_dtheta*left_tangent_p(1);

            lower((N+1)*nx+ 2*i) =  min(C1, C2);
            upper((N+1)*nx+ 2*i) = OsqpEigen::INFTY;

            lower((N+1)*nx+ 2*i+1) = -OsqpEigen::INFTY;
            upper((N+1)*nx+ 2*i+1) = max(C1, C2);

            // u_min < u < u_max
            if (i<N){
                for (int row=0; row<nu; row++){
                    constraintMatrix.insert((N+1)*nx+ 2*(N+1) +i*nu+row, (N+1)*nx+i*nu+row) = 1.0;
                }
                // input bounds: speed and steer
                lower.segment<nu>((N+1)*nx+ 2*(N+1) +i*nu) <<  -DECELERATION_MAX, -STEER_MAX;
                upper.segment<nu>((N+1)*nx+ 2*(N+1) +i*nu) << ACCELERATION_MAX, STEER_MAX;
            }

            //max velocity
            constraintMatrix.insert((N+1)*nx+ 2*(N+1) + N*nu +i, i*nx+3) = 1;
            lower((N+1)*nx+ 2*(N+1) + N*nu +i) = 0;
            upper((N+1)*nx+ 2*(N+1) + N*nu +i) = SPEED_MAX;

            // s_k >= 0
            constraintMatrix.insert((N+1)*nx + 2*(N+1) + N*nu + (N+1) + i, (N+1)*nx+N*nu +i) = 1.0;
            lower((N+1)*nx + 2*(N+1) + N*nu  + (N+1) + i) = 0;
            upper((N+1)*nx + 2*(N+1) + N*nu  + (N+1) + i) = OsqpEigen::INFTY;
        }
        int numOfConstraintsSoFar = (N+1)*nx + 2*(N+1) + N*nu + (N+1) + (N+1);

        // lamda's >= 0
        for (int i=0; i<2*K_NEAR; i++){
            constraintMatrix.insert(numOfConstraintsSoFar + i, (N+1)*nx+ N*nu + (N+1) + i) = 1.0;
            lower(numOfConstraintsSoFar + i) = 0;
            upper(numOfConstraintsSoFar + i) = OsqpEigen::INFTY;
        }
        numOfConstraintsSoFar += 2*K_NEAR;

        // terminal state constraints:  -s_t <= -x_N+1 + linear_combination(lambda's) <= s_t
        // 0 <= s_t -x_N+1 + linear_combination(lambda's) <= inf
        for (int i=0; i<2*K_NEAR; i++){
            for (int state_ind=0; state_ind<nx; state_ind++){
                constraintMatrix.insert(numOfConstraintsSoFar + state_ind, (N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].x(state_ind);
            }
        }
        for (int state_ind=0; state_ind<nx; state_ind++){
            constraintMatrix.insert(numOfConstraintsSoFar + state_ind, N*nx + state_ind) = -1;
            constraintMatrix.insert(numOfConstraintsSoFar+state_ind, (N+1)*nx+ N*nu + (N+1) + 2*K_NEAR + state_ind) = 1;
            lower(numOfConstraintsSoFar+state_ind) = 0;
            upper(numOfConstraintsSoFar+state_ind) = OsqpEigen::INFTY;
        }
        numOfConstraintsSoFar += nx;

        //-inf <= -x_N+1 + linear_combination(lambda's) - s_t <= 0
        for (int i=0; i<2*K_NEAR; i++){
            for (int state_ind=0; state_ind<nx; state_ind++){
                constraintMatrix.insert(numOfConstraintsSoFar + state_ind, (N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].x(state_ind);
            }
        }
        for (int state_ind=0; state_ind<nx; state_ind++){
            constraintMatrix.insert(numOfConstraintsSoFar + state_ind, N*nx + state_ind) = -1;
            constraintMatrix.insert(numOfConstraintsSoFar+state_ind, (N+1)*nx+ N*nu + (N+1) + 2*K_NEAR + state_ind) = -1;
            lower(numOfConstraintsSoFar+state_ind) = -OsqpEigen::INFTY;
            upper(numOfConstraintsSoFar+state_ind) = 0;
        }

        numOfConstraintsSoFar += nx;
        // sum of lamda's = 1;
        for (int i=0; i<2*K_NEAR; i++){
            constraintMatrix.insert(numOfConstraintsSoFar, (N+1)*nx+ N*nu + (N+1) + i) = 1;
        }

        lower(numOfConstraintsSoFar) = 1.0;
        upper(numOfConstraintsSoFar) = 1.0;
        numOfConstraintsSoFar++;
        if (numOfConstraintsSoFar != (N+1)*nx+ 2*(N+1) + N*nu + (N+1) + (N+1) + (2*K_NEAR) + 2*nx+1) throw;  // for debug

        // gradient
        int lowest_cost1 = terminal_CSS[K_NEAR-1].cost;
        int lowest_cost2 = terminal_CSS[2*K_NEAR-1].cost;
        for (int i=0; i<K_NEAR; i++){
            gradient((N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].cost-lowest_cost1;
        }
        for (int i=K_NEAR; i<2*K_NEAR; i++){
            gradient((N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].cost-lowest_cost2;
        }

        //
        for (int i=0; i<nx; i++){
            HessianMatrix.insert((N+1)*nx+ N*nu + (N+1) + 2*K_NEAR + i, (N+1)*nx+ N*nu + (N+1) + 2*K_NEAR + i) = q_s_terminal;
        }

        //x0 constraint
        lower.head(nx) = -x0;
        upper.head(nx) = -x0;


        SparseMatrix<double> H_t = HessianMatrix.transpose();
        SparseMatrix<double> sparse_I((N+1)*nx+ N*nu + (N+1)+ (2*K_NEAR) +nx, (N+1)*nx+ N*nu + (N+1)+ (2*K_NEAR) +nx);
        sparse_I.setIdentity();
        HessianMatrix = 0.5*(HessianMatrix + H_t) + 0.0000001*sparse_I;

        OsqpEigen::Solver solver;
        solver.settings()->setWarmStart(true);
        solver.settings()->setVerbosity(false);   // added: original used the default (verbose); output-only change
        // optional numerical knobs (absent from yaml = OSQP defaults = original behavior)
        if (osqp_max_iter_ > 0) solver.settings()->setMaxIteration(osqp_max_iter_);
        if (osqp_scaling_ >= 0) solver.settings()->setScaling(osqp_scaling_);
        if (osqp_eps_prim_inf_ > 0) solver.settings()->setPrimalInfeasibilityTolerance(osqp_eps_prim_inf_);
        if (osqp_eps_abs_ > 0) solver.settings()->setAbsoluteTolerance(osqp_eps_abs_);
        if (osqp_eps_rel_ > 0) solver.settings()->setRelativeTolerance(osqp_eps_rel_);
        solver.data()->setNumberOfVariables((N+1)*nx+ N*nu + (N+1)+ 2*K_NEAR +nx);
        solver.data()->setNumberOfConstraints((N+1)*nx+ 2*(N+1) + N*nu + (N+1) + (N+1) + 2*K_NEAR + 2*nx+1);

        if (!solver.data()->setHessianMatrix(HessianMatrix)) throw "fail set Hessian";
        if (!solver.data()->setGradient(gradient)){throw "fail to set gradient";}
        if (!solver.data()->setLinearConstraintsMatrix(constraintMatrix)) throw"fail to set constraint matrix";
        if (!solver.data()->setLowerBound(lower)){throw "fail to set lower bound";}
        if (!solver.data()->setUpperBound(upper)){throw "fail to set upper bound";}

        bool init_ok = solver.initSolver();
        if (!init_ok){ cout<< "fail to initialize solver"<<endl;}

        if(!init_ok || !solver.solve()) {
            // ---- debug dump (diagnostics only; controller behavior unchanged:
            // on failure the previous QPSolution_ keeps being used, as in the
            // original code) ----
            auto finite_count = [](const VectorXd& v){
                int bad = 0;
                for (int i=0; i<v.size(); i++) if (!std::isfinite(v(i))) bad++;
                return bad;
            };
            cout << "[LMPC DEBUG] QP failure (init_ok=" << init_ok << ")"
                 << " nonfinite: grad=" << finite_count(gradient)
                 << " lower=" << finite_count(VectorXd(lower.array().min(1e30).max(-1e30) - lower.array()*0))
                 << endl;
            int bad_lo = 0, bad_hi = 0;
            for (int i=0; i<lower.size(); i++){
                if (std::isnan(lower(i))) bad_lo++;
                if (std::isnan(upper(i))) bad_hi++;
            }
            int bad_H = 0;
            for (int k=0; k<HessianMatrix.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(HessianMatrix,k); it; ++it)
                    if (!std::isfinite(it.value())) bad_H++;
            int bad_A = 0;
            for (int k=0; k<constraintMatrix.outerSize(); ++k)
                for (SparseMatrix<double>::InnerIterator it(constraintMatrix,k); it; ++it)
                    if (!std::isfinite(it.value())) bad_A++;
            cout << "[LMPC DEBUG] NaN lower=" << bad_lo << " NaN upper=" << bad_hi
                 << " nonfinite H=" << bad_H << " nonfinite A=" << bad_A << endl;
            cout << "[LMPC DEBUG] x0 = " << x0.transpose() << " use_dyn=" << use_dyn_ << endl;
            for (int i=0; i<N+1; i++){
                Matrix<double,nx,1> xr = QPSolution_.segment<nx>(i*nx);
                double s_ref = track_->findTheta(xr(0), xr(1), 0, true);
                double lw = track_->getLeftHalfWidth(s_ref);
                double rw = track_->getRightHalfWidth(s_ref);
                double dxd = track_->x_eval_d(s_ref);
                double dyd = track_->y_eval_d(s_ref);
                bool row_bad = !std::isfinite(lower((N+1)*nx + 2*i)) || !std::isfinite(upper((N+1)*nx + 2*i+1))
                               || std::isnan(lower((N+1)*nx + 2*i)) || std::isnan(upper((N+1)*nx + 2*i+1));
                if (i < 3 || row_bad || lw > 5.0 || rw > 5.0 || !std::isfinite(dxd) || !std::isfinite(dyd)){
                    cout << "[LMPC DEBUG] k=" << i
                         << " xref=(" << xr(0) << "," << xr(1) << ") v=" << xr(3)
                         << " s_ref=" << s_ref << " lw=" << lw << " rw=" << rw
                         << " dx'=" << dxd << " dy'=" << dyd
                         << " boundlo=" << lower((N+1)*nx + 2*i)
                         << " boundhi=" << upper((N+1)*nx + 2*i+1) << endl;
                }
            }
            last_solve_ok_ = false;
            return;
        }
        last_solve_ok_ = true;
        QPSolution_ = solver.getSolution();

        solver.clearSolver();
    }
};

PYBIND11_MODULE(lmpc_core, m){
    m.doc() = "ROS-free LearningMPC core (verbatim controller logic from src/LMPC.cpp)";
    py::class_<LMPCCore>(m, "LMPCCore")
        .def(py::init<const std::map<std::string,double>&, py::array_t<int8_t>,
                      uint32_t, uint32_t, double, double, double,
                      const std::string&, const std::string&,
                      double, double, double>(),
             py::arg("params"), py::arg("grid_data"),
             py::arg("width"), py::arg("height"),
             py::arg("resolution"), py::arg("origin_x"), py::arg("origin_y"),
             py::arg("waypoint_file"), py::arg("init_data_file"),
             py::arg("x0"), py::arg("y0"), py::arg("yaw0"))
        .def("set_state", &LMPCCore::set_state,
             py::arg("x"), py::arg("y"), py::arg("yaw"),
             py::arg("vx"), py::arg("vy"), py::arg("yawdot"))
        .def("step", &LMPCCore::step)
        .def("s_curr", &LMPCCore::s_curr)
        .def("iter", &LMPCCore::iter)
        .def("use_dyn", &LMPCCore::use_dyn)
        .def("vel", &LMPCCore::vel)
        .def("track_length", &LMPCCore::track_length)
        .def("predicted_states", &LMPCCore::predicted_states);
}
