#ifndef LMPC__TRACK_HPP_
#define LMPC__TRACK_HPP_

#include <string>
#include <vector>

namespace lmpc {

// The one piece of track geometry the FHOCP needs at every horizon stage:
// curvature as a function of arclength s (the kappa argument threaded
// through dynamics::frenet_pose_kinematics()). Loads the same raw
// centerline CSV (x_m, y_m, w_tr_right_m, w_tr_left_m) that
// scripts/lmpc_collect_seed_lap.py projects vehicle poses onto to produce
// D^0's own s/ey/epsi columns -- s here must be the identical coordinate,
// or A_t/B_t/C_t would be linearized against a curvature that doesn't match
// where the state actually says the car is.
//
// Matches that script's conventions exactly:
//  - s: open (non-periodic) cumulative arc length over consecutive points,
//    s[0] = 0 (utils/waypoint_utils.cumulative_arc_lengths). NOT wrapped
//    modulo track length -- the seed lap's own s never wraps either, since
//    it's recorded over a single lap.
//  - heading: forward-difference tangent between consecutive points, with
//    the last point's heading measured back to the first (the CSV already
//    describes one closed loop, so this is the true closing tangent, not a
//    wraparound artifact) -- same as load_centerline_waypoints()'s use of
//    np.roll(xy, -1).
// Curvature is then the turn rate of that heading per unit arclength,
// finite-differenced at each sample and linearly interpolated between
// samples for an arbitrary query s.
class Track {
public:
  explicit Track(const std::string &centerline_csv_path);

  // Total arclength spanned by the loaded centerline (open path, one lap).
  double length() const { return s_.back(); }

  // Curvature at arclength s (positive = left turn), s clamped to
  // [0, length()] -- see class comment for why this is not periodic.
  double curvature(double s) const;

private:
  std::vector<double> s_; // cumulative arclength per sample, size N, s_[0]=0
  std::vector<double> kappa_; // curvature per sample, size N
};

} // namespace lmpc

#endif // LMPC__TRACK_HPP_
