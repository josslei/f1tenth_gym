#include "safe_set.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "dynamics/common.hpp"

namespace lmpc
{

namespace
{

// Parses the header written by
// scripts/lmpc_collect_seed_lap.py::write_seed_lap_csv:
// vx,vy,omega,epsi,s,ey,t,a,delta,J -- only x = [vx,vy,omega,epsi,s,ey] and
// J are needed for the safe-set query, so a/delta/t (which are blank on the
// CSV's last row anyway -- there is no successor state for it) are read and
// discarded.
std::vector<SafeSetSample> load_lap(const std::string & csv_path)
{
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("SafeSet: could not open seed-lap CSV: " + csv_path);
  }

  std::string header;
  if (!std::getline(file, header)) {
    throw std::runtime_error("SafeSet: empty seed-lap CSV: " + csv_path);
  }

  std::vector<SafeSetSample> samples;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(iss, field, ',')) {
      fields.push_back(field);
    }
    // vx,vy,omega,epsi,s,ey,t,a,delta,J -- 10 columns.
    if (fields.size() != 10) {
      throw std::runtime_error("SafeSet: malformed row in " + csv_path);
    }

    casadi::DM x = casadi::DM::zeros(dynamics::kStateDim, 1);
    x(dynamics::VX) = std::stod(fields[0]);
    x(dynamics::VY) = std::stod(fields[1]);
    x(dynamics::OMEGA) = std::stod(fields[2]);
    x(dynamics::EPSI) = std::stod(fields[3]);
    x(dynamics::S) = std::stod(fields[4]);
    x(dynamics::EY) = std::stod(fields[5]);
    const double J = std::stod(fields[9]);

    samples.push_back(SafeSetSample{x, J});
  }

  if (samples.empty()) {
    throw std::runtime_error("SafeSet: no data rows in " + csv_path);
  }
  return samples;
}

// Squared distance under D = diag(0,0,0,0,1,1) -- DESIGN.md SS2's pinned
// metric, nonzero only at IDX_S, IDX_EY.
double weighted_distance_sq(const casadi::DM & a, const casadi::DM & b)
{
  const double ds = static_cast<double>(a(dynamics::S)) - static_cast<double>(b(dynamics::S));
  const double dey = static_cast<double>(a(dynamics::EY)) - static_cast<double>(b(dynamics::EY));
  return ds * ds + dey * dey;
}

}  // namespace

SafeSet::SafeSet(const std::string & seed_lap_csv_path)
{
  laps.push_back(load_lap(seed_lap_csv_path));
}

void SafeSet::add_lap(const std::string & lap_csv_path)
{
  laps.push_back(load_lap(lap_csv_path));
}

SafeSet::QueryResult SafeSet::query(const casadi::DM & x_query, casadi_int K) const
{
  std::vector<casadi::DM> x_cols;
  std::vector<double> j_vals;

  for (const std::vector<SafeSetSample> & lap : laps) {
    const casadi_int k = std::min<casadi_int>(K, static_cast<casadi_int>(lap.size()));

    std::vector<std::size_t> idx(lap.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
      idx.begin(), idx.begin() + k, idx.end(),
      [&](std::size_t i, std::size_t j) {
        return weighted_distance_sq(lap[i].x, x_query) < weighted_distance_sq(lap[j].x, x_query);
      });

    for (casadi_int n = 0; n < k; ++n) {
      x_cols.push_back(lap[idx[n]].x);
      j_vals.push_back(lap[idx[n]].J);
    }
  }

  QueryResult result;
  result.X_ss = casadi::DM::horzcat(x_cols);
  result.J_ss = casadi::DM(j_vals);
  return result;
}

}  // namespace lmpc
