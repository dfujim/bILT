
#include <cmath>

#include "bnmr_fit_functions.h"

namespace triumf {
namespace bnmr {

// Exponential relaxation function, convoluted with square beam pulse
double pulsed_exponential(double *x, double *par) {
  double time = x[0];
  double nuclear_lifetime = par[0];
  double beam_pulse = par[1];
  double initial_asymmetry = par[2];
  double relaxation_rate = par[3];

  double lambda = relaxation_rate + 1.0 / nuclear_lifetime;

  // Before the beam pulse
  if (time == 0.0) {
    return initial_asymmetry;
  }
  // Durin the beam pulse
  if (time > 0.0 and time < beam_pulse) {
    double n = nuclear_lifetime * (1.0 - std::exp(-time / nuclear_lifetime));
    double dynamic_relaxation = (1.0 - std::exp(-lambda * time)) / lambda;
    return initial_asymmetry * dynamic_relaxation / n;
  }
  // After the beam pulse
  if (time >= 0.0 and time >= beam_pulse) {
    double n =
        nuclear_lifetime * (1.0 - std::exp(-beam_pulse / nuclear_lifetime));
    double dynamic_relaxation = (1.0 - std::exp(-lambda * beam_pulse)) / lambda;
    double relaxation = std::exp(-relaxation_rate * (time - beam_pulse));
    return initial_asymmetry * dynamic_relaxation * relaxation / n;
  }
  return 0.0;
}

// Biexponential relaxation function, convoluted with square beam pulse
double pulsed_bi_exponential(double *x, double *par) {
  double time = x[0];
  double nuclear_lifetime = par[0];
  double beam_pulse = par[1];
  double initial_asymmetry = par[2];
  double slow_relaxing_fraction = par[3];
  double slow_relaxation_rate = par[4];
  double fast_relaxation_rate = par[5];

  double lambda_slow = slow_relaxation_rate + 1.0 / nuclear_lifetime;
  double lambda_fast = fast_relaxation_rate + 1.0 / nuclear_lifetime;

  // Before the beam pulse
  if (time == 0.0) {
    return initial_asymmetry;
  }
  // Durin the beam pulse
  if (time > 0.0 and time < beam_pulse) {
    double n = nuclear_lifetime * (1.0 - std::exp(-time / nuclear_lifetime));
    double f_slow = slow_relaxing_fraction *
                    (1.0 - std::exp(-lambda_slow * time)) / lambda_slow;
    double f_fast = (1.0 - slow_relaxing_fraction) *
                    (1.0 - std::exp(-lambda_fast * time)) / lambda_fast;
    return initial_asymmetry * (f_slow + f_fast) / n;
  }
  // After the beam pulse
  if (time >= 0.0 and time >= beam_pulse) {
    double n =
        nuclear_lifetime * (1.0 - std::exp(-beam_pulse / nuclear_lifetime));
    double f_slow = slow_relaxing_fraction *
                    (1.0 - std::exp(-lambda_slow * beam_pulse)) / lambda_slow;
    double f_fast = (1.0 - slow_relaxing_fraction) *
                    (1.0 - std::exp(-lambda_fast * beam_pulse)) / lambda_fast;
    double r_slow = std::exp(-slow_relaxation_rate * (time - beam_pulse));
    double r_fast = std::exp(-fast_relaxation_rate * (time - beam_pulse));
    return initial_asymmetry * (f_slow * r_slow + f_fast * r_fast) / n;
  }
  return 0.0;
}

} // namespace bnmr
} // namespace triumf
