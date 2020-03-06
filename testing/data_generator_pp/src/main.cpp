#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <TApplication.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TH1D.h>
#include <THStack.h>
#include <TStyle.h>
#include <TVectorD.h>

#include <omp.h>

#include <yaml-cpp/yaml.h>

#include "beta_8Li.hpp"
#include "bnmr_fit_functions.h"
#include "pcg_random.hpp"

// similar to numpy.linspace in python
template <typename T> std::vector<T> linspace(T a, T b, std::size_t N) {
  T h = (b - a) / static_cast<T>(N - 1);
  std::vector<T> xs(N);
  typename std::vector<T>::iterator x;
  T val;
  for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
    *x = val;
  }
  return xs;
}

// calculate pi
template <typename T> T const pi = std::acos(-T(1));

// angular beta-emission function
double W(double theta, double A = 1.0, double P = 1.0, double v_over_c = 1.0) {
  return 1.0 + v_over_c * A * P * std::cos(theta);
}

// fractional velocity
double v_over_c(double E_k_keV, double rest_mass_MeV = 0.51099895000) {
  double E_k_MeV = E_k_keV * 1e-3;
  return std::sqrt(1.0 - 1.0 / std::pow(E_k_MeV / rest_mass_MeV + 1.0, 2));
}

// main routine
int main(int argc, char **argv) {

  // check input arguments
  if (argc != 2) {
    std::cout << "YAML input file required!\n";
    return EXIT_FAILURE;
  }

  // open the yaml control file
  YAML::Node config = YAML::LoadFile(argv[1]);

  // print the simulation parameters
  std::cout << "\n";
  std::cout << "Running new Monte Carlo simulation of β-NMR data!\n\n";
  std::cout << "Inputs:\n";
  std::cout << " - n_probes: " << config["n"].as<std::string>() << "\n";
  std::cout << " - Lifetime (s): " << config["lifetime (s)"].as<std::string>()
            << "\n";
  std::cout << " - Beam Pulse (s): "
            << config["beam pulse (s)"].as<std::string>() << "\n";
  std::cout << " - A_beta: " << config["A_beta"].as<std::string>() << "\n";
  std::cout << " - Polarization function f(x): \""
            << config["polarization function f(x)"].as<std::string>() << "\"\n";
  std::cout << "Saving the generated histograms to "
            << config["output"].as<std::string>() << " with:\n";
  std::cout << " - n_bins: " << config["histogram n_bins"].as<std::string>()
            << "\n";
  std::cout << " - t_min: " << config["histogram t_min"].as<std::string>()
            << "\n";
  std::cout << " - t_max: " << config["histogram t_max"].as<std::string>()
            << "\n";
  std::cout << "\n";
  std::cout << "Writing output to \"" << config["output"].as<std::string>()
            << "\"\n";

  // std::random_device rd;
  // std::mt19937_64 prng(rd());
  // std::mt19937_64 prng;
  // prng.seed(1234567890);

  // Seed with a real random value, if available
  pcg_extras::seed_seq_from<std::random_device> seed_source;

  // Make a random number engine
  pcg64 prng(seed_source);

  const double tau_8Li = config["lifetime (s)"].as<double>();
  const double beam_pulse = config["beam pulse (s)"].as<double>();

  std::uniform_real_distribution<double> d_arrive(0.0, beam_pulse);
  std::exponential_distribution<double> d_decay(1.0 / tau_8Li);
  std::piecewise_linear_distribution<double> d_energy(E.begin(), E.end(),
                                                      dNdE.begin());

  // larger number of sampling points makes simulation very slow...
  std::vector<double> theta_list = linspace<double>(0.0, 2.0 * pi<double>, 100);

  // histograms
  const int n_bins = config["histogram n_bins"].as<double>();
  const double t_min = config["histogram t_min"].as<double>();
  const double t_max = config["histogram t_max"].as<double>();

  //
  TH1D hF_p("hF_p", "F^{+} detector;Time (s);Counts", n_bins, t_min, t_max);
  TH1D hB_p("hB_p", "B^{+} detector;Time (s);Counts", n_bins, t_min, t_max);
  TH1D hF_m("hF_m", "F^{-} detector;Time (s);Counts", n_bins, t_min, t_max);
  TH1D hB_m("hB_m", "B^{-} detector;Time (s);Counts", n_bins, t_min, t_max);

  // total number of decays to simulate
  const unsigned int n_decays = config["n"].as<double>();
  const double A_8Li = config["A_beta"].as<double>();

  // relaxation function
  TF1 rlx =
      TF1("rlx", config["polarization function f(x)"].as<std::string>().c_str(),
          0, t_max * 10);

  // time the simulation
  auto start = std::chrono::high_resolution_clock::now();

// monte carlo! - divide n_decays to split events over both helicities
#pragma omp parallel for
  for (unsigned int i = 0; i < n_decays / 2; ++i) {
    // arrival time
    double t_arrive_p = d_arrive(prng);
    double t_arrive_m = d_arrive(prng);
    // decay time
    double t_decay_p = d_decay(prng);
    double t_decay_m = d_decay(prng);
    // dection time
    double t_detect_p = t_arrive_p + t_decay_p;
    double t_detect_m = t_arrive_m + t_decay_m;

    // polarization at the time of decay
    double p_decay_p = rlx(t_decay_p);
    double p_decay_m = -rlx(t_decay_m);

    // create the theta distribution using
    std::vector<double> weight_p;
    std::vector<double> weight_m;

    // reserve memory for vectors
    weight_p.reserve(theta_list.size());
    weight_m.reserve(theta_list.size());

    // what is the energy of the emitted beta?
    double e_p = d_energy(prng);
    double e_m = d_energy(prng);

    // calculate W(theta) for both helicities
    for (auto &tl : theta_list) {
      // positive helicity
      double w_p = W(tl, A_8Li, p_decay_p, v_over_c(e_p));
      weight_p.push_back(w_p);
      // negative helicity
      double w_m = W(tl, A_8Li, p_decay_m, v_over_c(e_m));
      weight_m.push_back(w_m);
    }

    // create the distributions
    std::piecewise_linear_distribution<double> d_angle_p(
        theta_list.begin(), theta_list.end(), weight_p.begin());
    std::piecewise_linear_distribution<double> d_angle_m(
        theta_list.begin(), theta_list.end(), weight_m.begin());

    // calculate the thetas using the distributions
    double theta_p = d_angle_p(prng);
    double theta_m = d_angle_m(prng);
    // std::cout << theta << "\n";

    // which detector detects the beta?

    // positive helicity
    if (0.0 <= theta_p and theta_p < 0.5 * pi<double>) {
      hF_p.Fill(t_detect_p);
    }
    if (0.5 * pi<double> < theta_p and theta_p < 1.5 * pi<double>) {
      hB_p.Fill(t_detect_p);
    }
    if (1.5 * pi<double> < theta_p and theta_p <= 2.0 * pi<double>) {
      hF_p.Fill(t_detect_p);
    }

    // negative helicity
    if (0.0 <= theta_m and theta_m < 0.5 * pi<double>) {
      hF_m.Fill(t_detect_m);
    }
    if (0.5 * pi<double> < theta_m and theta_m < 1.5 * pi<double>) {
      hB_m.Fill(t_detect_m);
    }
    if (1.5 * pi<double> < theta_m and theta_m <= 2.0 * pi<double>) {
      hF_m.Fill(t_detect_m);
    }
  }

  // time the simulation
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "\nDone!\n\n";
  std::chrono::duration<double> diff = end - start;
  std::cout << "Time to simulate " << n_decays << " probes: " << diff.count()
            << " s\n\n";

  /*
  double alpha = N_F->GetEffectiveEntries() / N_B->GetEffectiveEntries();
  double alpha_error =
      std::abs(alpha) * std::sqrt(1.0 / N_F->GetEffectiveEntries() +
                                  1.0 / N_B->GetEffectiveEntries());
  std::cout << "\n";
  std::cout << "alpha = " << alpha << " +/- " << alpha_error;
  std::cout << "\n";
  */

  // create a canvas to suppress warning of automatic creation
  TCanvas c1("c1", "c1");

  //
  THStack hsN("hsN", "All detectors;Time (s); Counts");
  hsN.Add(&hF_p);
  hsN.Add(&hB_p);
  hsN.Add(&hF_m);
  hsN.Add(&hB_m);
  // hs.Draw("nostack plc");

  // TCanvas *c2 = new TCanvas("c2", "c2");
  // auto hA = N_F->GetAsymmetry(N_B, alpha, alpha_error);
  auto hA_p = hF_p.GetAsymmetry(&hB_p);
  hA_p->SetName("hA_p");
  hA_p->SetTitle("Positive Asymmetry;Time (s);Asymmetry");
  hA_p->SetMarkerStyle(kCircle);
  hA_p->SetMarkerColor(kBlue);
  hA_p->SetLineColor(kBlue);

  auto hA_m = hF_m.GetAsymmetry(&hB_m);
  hA_m->SetName("hA_m");
  hA_m->SetTitle("Negative Asymmetry;Time (s);Asymmetry");
  hA_m->SetMarkerStyle(kCircle);
  hA_m->SetMarkerColor(kRed);
  hA_m->SetLineColor(kRed);

  THStack hsA("hsA", "Positive/Negative Asymmetry;Time (s);Asymmetry");
  hsA.Add(hA_p);
  hsA.Add(hA_m);

  // hA->Draw("E0");

  // calculate asy using 4 counter method_char
  TH1D hA("hA", "Combined Asymmetry;Time (s);Asymmetry", n_bins, t_min, t_max);
  hA.SetMarkerStyle(kCircle);
  hA.SetLineColor(kBlack);
  hA.SetMarkerColor(kBlack);

  // calculate combined asymmetry using 4-counter method
  for (int i = 1; i <= hA.GetNbinsX(); ++i) {
    //
    double F_p = hF_p.GetBinContent(i);
    double B_p = hB_p.GetBinContent(i);
    double F_m = hF_m.GetBinContent(i);
    double B_m = hB_m.GetBinContent(i);

    // Calculate asymmetry using the clever 4-counter method
    double r = std::sqrt((F_p / B_p) / (F_m / B_m));
    double asy = (1.0 - r) / (1.0 + r);
    double asy_error =
        r * std::sqrt(1.0 / B_p + 1.0 / F_m + 1.0 / B_m + 1.0 / F_p) /
        std::pow(r + 1.0, 2);
    // check if result is sensible
    if (!std::isnan(asy) and !std::isinf(asy) and !std::isnan(asy_error) and
        !std::isinf(asy_error)) {
      hA.SetBinContent(i, asy);
      hA.SetBinError(i, asy_error);
    } else {
      hA.SetBinContent(i, 0.0);
      hA.SetBinError(i, 0.0);
    }
  }

  // fit the spectra
  TF1 fpe("fpe", triumf::bnmr::pulsed_exponential, t_min, t_max, 4);

  fpe.SetNpx(1000);
  fpe.SetLineColor(kRed);

  fpe.SetParName(0, "Nuclear lifetime (s)");
  fpe.SetParName(1, "Beam pulse (s)");
  fpe.SetParName(2, "A_0");
  fpe.SetParName(3, "1/T_1 (s)");

  fpe.FixParameter(0, tau_8Li);
  fpe.FixParameter(1, beam_pulse);
  fpe.SetParLimits(2, 0.0, std::abs(A_8Li));
  fpe.SetParLimits(3, 0.0, 5.0);

  TF1 fpbe("fpbe", triumf::bnmr::pulsed_bi_exponential, t_min, t_max, 6);

  fpbe.SetNpx(1000);
  fpbe.SetLineColor(kBlue);

  fpbe.SetParName(0, "Nuclear lifetime (s)");
  fpbe.SetParName(1, "Beam pulse (s)");
  fpbe.SetParName(2, "A_0");
  fpbe.SetParName(3, "f_slow");
  fpbe.SetParName(4, "1/T_1_slow (s)");
  fpbe.SetParName(5, "1/T_1_fast (s)");

  fpbe.FixParameter(0, tau_8Li);
  fpbe.FixParameter(1, beam_pulse);
  fpbe.SetParLimits(2, 0.0, std::abs(A_8Li));
  fpbe.SetParLimits(3, 0.0, 1.0);
  fpbe.SetParLimits(4, 0.0, 2.0);
  fpbe.SetParLimits(5, 0.0, 10.0);

  // do the fits and save the results
  auto rpe = hA.Fit(&fpe, "QMERS+");
  rpe->SetName("rpe");
  /*
  auto rpbe = hA.Fit(&fpbe, "QMERS+");
  rpbe->SetName("rpbe");
  */

  // ROOT style junk...
  gStyle->SetOptStat("");
  gStyle->SetOptFit(1111);

  // write all the objects to a ROOT file
  std::string filename = config["output"].as<std::string>();
  TFile *file = new TFile(filename.c_str(), "recreate");

  hF_p.Write();
  hB_p.Write();
  hF_m.Write();
  hB_m.Write();
  hsN.Write();
  hA_m->Write();
  hA_p->Write();
  hsA.Write();
  hA.Write();

  fpe.Write();
  rpe->Write();

  // fpbe.Write();
  // rpbe->Write();

  // store the stats from the MC simulation in the ROOT file
  TVectorD mc_stats(2);
  mc_stats[0] = n_decays;
  mc_stats[1] = diff.count();

  mc_stats.Write("mc_stats");

  file->Write();

  // TApplication *app = new TApplication("app", &argc, argv);

  // app->Run(kTRUE);

  return EXIT_SUCCESS;
}
