#ifndef BNMR_FIT_FUNCTIONS_H
#define BNMR_FIT_FUNCTIONS_H

namespace triumf {
namespace bnmr {

double pulsed_exponential(double *x, double *par);
double pulsed_bi_exponential(double *x, double *par);
} // namespace bnmr
} // namespace triumf

#endif // BNMR_FIT_FUNCTIONS_H
