#ifndef CrossProducts_H
#define CrossProducts_H

#include <RcppArmadillo.h>
using namespace Rcpp;

arma::mat ParXTX(const arma::mat* x);

arma::mat XTX(const arma::mat* x, unsigned int B = 16);

#endif