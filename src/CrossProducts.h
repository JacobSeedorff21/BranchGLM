#ifndef CrossProducts_H
#define CrossProducts_H

#include <RcppArmadillo.h>
using namespace Rcpp;

arma::mat ParXTX(const arma::mat* x);

arma::mat XTX(const arma::mat* x, unsigned int B = 16);

arma::mat ParXTWX(const arma::mat* x, const arma::vec* w);

arma::mat XTWX(const arma::mat* x, const arma::vec* w, 
               unsigned int B = 16);

#endif