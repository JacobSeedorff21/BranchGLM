#ifndef ParBranchGLMHelpers_H
#define ParBranchGLMHelpers_H

#include <RcppArmadillo.h>
using namespace Rcpp;

double ParLogFact(const arma::vec* y);

void ParCheckBounds(arma::vec* mu, std::string Dist);

arma::vec ParLinkCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset, 
                  std::string Link, std::string Dist);

arma::vec ParDerivativeCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset,
                        arma::vec* mu, std::string Link, std::string Dist);

arma::vec ParVariance(arma::vec* mu, std::string Dist);

double ParLogLikelihoodCpp(const arma::mat* X, const arma::vec* Y, 
                        arma::vec* mu, std::string Dist);

double ParLogLikelihoodNull(const arma::mat* X, const arma::vec* Y, std::string Dist);

arma::vec ParScoreCpp(const arma::mat* X, const arma::vec* Y, arma::vec* Deriv,
                   arma::vec* Var, arma::vec* mu);

arma::mat ParFisherInfoCpp(const arma::mat* X, arma::vec* Deriv, 
                        arma::vec* Var);

arma::vec ParLBFGSHelperCpp(arma::vec* g1, arma::mat* s, arma::mat* y, 
                         unsigned int* k, unsigned int* m, 
                         arma::vec* r, arma::vec* alpha, const arma::mat* Info);

int ParLBFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                   const arma::vec* Y, const arma::vec* Offset,
                   std::string Link, std::string Dist, 
                   double tol, int maxit, unsigned int m = 5, 
                   double C1 = pow(10, -4));

int ParBFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                  const arma::vec* Y, const arma::vec* Offset,
                  std::string Link, std::string Dist,
                  double tol, int maxit, double C1 = pow(10, -4));

int ParFisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, 
                               const arma::vec* Y, const arma::vec* Offset,
                               std::string Link, std::string Dist,
                               double tol, int maxit, 
                               double C1 = pow(10, -4));

int ParLinRegCppShort(arma::vec* beta, const arma::mat* x, const arma::mat* y,
              const arma::vec* offset);

#endif