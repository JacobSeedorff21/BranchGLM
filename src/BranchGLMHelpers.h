#ifndef BranchGLMHelpers_H
#define BranchGLMHelpers_H

#include <RcppArmadillo.h>
using namespace Rcpp;

double LogFact(const arma::vec* y);

void CheckBounds(arma::vec* mu, std::string Dist);

arma::vec LinkCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset, 
                  std::string Link, std::string Dist);

arma::vec DerivativeCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset,
                        arma::vec* mu, std::string Link, std::string Dist);

arma::vec Variance(arma::vec* mu, std::string Dist);

double LogLikelihoodCpp(const arma::mat* X, const arma::vec* Y, 
                        arma::vec* mu, std::string Dist);

double LogLikelihoodNull(const arma::mat* X, const arma::vec* Y, std::string Dist);

double LogLikelihoodSat(const arma::mat* X, const arma::vec* Y, std::string Dist);

arma::vec ScoreCpp(const arma::mat* X, const arma::vec* Y, arma::vec* Deriv,
                   arma::vec* Var, arma::vec* mu);

arma::mat FisherInfoCpp(const arma::mat* X, arma::vec* Deriv, 
                        arma::vec* Var);

arma::vec LBFGSHelperCpp(arma::vec* g1, arma::mat* s, arma::mat* y, 
                         int* k, unsigned int* m, 
                         arma::vec* r, arma::vec* alpha, const arma::mat* Info);

int LBFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                   const arma::vec* Y, const arma::vec* Offset,
                   std::string Link, std::string Dist, 
                   double tol = pow(10, -8), int m = 5, 
                   double C1 = pow(10, -4));

int BFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                  const arma::vec* Y, const arma::vec* Offset,
                  std::string Link, std::string Dist,
                  double tol =  pow(10, -8), double C1 = pow(10, -4));

int FisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, 
                               const arma::vec* Y, const arma::vec* Offset,
                               std::string Link, std::string Dist,
                               double tol = pow(10, -8), 
                               double C1 = pow(10, -4));

List BranchGLMFitCpp(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                std::string method,  unsigned int m, std::string Link, std::string Dist,
                unsigned int nthreads, double tol, bool intercept);

#endif