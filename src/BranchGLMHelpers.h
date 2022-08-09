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
                   double tol, int maxit, int m = 5);

int BFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                  const arma::vec* Y, const arma::vec* Offset,
                  std::string Link, std::string Dist,
                  double tol, int maxit);

int FisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, 
                               const arma::vec* Y, const arma::vec* Offset,
                               std::string Link, std::string Dist,
                               double tol, int maxit);

List BranchGLMFitCpp(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                std::string method,  unsigned int m, std::string Link, std::string Dist,
                unsigned int nthreads, double tol, int maxit, bool GetInit = true);

int LinRegCppShort(arma::vec* beta, const arma::mat* x, const arma::mat* y,
              const arma::vec* offset);

double GetDispersion(const arma::mat* X, const arma::vec* Y, 
                     arma::vec* mu, double LogLik, std::string Dist, 
                     double tol);

void getInit(arma::vec* beta, const arma::mat* X, const arma::vec* Y, 
             const arma::vec* Offset, std::string Dist, std::string Link);

#endif