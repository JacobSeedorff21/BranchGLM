#ifndef ParBranchGLMHelpers_H
#define ParBranchGLMHelpers_H

#include <RcppArmadillo.h>
using namespace Rcpp;

arma::vec ParScoreCpp(const arma::mat* X, const arma::vec* XY, const arma::vec* Y, arma::vec* Deriv,
                   arma::vec* Var, arma::vec* mu);

arma::mat ParFisherInfoCpp(const arma::mat* X, arma::vec* Deriv, 
                        arma::vec* Var);

arma::vec ParLBFGSHelperCpp(arma::vec* g1, arma::mat* s, arma::mat* y, 
                         unsigned int* k, unsigned int* m, 
                         arma::vec* r, arma::vec* alpha, const arma::mat* Info);



int ParLBFGSGLMCpp(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX, 
                   const arma::vec* Y, const arma::vec* Offset,
                   std::string Link, std::string Dist, 
                   double tol, int maxit, unsigned int m, bool UseXTWX);

int ParBFGSGLMCpp(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX,
                  const arma::vec* Y, const arma::vec* Offset,
                  std::string Link, std::string Dist,
			double tol, int maxit, bool UseXTWX);

int ParFisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX,
                               const arma::vec* Y, const arma::vec* Offset,
                               std::string Link, std::string Dist,
                               double tol, int maxit, bool UseXTWX);

int ParLinRegCppShort(arma::vec* beta, const arma::mat* x, const arma::mat* XTWX,
const arma::mat* y,
              const arma::vec* offset);

void PargetInit(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX,
		    const arma::vec* Y, 
                const arma::vec* Offset, std::string Dist, std::string Link, 
		    bool* UseXTWX);

arma::vec ParLinkCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset, 
                     std::string Link, std::string Dist);

double ParLogLikelihoodCpp(const arma::mat* X, const arma::vec* Y, 
                           arma::vec* mu, std::string Dist);

#endif