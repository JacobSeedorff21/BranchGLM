#ifndef EffBranchGLMHelpers_H
#define EffBranchGLMHelpers_H

#include <RcppArmadillo.h>
using namespace Rcpp;

arma::vec EffScoreCpp(const arma::mat* X, const arma::vec* XY, const arma::vec* Y, arma::vec* Deriv,
                   arma::vec* Var, arma::vec* mu);

arma::mat EffFisherInfoCpp(const arma::mat* X, arma::vec* Deriv, 
                        arma::vec* Var);

arma::vec EffLBFGSHelperCpp(arma::vec* g1, arma::mat* s, arma::mat* y, 
                         unsigned int* k, unsigned int* m, 
                         arma::vec* r, arma::vec* alpha, const arma::mat* Info);



int EffLBFGSGLMCpp(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX, 
                   const arma::vec* Y, const arma::vec* Offset,
                   std::string Link, std::string Dist, 
                   double tol, int maxit, unsigned int m, bool UseXTWX);

int EffBFGSGLMCpp(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX,
                  const arma::vec* Y, const arma::vec* Offset,
                  std::string Link, std::string Dist,
			double tol, int maxit, bool UseXTWX);

int EffFisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX,
                               const arma::vec* Y, const arma::vec* Offset,
                               std::string Link, std::string Dist,
                               double tol, int maxit, bool UseXTWX);

int EffLinRegCppShort(arma::vec* beta, const arma::mat* x, const arma::mat* XTWX,
const arma::mat* y,
              const arma::vec* offset);

void EffgetInit(arma::vec* beta, const arma::mat* X, const arma::mat* XTWX,
		    const arma::vec* Y, 
                const arma::vec* Offset, std::string Dist, std::string Link, 
		    bool* UseXTWX);

#endif