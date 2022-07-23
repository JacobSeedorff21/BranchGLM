#include <RcppArmadillo.h>
#include "CrossProducts.h"
#include <cmath>
using namespace Rcpp;


// Calculating sum of logs of factorials 

double ParLogFact(const arma::vec* y){
  double sum = 0;
  double Max = max(*y);
  arma::vec logs(Max + 1, arma::fill::zeros);
  for(unsigned int i = 2; i < logs.n_elem;i++){
    logs(i) = logs(i - 1) + log(i);
  }
  for(unsigned int j = 0; j < y->n_elem; j++){
    if(y->at(j) > 1){
      sum += logs(y->at(j));
    }
  }
  
  return(sum);
}

// Checking bounds

void ParCheckBounds(arma::vec* mu, std::string Dist){
  if(Dist == "binomial"){
    mu->transform([](double val){
      if(val <= 0){val = FLT_EPSILON;}
      else if(val >= 1) {val = 1 - FLT_EPSILON;}
      return(val);
    });
  }
  else if(Dist == "poisson" || Dist == "gamma"){
    mu->transform([](double val){
      if(val <= 0){val = FLT_EPSILON;}
      return(val);
    });
  }
}

// Defining Link functions

arma::vec ParLinkCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset, 
                     std::string Link, std::string Dist){
  
  arma::vec XBeta = (*X * *beta) + *Offset;
  arma::vec mu(XBeta.n_elem);
  
  if(Link == "log"){
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = exp(XBeta.at(i));
    }
  }
  else if(Link == "logit"){
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = 1 / (1 + exp(-XBeta.at(i)));
    }
  }
  else if(Link == "probit"){
    mu = arma::normcdf(XBeta);
  }
  else if(Link == "cloglog"){
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = exp(-exp(XBeta.at(i)));
    }
  }
  else if(Link == "inverse"){
    mu = -1 / (XBeta);
  }
  else if(Link == "identity"){
    mu = XBeta;
  }
  else if(Link == "sqrt"){
    mu = pow(XBeta, 2);
  }
  
  ParCheckBounds(&mu, Dist);
  
  return(mu);
}

// Defining Derivative functions

arma::vec ParDerivativeCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset,
                           arma::vec* mu, std::string Link, std::string Dist){
  
  arma::vec Deriv(mu->n_elem);
  
  if(Link == "log"){
    Deriv = *mu; 
  }
  else if(Link == "logit"){
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Deriv.at(i) = mu->at(i) * (1 - mu->at(i));
    }
  }
  else if(Link == "probit"){
    Deriv = arma::normpdf(*X * *beta + *Offset);
  }
  else if(Link == "cloglog"){
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Deriv.at(i) = mu->at(i) * log(mu->at(i));
    }
  }
  else if(Link == "inverse"){
    Deriv = pow(*mu, 2);
  }
  else if(Link == "identity"){
    Deriv.fill(1);
  }
  else if(Link == "sqrt"){
    Deriv = 2 * sqrt(*mu);
  }
  
  return(Deriv);
}

// Defining Link functions

arma::vec ParVariance(arma::vec* mu, std::string Dist){
  
  arma::vec Var(mu->n_elem);
  
  if(Dist == "poisson"){
    Var = *mu; 
  }
  else if(Dist == "Negative binomial"){
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Var.at(i) = mu->at(i) * pow(mu->at(i), 2);
    }
  }
  else if(Dist == "binomial"){
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Var.at(i) = mu->at(i) * (1 - mu->at(i));
    }
  }
  else if(Dist == "gamma"){
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Var.at(i) = pow(mu->at(i), 2);
    }
  }
  else{
    Var.fill(1);
  }
  
  Var.replace(0, FLT_EPSILON);
  
  return(Var);
  
}

// Defining log likelihood for pointer armadillo vector beta

double ParLogLikelihoodCpp(const arma::mat* X, const arma::vec* Y, 
                           arma::vec* mu, std::string Dist){
  
  double LogLik = 0;
  
  if(Dist == "poisson"){
    for(unsigned int i = 0; i < Y->n_elem; i++){
      LogLik += -Y->at(i) * log(mu->at(i)) + mu->at(i);
    }
  }
  else if(Dist == "binomial"){
    for(unsigned int i = 0; i < Y->n_elem; i++){
      double theta = mu->at(i) / (1 - mu->at(i));
      LogLik += -Y->at(i) * log(theta) + log1p(theta);
    }
  }else if(Dist == "gamma"){
    arma::vec theta = -1 / *mu;
    LogLik = -arma::dot(*Y, theta) - arma::accu(log(-theta));
  }else{
    for(unsigned int i = 0; i < Y->n_elem; i++){
      LogLik += pow(Y->at(i) - mu->at(i), 2) /2;
    }
  }
  return(LogLik);
}

// Defining log likelihood for saturated model

double ParLogLikelihoodSat(const arma::mat* X, const arma::vec* Y, std::string Dist){
  
  double LogLik = 0;
  
  if(Dist == "poisson"){
    for(unsigned int i = 0; i< Y->n_elem;i++){
      if(Y->at(i) !=0){
        LogLik += Y->at(i) * (log(Y->at(i)) - 1);
      }
    }
  }
  else if(Dist == "binomial"){
    LogLik = 0;
  }else if(Dist == "gamma"){
    arma::vec theta = -1 / *Y;
    LogLik = arma::dot(*Y, theta) + arma::accu(log(-theta));
  }else{
    LogLik = 0 ;
  }
  
  return(LogLik);
}

// Defining score with pointers

arma::vec ParScoreCpp(const arma::mat* X, const arma::vec* Y, arma::vec* Deriv,
                      arma::vec* Var, arma::vec* mu){
  
  arma::vec FinalVec(X->n_cols);
  arma::vec w = *Deriv / *Var;
  arma::vec diff = *Y - *mu;
  w.replace(arma::datum::nan, 0);
  
  for(unsigned int i = 0; i < X->n_cols; i++){
    
    FinalVec(i) = -arma::dot(X->col(i) % w, diff);
    
  }
  return FinalVec;
}

// Defining fisher information with pointer

arma::mat ParFisherInfoCpp(const arma::mat* X, arma::vec* Deriv, 
                           arma::vec* Var){
  
  arma::vec w = pow(*Deriv, 2) / *Var;
  w.replace(arma::datum::nan, 0);
  arma::mat FinalMat(X->n_cols, X->n_cols);
  
  for(unsigned int i = 0; i < X->n_cols; i++){
    
    FinalMat(i, i) = arma::dot((X->col(i) % w), X->col(i));
    
    for(unsigned int j = i + 1; j < X->n_cols; j++){
      
      FinalMat(i, j) = arma::dot((X->col(j) % w), X->col(i));
      FinalMat(j, i) = FinalMat(i, j);
      
    } 
    
  }
  return FinalMat;
}

void ParGetStepSize(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                 arma::vec* mu, arma::vec* p, arma::vec* beta, 
                 std::string Dist, std::string Link, 
                 double* f0, double* f1, double* t, double *C1, double* alpha, 
                 std::string method){
  if(method == "backtrack"){
    // Finding alpha with backtracking line search using Armijo-Goldstein condition
    while((*f0 < *f1 + *alpha * *t) && (*alpha > *C1)){
      *alpha /= 2;
      *beta -= *alpha * *p;
      *mu = ParLinkCpp(X, beta, Offset, Link, Dist);
      *f1 = ParLogLikelihoodCpp(X, Y, mu, Dist);
    }
  }else{
    // Add other methods here
  }
}

// Creating LBFGS helper function

arma::vec ParLBFGSHelperCpp(arma::vec* g1, arma::mat* s, arma::mat* y, 
                            int* k, unsigned int* m, 
                            arma::vec* r, arma::vec* alpha, const arma::mat* Info){
  if(*k > 0) {
    unsigned int max = std::min(*k, (int)*m);
    unsigned int index;
    for(unsigned int i = 1; i <= max; i++){
      index = (*k - i) % *m;
      alpha->at(index) = arma::dot(s->col(index), *g1)/arma::dot(y->col(index), s->col(index));
      *g1 -= alpha->at(index) * y->col(index);
      
    }
    index = (*k - 1)% *m;
    *r = *Info * *g1;
    for(unsigned int j = max; j > 0; j--){
      index = (*k - j) % *m;
      *r += s->col(index) * (alpha->at(index) - arma::dot(y->col(index), *r)/arma::dot(y->col(index), s->col(index)));
    }
    
    return *r;
  }
  
  return *Info * *g1;
}

// Creating LBFGS for GLMs for parallel functions
int ParLBFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                      const arma::vec* Y, const arma::vec* Offset,
                      std::string Link, std::string Dist, 
                      double tol, int maxit, unsigned int m = 5, 
                      double C1 = pow(10, -4)){
  
  int k = 0;
  arma::vec mu = ParLinkCpp(X, beta, Offset, Link, Dist);
  arma::vec Deriv = ParDerivativeCpp(X, beta, Offset, &mu, Link, Dist);
  arma::vec Var = ParVariance(&mu, Dist);
  m = std::min(beta->n_elem, m);
  arma::vec p(beta->n_elem);
  arma::vec g0(beta->n_elem);
  arma::vec g1 = ParScoreCpp(X, Y, &Deriv, &Var, &mu);
  arma::vec r(beta->n_elem);
  arma::vec alphavec(m);
  arma::mat s(beta->n_elem, m);
  arma::mat y(beta->n_elem, m);
  arma::mat Info(beta->n_elem, beta->n_elem);
  if(!inv_sympd(Info, ParFisherInfoCpp(X, &Deriv, &Var))){
    return(-2);
  }
  double f0;
  double f1 = ParLogLikelihoodCpp(X, Y, &mu, Dist);
  double t;
  double alpha;
  
  while(arma::norm(g1) > tol){
    if(k >= maxit){ 
      k = -1;
      break;
    }
    alpha = 1;
    g0 = g1;
    f0 = f1;
    p = -ParLBFGSHelperCpp(&g1, &s, &y, &k, &m, &r, &alphavec, &Info);
    t = -C1 * arma::dot(g0, p);
    *beta += alpha * p;
    mu = ParLinkCpp(X, beta, Offset, Link, Dist);
    f1 = ParLogLikelihoodCpp(X, Y, &mu, Dist);
    
    // Finding alpha with backtracking linesearch using Armijo-Goldstein condition
    ParGetStepSize(X, Y, Offset, &mu, &p, beta, Dist, Link, &f0 ,&f1, &t, &C1, &alpha, "backtrack");
    
    if(std::fabs(f1 -  f0) < tol || all(abs(alpha * p) < tol)){
      if(std::isinf(f1)|| beta->has_nan()){
        k = -2;
      }
      k++;
      break;}
    
    Deriv = ParDerivativeCpp(X, beta, Offset, &mu, Link, Dist);
    Var = ParVariance(&mu, Dist);
    g1 = ParScoreCpp(X, Y, &Deriv, &Var, &mu);
    s.col(k % m) = alpha * p;
    y.col(k % m) = g1 - g0;
    k++;
  }
  return(k);
}


// Creating BFGS for GLMs for parallel functions
int ParBFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                     const arma::vec* Y, const arma::vec* Offset,
                     std::string Link, std::string Dist,
                     double tol, int maxit, double C1 = pow(10, -4)){
  
  int k = 0;
  arma::vec mu = ParLinkCpp(X, beta, Offset, Link, Dist);
  arma::vec Deriv = ParDerivativeCpp(X, beta, Offset, &mu, Link, Dist);
  arma::vec Var = ParVariance(&mu, Dist);
  arma::vec g1 = ParScoreCpp(X, Y, &Deriv, &Var, &mu);
  arma::vec p(beta->n_elem);
  arma::vec s(beta->n_elem);
  arma::vec y(beta->n_elem);
  arma::vec g0(beta->n_elem);
  arma::mat H1(beta->n_elem, beta->n_elem);
  
  if(!inv_sympd(H1, ParFisherInfoCpp(X, &Deriv, &Var))){
    return(-2);
  }
  double f0;
  double f1 = ParLogLikelihoodCpp(X, Y, &mu, Dist);
  double rho;
  double alpha;
  double t;
  
  while(arma::norm(g1) > tol){
    if(k >= maxit){ 
      k = -1;
      break;
    }
    alpha = 1;
    g0 = g1;
    f0 = f1;
    p = -H1 * g1;
    t = -C1 * arma::dot(g0, p);
    *beta += alpha * p;
    mu = ParLinkCpp(X, beta, Offset, Link, Dist);
    f1 = ParLogLikelihoodCpp(X, Y, &mu, Dist);
    
    // Finding alpha with backtracking linesearch using Armijo-Goldstein condition
    ParGetStepSize(X, Y, Offset, &mu, &p, beta, Dist, Link, &f0 ,&f1, &t, &C1, &alpha, "backtrack");
    k++;
    
    if(std::fabs(f1 -  f0) < tol || all(abs(alpha * p) < tol)){
      if(std::isinf(f1)|| beta->has_nan()){
        k = -1;
      }
      break;}
    
    Deriv = ParDerivativeCpp(X, beta, Offset, &mu, Link, Dist);
    Var = ParVariance(&mu, Dist);
    g1 = ParScoreCpp(X, Y, &Deriv, &Var, &mu);
    s = alpha * p;
    y = g1 - g0;
    rho = 1/arma::dot(s, y);
    
    // Calculating next approximate inverse hessian
    
    H1 = (arma::diagmat(arma::ones(beta->n_elem)) - rho * s * y.t()) * H1 * 
      (arma::diagmat(arma::ones(beta->n_elem)) - rho * y * s.t()) + rho * s * s.t();
  }
  return(k);
}


// Creating Fisher Scoring for GLMs for parallel functions
int ParFisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, 
                             const arma::vec* Y, const arma::vec* Offset,
                             std::string Link, std::string Dist,
                             double tol, int maxit, 
                             double C1 = pow(10, -4)){
  
  int k = 0;
  arma::vec mu = ParLinkCpp(X, beta, Offset, Link, Dist);
  arma::vec Deriv = ParDerivativeCpp(X, beta, Offset, &mu, Link, Dist);
  arma::vec Var = ParVariance(&mu, Dist);
  arma::vec g1 = ParScoreCpp(X, Y, &Deriv, &Var, &mu);
  arma::vec p(beta->n_elem);
  arma::mat H1 = ParFisherInfoCpp(X, &Deriv, &Var);
  double f0;
  double f1 = ParLogLikelihoodCpp(X, Y, &mu, Dist);
  double alpha;
  double t;
  
  while(arma::norm(g1) > tol){
    alpha = 1;
    
    // Checks if we've reached maxit iterations and stops if we have
    if(k >= maxit){ 
      k = -1;
      break;
    }
    
    f0 = f1;
    if(!arma::solve(p, -H1, g1, arma::solve_opts::no_approx)){
      return(-2);
    }
    t = -C1 * arma::dot(g1, p);
    *beta += alpha * p;
    mu = ParLinkCpp(X, beta, Offset, Link, Dist);
    f1 = ParLogLikelihoodCpp(X, Y, &mu, Dist);
    
    // Finding alpha with backtracking linesearch using Armijo-Goldstein condition
    ParGetStepSize(X, Y, Offset, &mu, &p, beta, Dist, Link, &f0 ,&f1, &t, &C1, &alpha, "backtrack");
    k++;
    
    if(std::fabs(f1 -  f0) < tol || all(abs(alpha * p) < tol)){
      if(std::isinf(f1)|| beta->has_nan()){
        k = -1;
      }
      break;}
    
    Deriv = ParDerivativeCpp(X, beta, Offset, &mu, Link, Dist);
    Var = ParVariance(&mu, Dist);
    g1 = ParScoreCpp(X, Y, &Deriv, &Var, &mu);
    H1 = ParFisherInfoCpp(X, &Deriv, &Var);
  }
  return(k);
}

int ParLinRegCppShort(arma::vec* beta, const arma::mat* x, const arma::mat* y,
                   const arma::vec* offset){
  
  arma::mat FinalMat = XTX(x, 16);
  
  // calculating inverse of X'X
  arma::mat InvXX(x->n_cols, x->n_cols, arma::fill::zeros);
  if(!arma::inv_sympd(InvXX, FinalMat)){
    return(-2);
  }
  
  // Calculating beta, dispersion parameter, and beta variances
  *beta = InvXX * x->t() * (*y - *offset);
  return(1);
}

// Gets initial values for gamma and gaussian regression with log/inverse/sqrt link with 
// transformed y linear regression
void PargetInit(arma::vec* beta, const arma::mat* X, const arma::vec* Y, 
             const arma::vec* Offset, std::string Dist, std::string Link){
  if(Link == "log" && (Dist == "gamma" || Dist == "gaussian")){
    arma::vec NewY = log(*Y);
    ParLinRegCppShort(beta, X, &NewY, Offset);
  }else if(Link == "inverse" && (Dist == "gamma" || Dist == "gaussian")){
    const arma::vec NewY = -1 / (*Y);
    ParLinRegCppShort(beta, X, &NewY, Offset);
  }else if(Link == "sqrt" && (Dist == "gamma" || Dist == "gaussian"|| Dist == "poisson")){
    const arma::vec NewY = sqrt(*Y);
    ParLinRegCppShort(beta, X, &NewY, Offset);
  }else if(Link == "identity" && (Dist == "gamma" || Dist == "poisson")){
    ParLinRegCppShort(beta, X, Y, Offset);
  }
}
