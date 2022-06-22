#include <RcppArmadillo.h>
#include "CrossProducts.h"
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

// Calculating sum of logs of factorials, this is used for poisson likelihood
double LogFact(const arma::vec* y){
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
void CheckBounds(arma::vec* mu, std::string Dist){
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
arma::vec LinkCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset, 
                  std::string Link, std::string Dist){
  
  // Calculating linear predictors
  arma::vec XBeta = (*X * *beta) + *Offset;
  arma::vec mu(XBeta.n_elem);
  
  if(Link == "log"){
#pragma omp parallel for
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = exp(XBeta.at(i));
    }
  }
  else if(Link == "logit"){
#pragma omp parallel for
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = 1 / (1 + exp(-XBeta.at(i)));
    }
  }
  else if(Link == "probit"){
    mu = arma::normcdf(XBeta);
  }
  else if(Link == "cloglog"){
#pragma omp parallel for
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = exp(-exp(XBeta.at(i)));
    }
  }
  else if(Link == "inverse"){
#pragma omp parallel for
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = -1 / (XBeta.at(i));
    }
  }
  else if(Link == "identity"){
    mu = XBeta;
  }
  else if(Link == "sqrt"){
#pragma omp parallel for
    for(unsigned int i = 0; i < Offset->n_elem; i++){
      mu.at(i) = pow(XBeta.at(i), 2);
    }
  }
  
  CheckBounds(&mu, Dist);
  
  return(mu);
}

// Defining Derivative functions for each link function
arma::vec DerivativeCpp(const arma::mat* X, arma::vec* beta, const arma::vec* Offset,
                        arma::vec* mu, std::string Link, std::string Dist){
  
  arma::vec Deriv(mu->n_elem);
  
  if(Link == "log"){
    Deriv = *mu; 
  }
  else if(Link == "logit"){
#pragma omp parallel for
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Deriv.at(i) = mu->at(i) * (1 - mu->at(i));
    }
  }
  else if(Link == "probit"){
    Deriv = arma::normpdf(*X * *beta + *Offset);
  }
  else if(Link == "cloglog"){
#pragma omp parallel for
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Deriv.at(i) = mu->at(i) * log(mu->at(i));
    }
  }
  else if(Link == "inverse"){
    Deriv = pow(*mu, 2);
  }
  else if(Link == "identity"){
    Deriv.fill(1);
  }else if(Link == "sqrt"){
    Deriv = 2 * sqrt(*mu);
  }
  
  return(Deriv);
}

// Defining Variance functions for each family
arma::vec Variance(arma::vec* mu, std::string Dist){
  
  arma::vec Var(mu->n_elem);
  
  if(Dist == "poisson"){
    Var = *mu; 
  }
  else if(Dist == "binomial"){
#pragma omp parallel for
    for(unsigned int i = 0; i < mu->n_elem; i++){
      Var.at(i) = mu->at(i) * (1 - mu->at(i));
    }
  }
  else if(Dist == "gamma"){
#pragma omp parallel for
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

// Defining log likelihood function
double LogLikelihoodCpp(const arma::mat* X, const arma::vec* Y, 
                        arma::vec* mu, std::string Dist){
  
  double LogLik = 0;
  
  if(Dist == "poisson"){
#pragma omp parallel for reduction(+:LogLik)
    for(unsigned int i = 0; i < Y->n_elem; i++){
      LogLik += -Y->at(i) * log(mu->at(i)) + mu->at(i);
    }
  }
  else if(Dist == "binomial"){
#pragma omp parallel for reduction(+:LogLik)
    for(unsigned int i = 0; i < Y->n_elem; i++){
      double theta = mu->at(i) / (1 - mu->at(i));
      LogLik += -Y->at(i) * log(theta) + log1p(theta);
    }
  }else if(Dist == "gamma"){
#pragma omp parallel for reduction(+:LogLik)
    for(unsigned int i = 0; i < Y->n_elem; i++){
      double theta = -1 / mu->at(i);
      LogLik += -Y->at(i) * theta - log(-theta);
    }
  }else{
#pragma omp parallel for reduction(+:LogLik)
    for(unsigned int i = 0; i < Y->n_elem; i++){
      LogLik += pow(Y->at(i) - mu->at(i), 2) /2;
    }
  }
  return(LogLik);
}

// Defining log likelihood for saturated model
double LogLikelihoodSat(const arma::mat* X, const arma::vec* Y, std::string Dist){
  
  double LogLik = 0;
  
  if(Dist == "poisson"){
    for(unsigned int i = 0; i< Y->n_elem;i++){
      if(Y->at(i) != 0){
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
    LogLik = 0;
  }
  
  return(LogLik);
}

// Defining score function
arma::vec ScoreCpp(const arma::mat* X, const arma::vec* Y, arma::vec* Deriv,
                   arma::vec* Var, arma::vec* mu){
  
  // Initializing vector to store results
  arma::vec FinalVec(X->n_cols);
  
  //Calculating weights and difference in observed and expected
  arma::vec w = *Deriv / *Var;
  arma::vec diff = *Y - *mu;
  w.replace(arma::datum::nan, 0);
  
  // Calculating score
#pragma omp parallel for
  for(unsigned int i = 0; i < X->n_cols; i++){
    
    FinalVec(i) = -arma::dot(X->col(i) % w, diff);
    
  }
  return FinalVec;
}

// Defining fisher information function
arma::mat FisherInfoCpp(const arma::mat* X, arma::vec* Deriv, 
                        arma::vec* Var){
  
  // Initializing matrix to store results
  arma::mat FinalMat(X->n_cols, X->n_cols);
  
  // Calculating weight vector, this is the diagonal of the W matrix
  arma::vec w = pow(*Deriv, 2) / *Var;
  w.replace(arma::datum::nan, 0);
  checkUserInterrupt();
  
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int i = 0; i < X->n_cols; i++){
    
    FinalMat(i, i) = arma::dot((X->col(i) % w), X->col(i));
    
    for(unsigned int j = i + 1; j < X->n_cols; j++){
      
      FinalMat(i, j) = arma::dot((X->col(j) % w), X->col(i));
      FinalMat(j, i) = FinalMat(i, j);
      
    } 
  }
  return FinalMat;
}

// LBFGS helper function
arma::vec LBFGSHelperCpp(arma::vec* g1, arma::mat* s, arma::mat* y, 
                         int* k, int* m, 
                         arma::vec* r, arma::vec* alpha, const arma::mat* Info){
  if(*k > 0) {
    unsigned int max = std::min(*k, *m);
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
  
  // Returns this if no gradients have been evaluated yet
  return *Info * *g1;
}

// LBFGS
int LBFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
                const arma::vec* Y, const arma::vec* Offset,
                std::string Link, std::string Dist, 
                double tol, int maxit, int m, 
                double C1 = pow(10, -4)){
  
  // Initializing vectors and matrices 
  arma::vec mu = LinkCpp(X, beta, Offset, Link, Dist);
  arma::vec Deriv = DerivativeCpp(X, beta, Offset, &mu, Link, Dist);
  arma::vec Var = Variance(&mu, Dist);
  arma::vec p(beta->n_elem);
  arma::vec g0(beta->n_elem);
  arma::vec g1 = ScoreCpp(X, Y, &Deriv, &Var, &mu);
  arma::vec r(beta->n_elem);
  arma::vec alphavec(m);
  arma::mat s(beta->n_elem, m);
  arma::mat y(beta->n_elem, m);
  arma::mat Info(beta->n_elem, beta->n_elem);
  if(!inv_sympd(Info, FisherInfoCpp(X, &Deriv, &Var))){
    warning("Fisher information not invertible");
    return(-2);
  }
  
  // Initializing int and doubles
  int k = 0;
  double f0;
  double f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
  double t;
  double alpha;
  
  // Fitting the model
  while(arma::norm(g1) > tol){
    checkUserInterrupt();
    if(k == maxit){ 
      warning("LBFGS failed to converge");
      k = -1;
      break;
    }
    alpha = 1;
    g0 = g1;
    f0 = f1;
    
    // Calculating p (search direction) based on L-BFGS approximation to inverse info
    p = -LBFGSHelperCpp(&g1, &s, &y, &k, &m, &r, &alphavec, &Info);
    t = -C1 * arma::dot(g0, p);
    *beta += alpha * p;
    mu = LinkCpp(X, beta, Offset, Link, Dist);
    f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
    
    // Finding alpha with backtracking linesearch using Armijo-Goldstein condition
    while((f0 < f1 + alpha * t) && (alpha > C1)){
      alpha /= 2;
      *beta -= alpha * p;
      mu = LinkCpp(X, beta, Offset, Link, Dist);
      f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
    }
    // Checking for convergence or nan/inf
    if(std::fabs(f1 -  f0) < tol || all(abs(alpha * p) < tol)){
      if(std::isinf(f1) || beta->has_nan()){
        warning("LBFGS failed to converge");
        k = -2;
      }
      k++;
      break;}
    
    Deriv = DerivativeCpp(X, beta, Offset, &mu, Link, Dist);
    Var = Variance(&mu, Dist);
    g1 = ScoreCpp(X, Y, &Deriv, &Var, &mu);
    s.col(k % m) = alpha * p;
    y.col(k % m) = g1 - g0;
    k++;
  }
  return(k);
}


// BFGS
int BFGSGLMCpp(arma::vec* beta, const arma::mat* X, 
               const arma::vec* Y, const arma::vec* Offset,
               std::string Link, std::string Dist,
               double tol, int maxit, 
               double C1 = pow(10, -4)){
  
  // Initializing vectors and matrices
  arma::vec mu = LinkCpp(X, beta, Offset, Link, Dist);
  arma::vec Deriv = DerivativeCpp(X, beta, Offset, &mu, Link, Dist);
  arma::vec Var = Variance(&mu, Dist);
  arma::vec g1 = ScoreCpp(X, Y, &Deriv, &Var, &mu);
  arma::vec p(beta->n_elem);
  arma::vec s(beta->n_elem);
  arma::vec y(beta->n_elem);
  arma::vec g0(beta->n_elem);
  arma::mat H1(beta->n_elem, beta->n_elem);
  if(!inv_sympd(H1, FisherInfoCpp(X, &Deriv, &Var))){
    warning("Fisher information not invertible");
    return(-2);
  }
  
  // Initializing int and doubles
  int k = 0;
  double f0;
  double f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
  double rho;
  double alpha;
  double t;
  
  // Fitting the model
  while(arma::norm(g1) > tol){
    checkUserInterrupt();
    if(k == maxit){ 
      warning("BFGS failed to converge");
      k = -1;
      break;
    }
    alpha = 1;
    g0 = g1;
    f0 = f1;
    p = -H1 * g1;
    t = -C1 * arma::dot(g0, p);
    *beta += alpha * p;
    mu = LinkCpp(X, beta, Offset, Link, Dist);
    f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
    
    // Finding alpha with backtracking linesearch using Armijo-Goldstein condition
    
    while((f0 < f1 + alpha * t) && (alpha > C1)){
      alpha /= 2;
      *beta -= alpha * p;
      mu = LinkCpp(X, beta, Offset, Link, Dist);
      f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
    }
    
    k++;
    
    // Checking for convergence or nan/inf
    if(std::fabs(f1 -  f0) < tol || all(abs(alpha * p) < tol)){
      if(std::isinf(f1)|| beta->has_nan()){
        warning("BFGS failed to converge");
        k = -1;
      }
      break;}
    
    Deriv = DerivativeCpp(X, beta, Offset, &mu, Link, Dist);
    Var = Variance(&mu, Dist);
    g1 = ScoreCpp(X, Y, &Deriv, &Var, &mu);
    s = alpha * p;
    y = g1 - g0;
    rho = 1/arma::dot(s, y);
    
    // Calculating next approximate inverse hessian
    H1 = (arma::diagmat(arma::ones(beta->n_elem)) - rho * s * y.t()) * H1 * 
      (arma::diagmat(arma::ones(beta->n_elem)) - rho * y * s.t()) + rho * s * s.t();
  }
  return(k);
}


// Fisher's scoring

int FisherScoringGLMCpp(arma::vec* beta, const arma::mat* X, 
                        const arma::vec* Y, const arma::vec* Offset,
                        std::string Link, std::string Dist,
                        double tol, int maxit, 
                        double C1 = pow(10, -4)){
  
  // Initializing vector and matrices
  arma::vec mu = LinkCpp(X, beta, Offset, Link, Dist);
  arma::vec Deriv = DerivativeCpp(X, beta, Offset, &mu, Link, Dist);
  arma::vec Var = Variance(&mu, Dist);
  arma::vec g1 = ScoreCpp(X, Y, &Deriv, &Var, &mu);
  arma::vec p(beta->n_elem);
  arma::mat H1 = FisherInfoCpp(X, &Deriv, &Var);
  
  // Initializing int and doubles
  int k = 0;
  double f0;
  double f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
  double alpha;
  double t;
  
  // Fitting the model
  while(arma::norm(g1) > tol){
    checkUserInterrupt();
    alpha = 1;
    
    // Checks if we've reached maxit iterations and stops if we have
    if(k == maxit){ 
      warning("Fisher Scoring failed to converge");
      k = -1;
      break;
    }
    
    f0 = f1;
    if(!arma::solve(p, -H1, g1, arma::solve_opts::no_approx)){
      warning("Fisher info not invertible");
      return(-2);
    };
    t = -C1 * arma::dot(g1, p);
    *beta += alpha * p;
    mu = LinkCpp(X, beta, Offset, Link, Dist);
    f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
    
    // Finding alpha with backtracking line search using Armijo-Goldstein condition
    while((f0 < f1 + alpha * t) && (alpha > C1)){
      alpha /= 2;
      *beta -= alpha * p;
      mu = LinkCpp(X, beta, Offset, Link, Dist);
      f1 = LogLikelihoodCpp(X, Y, &mu, Dist);
    }
    k++;
    
    //Checking for convergence or nan/inf
    if(std::fabs(f1 -  f0) < tol || all(abs(alpha * p) < tol)){
      if(std::isinf(f1)|| beta->has_nan()){
        warning("Fisher Scoring failed to converge");
        k = -1;
      }
      break;}
    
    Deriv = DerivativeCpp(X, beta, Offset, &mu, Link, Dist);
    Var = Variance(&mu, Dist);
    g1 = ScoreCpp(X, Y, &Deriv, &Var, &mu);
    H1 = FisherInfoCpp(X, &Deriv, &Var);
  }
  return(k);
}

int LinRegCpp(arma::vec* beta, const arma::mat* x, const arma::mat* y,
              const arma::vec* offset, arma::vec* SE1, unsigned int nthreads){
  
  arma::mat FinalMat(x->n_cols, x->n_cols);
  if(nthreads > 1){
    FinalMat = ParXTX(x);
  }else{
    FinalMat = XTX(x, 16);
  }
  
  // calculating inverse of X'X
  arma::mat InvXX(x->n_cols, x->n_cols, arma::fill::zeros);
  if(!arma::inv_sympd(InvXX, FinalMat)){
    warning("Fisher info not invertible");
    return(-2);
  }
  
  // Calculating beta, dispersion parameter, and beta variances
  *beta = InvXX * x->t() * (*y - *offset);
  *SE1 = arma::diagvec(InvXX);
  return(1);
} 

int LinRegCppShort(arma::vec* beta, const arma::mat* x, const arma::mat* y,
                   const arma::vec* offset, unsigned int nthreads){
  
  arma::mat FinalMat(x->n_cols, x->n_cols);
  if(nthreads > 1){
    FinalMat = ParXTX(x);
  }else{
    FinalMat = XTX(x, 16);
  }
  // calculating inverse of X'X
  arma::mat InvXX(x->n_cols, x->n_cols, arma::fill::zeros);
  if(!arma::inv_sympd(InvXX, FinalMat)){
    warning("Fisher info not invertible");
    return(-2);
  }
  
  // Calculating beta, dispersion parameter, and beta variances
  *beta = InvXX * x->t() * (*y - *offset);
  return(1);
}

double GetDispersion(const arma::mat* X, const arma::vec* Y, 
                     arma::vec* mu, double LogLik, std::string Dist, 
                     double tol, double C1 = pow(10, -4)){
  double dispersion = 1;
  if(Dist == "gaussian"){
    dispersion = arma::accu(pow(*Y - *mu, 2)) / (X->n_rows - X->n_cols);
  }else if(Dist == "gamma"){
    unsigned int it = 0;
    double alpha = 1;
    double dispersion2 = dispersion + 2 * tol;
    double fixed = LogLik + arma::accu(log(*Y)) + X->n_rows;
    
    // Initializing score and info
    double score = fixed + X->n_rows * (log(dispersion) - boost::math::digamma(dispersion)); 
    
    double info = X->n_rows * (-1 / dispersion + boost::math::trigamma(dispersion));
    
    while(std::fabs(score) > tol && std::fabs(dispersion - dispersion2) > tol && it < 25){
      alpha = 1;
      dispersion2 = dispersion;
      dispersion += score / info;
      while(dispersion <= 0 && alpha >= C1){
        alpha /= 2;
        dispersion -= alpha * score / info; 
      }
      score = fixed + X->n_rows * (log(dispersion) - boost::math::digamma(dispersion));
      info = X->n_rows * (-1 / dispersion + boost::math::trigamma(dispersion));
      it++;
    }
    dispersion = 1 / dispersion;
  }
  return(dispersion);
}

// Gets initial values for gamma, poisson, and gaussian regression
void getInit(arma::vec* beta, const arma::mat* X, const arma::vec* Y, 
             const arma::vec* Offset, std::string Dist, std::string Link, 
             unsigned int nthreads){
  
  if(Link == "log" && (Dist == "gamma" || Dist == "gaussian")){
    arma::vec NewY = log(*Y);
    LinRegCppShort(beta, X, &NewY, Offset, nthreads);
  }else if(Link == "inverse" && (Dist == "gamma" || Dist == "gaussian")){
    const arma::vec NewY = -1 / (*Y);
    LinRegCppShort(beta, X, &NewY, Offset, nthreads);
  }else if(Link == "sqrt" && (Dist == "gamma" || Dist == "gaussian"|| Dist == "poisson")){
    const arma::vec NewY = sqrt(*Y);
    LinRegCppShort(beta, X, &NewY, Offset, nthreads);
  }else if(Link == "identity" && (Dist == "gamma" || Dist == "poisson")){
    LinRegCppShort(beta, X, Y, Offset, nthreads);
  }
}

// [[Rcpp::export]]
List BranchGLMfit(NumericMatrix x, NumericVector y, NumericVector offset,
                  NumericVector init,
                  std::string method,  unsigned int m, std::string Link, 
                  std::string Dist,
                  unsigned int nthreads, double tol, int maxit, bool GetInit){
  
  // Initializing vectors and matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true); 
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Init(init.begin(), init.size(), false, true);
  arma::vec beta = Init;
  arma::mat Info(beta.n_elem, beta.n_elem);
  arma::vec SE1(beta.n_elem);
  
  // Initializing doubles
  double Iter;
  double dispersion = 1;
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Getting initial values
  if(GetInit){
    getInit(&beta, &X, &Y, &Offset, Dist, Link, nthreads);
  }
  
  // Fitting model
  if(Dist == "gaussian" && Link == "identity"){
    Iter = LinRegCpp(&beta, &X, &Y, &Offset, &SE1, nthreads);
  }else if(method == "BFGS"){
    Iter = BFGSGLMCpp(&beta, &X, &Y, &Offset, Link, Dist, tol, maxit);
  }
  else if(method == "LBFGS"){
    Iter = LBFGSGLMCpp(&beta, &X, &Y, &Offset, Link, Dist, tol, maxit, m);
  }
  else{
    Iter = FisherScoringGLMCpp(&beta, &X, &Y, &Offset, Link, Dist, tol, maxit);
  }
  // Checking for non-invertible fisher info error
  if(Iter == -2){
    stop("Algorithm failed to converge because the fisher info was not invertible");
  }
  
  // Calculating means
  arma::vec mu = LinkCpp(&X, &beta, &Offset, Link, Dist);
  
  // Calculating variances for betas for non-linear regression
  if(Dist != "gaussian" || Link != "identity"){
    
    // Calculating derivatives, and variances to be used for info
    arma::vec Deriv = DerivativeCpp(&X, &beta, &Offset, &mu, Link, Dist);
    arma::vec Var = Variance(&mu, Dist);
    
    // Calculating info and initaliazing inverse info
    Info = FisherInfoCpp(&X, &Deriv, &Var);
    arma::mat InfoInv = Info;
    
    // Calculating inverse info and returning error if not invertible
    if(!arma::inv_sympd(InfoInv, Info)){
      stop("Fisher info not invertible");
    }
    
    // Calculating variances which are later converted to SEs
    SE1 = arma::diagvec(InfoInv);
  }
  
  NumericVector SE = NumericVector(SE1.begin(), SE1.end());
  
  // Converting variances to SEs
  SE = sqrt(SE);
  
  // Returning results
  double satLogLik = LogLikelihoodSat(&X, &Y, Dist);
  double LogLik = -LogLikelihoodCpp(&X, &Y, &mu, Dist);
  double resDev = -2 * (LogLik - satLogLik);
  double AIC = -2 * LogLik + 2 * X.n_cols;
  
  NumericVector beta1 = NumericVector(beta.begin(), beta.end());
  
  arma::vec linPreds = X * beta + Offset;
  
  NumericVector linPreds1 = NumericVector(linPreds.begin(), linPreds.end());
  
  // Getting dispersion paramater
  dispersion = GetDispersion(&X, &Y, &mu, LogLik, Dist, tol);
  
  // Checking for valid dispersion parameter
  if(dispersion <= 0){
    stop("dispersion parameter was estimated to be non-positive");
  }
  
  if(Dist == "gaussian"){
    double temp = Y.n_elem/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
    AIC = -2 * LogLik + 2 * (X.n_cols + 1);
  }
  else if(Dist == "poisson"){
    LogLik -=  LogFact(&Y);
    AIC = -2 * LogLik + 2 * (X.n_cols);
  }else if(Dist == "gamma"){
    double shape = 1 / dispersion;
    LogLik = shape * LogLik + 
      X.n_rows * (shape * log(shape) - lgamma(shape)) + 
      (shape - 1) * arma::accu(log(Y));
    AIC = -2 * LogLik + 2 * (X.n_cols + 1);
  }
  
  SE = sqrt(dispersion) * SE;
  
  // Calculating z-values
  NumericVector z = NumericVector(beta.begin(), beta.end()) / SE;
  
  // Calculating p-values
  NumericVector p = 2 * pnorm(abs(z), 0, 1, false, false);
  
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return List::create(Named("coefficients") = DataFrame::create(Named("Estimate") = beta1,  
                            Named("SE") = sqrt(dispersion) * SE,
                            Named("z") = z, 
                            Named("p-values") = p),
                            Named("iterations") = Iter,
                            Named("dispersion") = dispersion,
                            Named("logLik") =  LogLik,
                            Named("resDev") = resDev,
                            Named("AIC") = AIC,
                            Named("preds") = NumericVector(mu.begin(), mu.end()),
                            Named("linPreds") = linPreds1);
}

// Model fitting function for final model for variable selection functions
List BranchGLMFitCpp(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                     std::string method,  unsigned int m, std::string Link, std::string Dist,
                     unsigned int nthreads, double tol, int maxit, bool GetInit = true){
  
  
  // Initializing vectors and matrices
  arma::vec beta(X->n_cols, arma::fill::zeros);
  arma::mat Info(beta.n_elem, beta.n_elem);
  arma::vec SE1(beta.n_elem);
  // Initializing doubles
  double Iter;
  double dispersion = 1;
  // Getting initial values
  if(GetInit){
    getInit(&beta, X, Y, Offset, Dist, Link, nthreads);
  }
  
  // Fitting model
  if(Dist == "gaussian" && Link == "identity"){
    Iter = LinRegCpp(&beta, X, Y, Offset, &SE1, nthreads);
  }else if(method == "BFGS"){
    Iter = BFGSGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, maxit);
  }
  else if(method == "LBFGS"){
    Iter = LBFGSGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, maxit, m);
  }
  else{
    Iter = FisherScoringGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, maxit);
  }
  
  // Checking for non-invertible fisher info error
  if(Iter == -2){
    stop("Algorithm failed to converge because the fisher info was not invertible");
  }
  
  // Calculating means
  arma::vec mu = LinkCpp(X, &beta, Offset, Link, Dist);
  
  // Calculating variances for betas for non-linear regression
  if(Dist != "gaussian" || Link != "identity"){
    
    // Calculating derivatives, and variances to be used for info
    arma::vec Deriv = DerivativeCpp(X, &beta, Offset, &mu, Link, Dist);
    arma::vec Var = Variance(&mu, Dist);
    
    // Calculating info and initaliazing inverse info
    Info = FisherInfoCpp(X, &Deriv, &Var);
    arma::mat InfoInv = Info;
    
    // Calculating inverse info and returning error if not invertible
    if(!arma::inv_sympd(InfoInv, Info)){
      stop("Fisher info for final model not invertible");
    }
    
    // Calculating variances which are later converted to SEs
    SE1 = arma::diagvec(InfoInv);
  }
  
  NumericVector SE = NumericVector(SE1.begin(), SE1.end());
  
  // Converting variances to SEs
  SE = sqrt(SE);
  
  // Returning results
  double satLogLik = LogLikelihoodSat(X, Y, Dist);
  double LogLik = -LogLikelihoodCpp(X, Y, &mu, Dist);
  double resDev = -2 * (LogLik - satLogLik);
  double AIC = -2 * LogLik + 2 * X->n_cols;
  
  NumericVector beta1 = NumericVector(beta.begin(), beta.end());
  
  arma::vec linPreds = *X * beta + *Offset;
  
  NumericVector linPreds1 = NumericVector(linPreds.begin(), linPreds.end());
  
  // Getting dispersion parameter
  dispersion = GetDispersion(X, Y, &mu, LogLik, Dist, tol);
  
  // Checking for valid dispersion parameter
  if(dispersion <= 0){
    stop("dispersion parameter was estimated to be non-positive");
  }
  
  if(Dist == "gaussian"){
    double temp = Y->n_elem/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
    AIC = -2 * LogLik + 2 * (X->n_cols + 1);
  }
  else if(Dist == "poisson"){
    LogLik -=  LogFact(Y);
    AIC = -2 * LogLik + 2 * (X->n_cols);
  }else if(Dist == "gamma"){
    double shape = 1 / dispersion;
    LogLik = shape * LogLik + 
      X->n_rows * (shape * log(shape) - lgamma(shape)) + 
      (shape - 1) * arma::accu(log(*Y));
    AIC = -2 * LogLik + 2 * (X->n_cols + 1);
  }
  
  SE = sqrt(dispersion) * SE;
  
  // Calculating z-values
  NumericVector z = NumericVector(beta.begin(), beta.end()) / SE;
  
  // Calculating p-values
  NumericVector p = 2 * pnorm(abs(z), 0, 1, false, false);
  
  return List::create(Named("coefficients") = DataFrame::create(Named("Estimate") = beta1,  
                            Named("SE") = sqrt(dispersion) * SE,
                            Named("z") = z, 
                            Named("p-values") = p),
                            Named("iterations") = Iter,
                            Named("dispersion") = dispersion,
                            Named("logLik") =  LogLik,
                            Named("resDev") = resDev,
                            Named("AIC") = AIC,
                            Named("preds") = NumericVector(mu.begin(), mu.end()),
                            Named("linPreds") = linPreds1);
}