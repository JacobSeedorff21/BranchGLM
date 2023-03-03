#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
#include "BranchGLMHelpers.h"
#include "VariableSelection.h"
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

double NullHelper(double beta, const arma::mat* X, const arma::vec* Y, 
                  const arma::vec* Offset, double tol, std::string Link, std::string Dist, 
                  std::string metric){
  // Creating beta
  arma::vec betavec(1);
  betavec.at(0) = beta;
  arma::vec mu = ParLinkCpp(X, &betavec, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(X, Y, &mu, Dist);
  double dispersion = GetDispersion(X, Y, &mu, LogLik, Dist, tol);
  
  if(dispersion <= 0 || std::isnan(LogLik)){
    return(arma::datum::inf);
  }
  
  if(Dist == "gaussian"){
    double temp = X->n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  LogFact(Y);
  }
  else if(Dist == "gamma"){
    double shape = 1 / dispersion;
    LogLik = shape * LogLik + 
      X->n_rows * (shape * log(shape) - lgamma(shape)) + 
      (shape - 1) * arma::accu(log(*Y));
  }
  if(std::isnan(LogLik)){
    return(arma::datum::inf);
  }
  double tempMetric = GetMetric(X, LogLik, Dist, metric);
  
  // Changing best since we include 1 covariate as offset
  if(metric == "AIC"){
    tempMetric-= 2;
  }
  else if(metric == "BIC"){
    tempMetric -= log(X->n_rows);
  }
  else if(metric == "HQIC"){
    tempMetric -= 2 * log(log(X->n_rows));
  }
  return(tempMetric);
  
}

double GetBest(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
               arma::ivec* Indices,
               std::string method, int m, std::string Link, std::string Dist, 
               double tol, int maxit, std::string metric, const arma::imat* Models, unsigned int cur, 
               double beta){
  arma::vec Metrics(Models->n_cols);
  arma::vec tempOffset = *Offset + beta * X->col(as_scalar(arma::find(*Indices == cur)));
  Metrics.fill(arma::datum::inf);
  for(unsigned int i = 0; i < Models->n_cols; i++){
    if(Models->at(cur, i) == 1){
      arma::ivec CurModel = Models->col(i);
      CurModel.at(cur) = 0;
      if(all(CurModel == 0)){
        // If this is the only variable then we don't need to fit anything
        CurModel = Models->col(i);
        arma::mat xTemp = GetMatrix(X, &CurModel, Indices);
        Metrics.at(i) = NullHelper(beta, &xTemp, Y, Offset, tol, Link, Dist, metric);
      }else{
        // Fitting model if there are more than 1 variable
        arma::mat xTemp = GetMatrix(X, &CurModel, Indices);
        Metrics.at(i) = MetricHelper(&xTemp, XTWX, Y, &tempOffset, Indices, &CurModel, 
                   method, m, Link, Dist, tol, maxit, metric);
      }
    }
  }
  return(min(Metrics));
}

double ITPMethod(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                 arma::ivec* Indices,
                 std::string method, int m, std::string Link, std::string Dist, 
                 double tol, int maxit, std::string metric, const arma::imat* Models, unsigned int cur, 
                 double init1, double lowerval, 
                 double init2, double upperval, 
                 double goal){
  
  // Setting initial values
  //// init1 corresponds to the beta corresponding to the optimal model
  //// init2 corresponds to the outer value
  double init3 = init2;
  
  // Initializing variables
  unsigned int iter = 0;
  double MetricVal = lowerval;
  double MetricVal2 = upperval;
  double MetricVal3 = MetricVal2;
  
  // Checking for valid bounds
  if((MetricVal2 - goal) * (MetricVal - goal) >  0 && std::fabs(MetricVal2 - goal) > pow(10, -6)){
    //Rcout << "ITP method not given valid interval for " << cur << std::endl;
    return(arma::datum::inf);
  }
  
  // Swapping inits depending on which one is larger
  if(init2 < init1){
    init2 = init1;
    init1 = init3;
    
    // Swapping metric values
    MetricVal2 = MetricVal;
    MetricVal = MetricVal3;
  }
  
  // Setting hyperparameters
  // These should be reasonable
  double k1 = 0.2 / (init2 - init1);
  double k2 = 2.;
  double n0 = 1.;
  double n12 = floor(log2f((init2 - init1)) - log2f((2 * pow(10, -6))));
  
  // Performing ITP method
  // Finds root of MetricVal - Upper
  while(std::fabs(MetricVal3 - goal) > pow(10, -6) && iter < 100){
    // ITP steps
    //// Interpolation step
    double x12 = (init2 + init1) / 2;
    double xf = (init2 * (MetricVal - goal) - init1 * (MetricVal2 - goal)) / (MetricVal - MetricVal2);
    
    //// Truncation step
    double sigma;
    if(x12 - xf >= 0){
      sigma = 1;
    }
    else{
      sigma = -1;
    }
    double delta = std::min(k1 * pow(std::fabs(init2 - init1), k2), std::fabs(x12 - xf));
    double xt = xf + sigma * delta;
    
    //// Projection step
    double rho = std::min(pow(10, -6) * pow(2, n12 + n0 - iter) - (init2 - init1) / (2), 
                          std::fabs(xt - x12));
    
    init3 = x12 - sigma * rho;
     
    // Fitting new model
    MetricVal3 = GetBest(X, XTWX, Y, Offset, Indices, 
                         method, m, Link, Dist, tol, maxit, metric, Models, cur, init3);
    
    //Rcout << MetricVal3 - goal << ", " << init3  << ", " << MetricVal - goal << ", " << init1 << std::endl;
    
    // Changing inits based on results
    if((MetricVal3 - goal) * (MetricVal2 - goal) > 0){
      init2 = init3;
      MetricVal2 = MetricVal3;
    }
    else if((MetricVal3 - goal) * (MetricVal - goal) > 0){
      init1 = init3;
      MetricVal = MetricVal3;
    }
    else{
      iter++;
      break;
    }
    
    // Incrementing iter
    iter++;
  }
  if(iter >= 100){
    //Rcout << "ITP method failed to converge for " << cur << std::endl;
    //Rcout << MetricVal3 - goal << std::endl;
    return(arma::datum::inf);
  }
  
  return(init3);
}

double SecantMethodCpp(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                       arma::ivec* Indices,
                       std::string method, int m, std::string Link, std::string Dist, 
                       double tol, int maxit, std::string metric, const arma::imat* Models, unsigned int cur, 
                       double bound, double val, double init, double goal, 
                       std::string rootMethod){
  // Creating stuff
  double init1 = bound;
  double init2 = init;
  double  init3 = init1;
  unsigned int iter = 0;
  
  // Fitting model with given initial value
  double MetricVal = val; 
  double MetricVal2;
  double MetricVal3 = val;
  
  while(std::fabs(MetricVal - goal) > pow(10, -6) && iter < 100){
    // Using secant method
    //// Fitting model
    MetricVal2 = MetricVal;
    MetricVal = GetBest(X, XTWX, Y, Offset, Indices, 
                        method, m, Link, Dist, tol, maxit, metric, Models, cur, init2);
    
    // Checking for bounds
    if((MetricVal3 - goal) * (MetricVal - goal) < 0 && rootMethod == "ITP"){
      // Switching to ITP method since we now have valid bounds
      return(ITPMethod(X, XTWX, Y, Offset, Indices, 
                       method, m, Link, Dist, tol, maxit, metric, Models, cur, 
                       init3, MetricVal3, init2, MetricVal, 
                       goal));
    }
    else{
      init3 = init2;
      MetricVal3 = MetricVal;
    }
    
    if(MetricVal2 - MetricVal == 0 || std::isinf(MetricVal)){
      // Return infinity since secant step won't be defined
      // Rcout << "Secant method experienced undefined behavior" << std::endl;
      return(arma::datum::inf);
    }
    
    // Updating beta value
    double tempinit = init2;
    init2 -= (MetricVal - goal) * (init2 - init1) / (MetricVal - MetricVal2);
    init1 = tempinit;
    
    // Incrementing iter
    iter++;
  }
  if(iter >= 100){
    // Secant method failed to converge
    return(arma::datum::inf);
  }
  
  return(init1);
}

// Metric Interval
// [[Rcpp::export]]
List MetricIntervalCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                                 IntegerVector indices, IntegerVector num,
                                 IntegerMatrix models,
                                 std::string method, int m,
                                 std::string Link, std::string Dist,
                                 unsigned int nthreads, double tol, int maxit, 
                                 std::string metric, NumericVector mle, NumericVector se,
                                 NumericVector best, double goal, 
                                 std::string rootMethod){
  
  // Creating necessary vectors/matrices
  const arma::imat Models(models.begin(), models.rows(), models.cols(), false, true);
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec MLE(mle.begin(), mle.size(), false, true);
  const arma::vec SE(se.begin(), se.size(), false, true);
  const arma::vec Best(best.begin(), best.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec Counts(num.begin(), num.size(), false, true);
  
  // Setting number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
  // Getting metrics
  arma::vec UpperVals(Models.n_rows);
  UpperVals.fill(arma::datum::inf);
  arma::vec LowerVals(Models.n_rows);
  LowerVals.fill(-arma::datum::inf);
  
  // Changing best since we include 1 covariate as offset
  if(metric == "AIC"){
    goal = goal - 2;
    best = best - 2;
  }
  else if(metric == "BIC"){
    goal = goal - log(X.n_rows);
    best = best - log(X.n_rows);
  }
  else if(metric == "HQIC"){
    goal = goal - 2 * log(log(X.n_rows));
    best = best - 2 * log(log(X.n_rows));
  }
  
  for(unsigned int j = 0; j < Models.n_rows; j += 2 * nthreads){
    unsigned int maxval = std::min(j + 2 * nthreads, Models.n_rows);
#pragma omp parallel for
    for(unsigned int i = j; i < maxval; i++){
      if(all(Models.row(i) != 1) || Counts.at(i) > 1){
        // Do nothing
      }else{
        unsigned int cur = as_scalar(arma::find(Indices == i));
        double curMLE = MLE.at(cur);
        UpperVals.at(i) = SecantMethodCpp(&X, &XTWX, &Y, &Offset, &Indices, 
                   method, m, Link, Dist, tol, maxit, metric, &Models, i, 
                   curMLE, Best.at(i), curMLE + SE.at(i), goal, rootMethod);
        LowerVals.at(i) = SecantMethodCpp(&X, &XTWX, &Y, &Offset, &Indices, 
                     method, m, Link, Dist, tol, maxit, metric, &Models, i, 
                     curMLE, Best.at(i), curMLE - SE.at(i), goal, rootMethod);
        
        // Checking to make sure they are on the correct side
        if(UpperVals.at(i) < curMLE){
          UpperVals.at(i) = arma::datum::inf;
        }
        if(LowerVals.at(i) > curMLE){
          LowerVals.at(i) = arma::datum::inf;
        }
      }
    }
    checkUserInterrupt();
  }
  
  // Setting number of threads to 1 if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  List FinalList = List::create(Named("LowerBounds") = LowerVals, 
                                Named("UpperBounds") = UpperVals);
  return(FinalList);
}