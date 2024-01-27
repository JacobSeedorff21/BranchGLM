#include <RcppArmadillo.h>
#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
#include "BranchGLMHelpers.h"
#include "VariableSelection.h"
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

arma::vec GetY(const arma::mat* y, std::string Link){
  arma::vec NewY = *y;
  if(Link == "log"){
    NewY = log(NewY.replace(0, 1e-4));
    
  }else if(Link == "inverse"){
    NewY = 1 / (NewY.replace(0, 1e-4));
    
  }else if(Link == "sqrt"){
    NewY = sqrt(NewY);
    
  }else if(Link == "logit"){
    NewY = NewY.clamp(1e-4, 1 - 1e-4);
    NewY = log(NewY / (1 - NewY));
    
  }else if(Link == "probit"){ 
    double val0 = boost::math::quantile(boost::math::normal(0.0, 1.0), 1e-4);
    double val1 = boost::math::quantile(boost::math::normal(0.0, 1.0), 1 - 1e-4);
    for(unsigned int i = 0; i < NewY.n_elem; i++){
      if(NewY.at(i) == 0){
        NewY.at(i) = val0;
      }else{ 
        NewY.at(i) = val1;
      }
    } 
    
  }else if(Link == "cloglog"){
    NewY = NewY.clamp(1e-4, 1 - 1e-4);
    NewY = log(-log(1 - NewY));
  }
  return(NewY);
}

bool GetXTXXT(const arma::mat* X, const arma::mat* XTWX, arma::mat* res){
  return(arma::solve(*res, *XTWX, X->t(), arma::solve_opts::no_approx + arma::solve_opts::likely_sympd));
}

// Function used to fit models and calculate desired metric
double MetricHelperWithBetas(const arma::mat* oldX, const arma::mat* XTWX, 
                    const arma::vec* Y, const arma::vec* Offset,
                    const arma::ivec* Indices, const arma::ivec* CurModel,
                    std::string method, 
                    int m, std::string Link, std::string Dist,
                    double tol, int maxit, const arma::vec* pen,
                    arma::vec* betas, arma::vec* SEs){
  
  // If x has more columns than rows, then we cannot fit any model
  if(oldX->n_cols > oldX->n_rows){
    return(arma::datum::inf);
  }
  
  // Getting submatrix of XTWX
  unsigned count = 0;
  for(unsigned int i = 0; i < Indices->n_elem; i++){
    if(CurModel->at(Indices->at(i)) != 0){
      count++;
    }
  } 
  arma::uvec NewInd(count);
  count = 0;
  for(unsigned int i = 0; i < Indices->n_elem; i++){
    if(CurModel->at(Indices->at(i)) != 0){
      NewInd.at(count++) = i;
    }
  } 
  
  arma::mat NewXTWX = XTWX->submat(NewInd, NewInd);
  arma::mat X = oldX->cols(NewInd);
  bool UseXTWX = true;
  arma::vec beta(X.n_cols, arma::fill::zeros);
  
  // Getting initial values
  PargetInit(&beta, &X, &NewXTWX, Y, Offset, Dist, Link, &UseXTWX);
  
  int Iter;
  
  if(Dist == "gaussian" && Link == "identity"){
    Iter = ParLinRegCppShort(&beta, &X, &NewXTWX, Y, Offset);
  }else if(method == "BFGS"){ 
    Iter = ParBFGSGLMCpp(&beta, &X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  } 
  else if(method == "LBFGS"){
    Iter = ParLBFGSGLMCpp(&beta, &X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, m, UseXTWX);
  } 
  else{
    Iter = ParFisherScoringGLMCpp(&beta, &X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  } 
  
  if(Iter <= 0){
    return(arma::datum::inf);
  } 
  
  arma::vec mu = ParLinkCpp(&X, &beta, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(&X, Y, &mu, Dist);
  double dispersion = GetDispersion(&X, Y, &mu, LogLik, Dist, tol);
  if(dispersion < 0 || std::isnan(LogLik) || std::isinf(dispersion)){
    return(arma::datum::inf);
  } 
  
  if(Dist == "gaussian"){
    double temp = X.n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  } 
  else if(Dist == "poisson"){
    LogLik -=  LogFact(Y);
  } 
  else if(Dist == "gamma"){
    double shape = 1 / dispersion;
    LogLik = shape * LogLik + 
      X.n_rows * (shape * log(shape) - lgamma(shape)) +
      (shape - 1) * arma::accu(log(*Y));
  } 
  if(std::isnan(LogLik)){
    return(arma::datum::inf);
  } 
  
  // Calculate SEs
  // Calculating derivatives, and variances to be used for info
  arma::vec Deriv = ParDerivativeCpp(&X, &beta, Offset, &mu, Link, Dist);
  arma::vec Var = ParVariance(&mu, Dist);
  
  // Calculating info and initalizing inverse info
  arma::mat Info = ParFisherInfoCpp(&X, &Deriv, &Var);
  arma::mat InfoInv = Info;
  
  // Calculating inverse info and returning error if not invertible
  if(arma::solve(InfoInv, Info, arma::eye(arma::size(InfoInv)), arma::solve_opts::no_approx + arma::solve_opts::likely_sympd)){
    // Calculating SE
    arma::vec SE = sqrt(arma::diagvec(InfoInv) * dispersion);
    SEs->elem(NewInd) = SE;
  }
  // Getting betas
  betas->elem(NewInd) = beta;
  
  return(-2 * LogLik + arma::accu(pen->elem(find(*CurModel != 0))));
} 

// Function used to fit models and calculate desired metric
double MetricHelper2(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y,
                     const arma::mat* XTXXT, const arma::vec* NewY,
                             const arma::vec* Offset,
                             const arma::ivec* Indices, const arma::ivec* CurModel,
                             std::string method, 
                             int m, std::string Link, std::string Dist,
                             double tol, int maxit, const arma::vec* pen){
  
  
  // Getting beta
  arma::vec beta = *XTXXT * (*NewY - *Offset);
  int Iter;
  bool UseXTWX = false;
  if(Dist == "gaussian" && Link == "identity"){
    // Do nothing
    Iter = 1;
  }else if(method == "BFGS"){  
    Iter = ParBFGSGLMCpp(&beta, X, XTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }  
  else if(method == "LBFGS"){
    Iter = ParLBFGSGLMCpp(&beta, X, XTWX, Y, Offset, Link, Dist, tol, maxit, m, UseXTWX);
  }  
  else{
    Iter = ParFisherScoringGLMCpp(&beta, X, XTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }  
  
  if(Iter <= 0){
    return(arma::datum::inf);
  }  
  
  arma::vec mu = ParLinkCpp(X, &beta, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(X, Y, &mu, Dist);
  double dispersion = GetDispersion(X, Y, &mu, LogLik, Dist, tol);
  if(dispersion <= 0 || std::isnan(LogLik) || std::isinf(dispersion)){
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
  return(-2 * LogLik + arma::accu(pen->elem(find(*CurModel != 0))));
}  

double NullHelper(double beta, const arma::mat* X, const arma::vec* Y, 
                  const arma::vec* Offset, double tol, std::string Link, std::string Dist, 
                  const arma::vec* pen){
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
  
  return(-2 * LogLik);
}

double GetBest(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, 
               const arma::mat* XTXXT, const arma::vec* NewY, const arma::vec* curCol,
               const arma::vec* Offset,
               arma::ivec* Indices,
               std::string method, int m, std::string Link, std::string Dist, 
               double tol, int maxit, const arma::vec* pen, const arma::ivec* CurModel, 
               unsigned int cur, 
               double beta, double goal, const double Metric){
  double curMetric = arma::datum::inf;
  if(Metric <= goal){
    if(all(*CurModel == 0)){
      // If this is the only variable then we don't need to fit anything
      curMetric = NullHelper(beta, curCol, Y, Offset, tol, Link, Dist, pen);
    }else{
      // Fitting model if there are more than 1 variable
      arma::vec tempOffset = *Offset + beta * *curCol;
      curMetric = MetricHelper2(X, XTWX, Y, XTXXT, NewY, &tempOffset, Indices, CurModel, 
                 method, m, Link, Dist, tol, maxit, pen);
    }
  }
  return(curMetric);
}

double ITPMethod(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, 
                 const arma::mat* XTXXT, const arma::vec* NewY, const arma::vec* curCol,
                 const arma::vec* Offset,
                 arma::ivec* Indices,
                 std::string method, int m, std::string Link, std::string Dist, 
                 double tol, int maxit, const arma::vec* pen, const arma::ivec* CurModel, 
                 unsigned int cur, 
                 double init1, double lowerval, 
                 double init2, double upperval, 
                 double goal, const double Metric){
  
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
    MetricVal3 = GetBest(X, XTWX, Y, XTXXT, NewY, curCol, Offset, Indices, 
                         method, m, Link, Dist, tol, maxit, pen, CurModel, cur, 
                         init3, goal, Metric);
 
    
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
    return(arma::datum::inf);
  }
  return(init3);
}

double SecantMethodCpp(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, 
                       const arma::mat* XTXXT, const arma::vec* NewY, const arma::vec* curCol,
                       const arma::vec* Offset,
                       arma::ivec* Indices,
                       std::string method, int m, std::string Link, std::string Dist, 
                       double tol, int maxit, const arma::vec* pen, const arma::ivec* CurModel, unsigned int cur, 
                       double bound, double val, double init, double goal, 
                       const double Metric,
                       std::string rootMethod, std::string direction){
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
    MetricVal = GetBest(X, XTWX, Y, XTXXT, NewY, curCol, Offset, Indices, 
                        method, m, Link, Dist, tol, maxit, pen, CurModel, cur, 
                        init2, goal, Metric);
    
    //// Going backwards if we have gone too far and metric value is infinite
    unsigned int newIter = 0;
    while(std::isinf(MetricVal) && newIter < 10){
      init2 = (init2 + init3) / 2;
      MetricVal = GetBest(X, XTWX, Y, XTXXT, NewY, curCol, Offset, Indices, 
                          method, m, Link, Dist, tol, maxit, pen, CurModel, cur, 
                          init2, goal, Metric);
      newIter++;
    }
    if(std::isinf(MetricVal)){
      // Return infinity since secant step won't be defined
      return(arma::datum::inf);
    }
    
    // Checking for bounds
    if((MetricVal3 - goal) * (MetricVal - goal) < 0 && rootMethod == "ITP"){
      // Switching to ITP method since we now have valid bounds
      return(ITPMethod(X, XTWX, Y, XTXXT, NewY, curCol, Offset, Indices, 
                       method, m, Link, Dist, tol, maxit, pen, CurModel, cur, 
                       init3, MetricVal3, init2, MetricVal, goal, Metric));
    }
    else{
      init3 = init2;
      MetricVal3 = MetricVal;
    }
    
    // Updating beta value
    double tempinit = init2;
    init2 -= (MetricVal - goal) * (init2 - init1) / (MetricVal - MetricVal2);
    init1 = tempinit;
    
    // Making sure that init2 is in the right direction
    if(direction == "upper" && init2 < init3){
      init2 = 2 * init3 - init2;
    }
    else if(direction == "lower" && init2 > init3){
      init2 = 2 * init3 - init2;
    }
    
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
                                 IntegerVector model,
                                 std::string method, int m,
                                 std::string Link, std::string Dist,
                                 unsigned int nthreads, double tol, int maxit, 
                                 NumericVector pen, NumericVector mle, NumericVector se,
                                 NumericVector best, double cutoff, double Metric,
                                 std::string rootMethod){
  
  // Creating necessary vectors/matrices
  arma::ivec CurModel2(model.begin(), model.size(), false, true);
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::vec MLE(mle.begin(), mle.size(), false, true);
  const arma::vec SE(se.begin(), se.size(), false, true);
  arma::vec Best(best.begin(), best.size(), true, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec Counts(num.begin(), num.size(), false, true);
  
  // Setting number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
  // Getting metrics
  arma::vec UpperVals(CurModel2.n_elem);
  UpperVals.fill(arma::datum::inf);
  arma::vec LowerVals(CurModel2.n_elem);
  LowerVals.fill(-arma::datum::inf);
  
  // Changing best since we include 1 covariate as offset
  Best -= min(Pen);
  Metric = Metric - min(Pen);
  
  for(unsigned int j = 0; j < CurModel2.n_elem; j += 2 * nthreads){
    unsigned int maxval = std::min(j + 2 * nthreads, CurModel2.n_elem);
#pragma omp parallel for schedule(dynamic)
    for(unsigned int i = j; i < maxval; i++){
      arma::ivec CurModel = CurModel2;
      if(Counts(i) > 1 || CurModel(i) == -1){
        // Do nothing
      }
      else if(CurModel(i) == 0){
        // Set these intervals to be 0
        UpperVals.at(i) = 0;
        LowerVals.at(i) = 0;
      }
      else{
        unsigned int cur = as_scalar(arma::find(Indices == i));
        double curMLE = MLE.at(cur);
        double curSE = SE.at(cur);
        
        // Getting X and XTWX for this model
        CurModel(i) = 0;
        
        // Getting submatrix of XTWX
        unsigned count = 0;
        for(unsigned int i = 0; i < Indices.n_elem; i++){
          if(CurModel(Indices(i)) != 0){
            count++;
          }
        } 
        arma::uvec NewInd(count);
        count = 0;
        for(unsigned int i = 0; i < Indices.n_elem; i++){
          if(CurModel(Indices(i)) != 0){
            NewInd(count++) = i;
          }
        }  
        arma::mat NewXTWX = XTWX.submat(NewInd, NewInd);
        arma::mat NewX = X.cols(NewInd);
        arma::mat XTXXT;
        arma::vec NewY = GetY(&Y, Link);
        bool check = GetXTXXT(&NewX, &NewXTWX, &XTXXT);
        arma::vec curCol = X.col(cur);
        if(!check){
          // Do nothing
        }else{
          UpperVals.at(i) = SecantMethodCpp(&NewX, &NewXTWX, &Y, 
                       &XTXXT, &NewY, &curCol, 
                       &Offset, &Indices, 
                     method, m, Link, Dist, tol, maxit, &Pen, &CurModel, i, 
                     curMLE, Best.at(i), curMLE + curSE, 
                     Best.at(i) + cutoff, Metric, rootMethod, "upper");
          LowerVals.at(i) = SecantMethodCpp(&NewX, &NewXTWX, &Y, 
                       &XTXXT, &NewY, &curCol, &Offset, &Indices, 
                       method, m, Link, Dist, tol, maxit, &Pen, &CurModel, i, 
                       curMLE, Best.at(i), curMLE - curSE, 
                       Best.at(i) + cutoff, Metric, rootMethod, "lower");
          
          // Checking to make sure they are on the correct side
          if(UpperVals.at(i) < curMLE){
             UpperVals.at(i) = arma::datum::inf;
          }
          if(LowerVals.at(i) > curMLE){
             LowerVals.at(i) = -arma::datum::inf;
          }
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
