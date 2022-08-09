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

// Fits upper model for a set of models and calculates the bound for the desired metric
double GetBound(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                         std::string method, int m, std::string Link, std::string Dist,
                         arma::ivec* CurModel,  arma::ivec* indices, 
                         double tol, int maxit,
                         std::string metric, unsigned int cur, int minsize,
                         arma::uvec* NewOrder, double LowerBound, double CurMetric,
                         arma::vec* Metrics){
  
  
  // Checking if we need to fit model for upper bound and updating bounds if we don't need to
  if(cur == 1){
    double MetricVal = 0;
    if(metric == "AIC"){
      MetricVal = 2;
    }
    else if (metric == "BIC"){
      MetricVal = log(X->n_rows);
    }
    if(CurMetric - LowerBound > MetricVal){
      return(UpdateBound(X, indices, NewOrder->at(cur - 1), LowerBound, metric, minsize));
    }
    else{
      return(arma::datum::inf);
    }
  }
  
  // Creating vector for the upper model
  arma::ivec UpperModel = *CurModel;
  for(unsigned int i = cur; i < NewOrder->n_elem; i++){
    UpperModel.at(NewOrder->at(i)) = 1;
  }
  
  // Getting submatrix of XTWX
  unsigned count = 0;
  for(unsigned int i = 0; i < indices->n_elem; i++){
    if(UpperModel.at(indices->at(i)) != 0){
      count++;
    }
  }
  
  arma::uvec NewInd(count);
  count = 0;
  for(unsigned int i = 0; i < indices->n_elem; i++){
    if(UpperModel.at(indices->at(i)) != 0){
      NewInd.at(count++) = i;
    }
  }
  
  // Getting new XTWX matrix
  arma::mat NewXTWX = XTWX->submat(NewInd, NewInd);
  bool UseXTWX = true;
  
  
  // Defining Iter
  int Iter;
  
  // Creating matrix for upper model and fitting it
  arma::mat xTemp = GetMatrix(X, &UpperModel, indices);
  arma::vec beta(xTemp.n_cols, arma::fill::zeros);
  PargetInit(&beta, &xTemp, &NewXTWX, Y, Offset, Dist, Link, &UseXTWX);
  
  // Fitting model
  if(Dist == "gaussian" && Link == "identity"){
    Iter = ParLinRegCppShort(&beta, &xTemp, &NewXTWX, Y, Offset);
  }else if(method == "BFGS"){
    Iter = ParBFGSGLMCpp(&beta, &xTemp, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }
  else if(method == "LBFGS"){
    Iter = ParLBFGSGLMCpp(&beta, &xTemp, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, m, UseXTWX);
  }
  else{
    Iter = ParFisherScoringGLMCpp(&beta, &xTemp, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }
  
  // Checking for non-invertible fisher info
  if(Iter == -2){
    return(LowerBound);
  }
  
  // Calculating metric value
  arma::vec mu = ParLinkCpp(&xTemp, &beta, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(&xTemp, Y, &mu, Dist);
  double dispersion = GetDispersion(&xTemp, Y, &mu, LogLik, Dist, tol);
  
  // Checking for non-positive dispersion
  if(dispersion <= 0){
    return(LowerBound);
  }
  
  // Final computation of log-likelihood
  if(Dist == "gaussian"){
    double temp = xTemp.n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  LogFact(Y);
  }
  else if(Dist == "gamma"){
    double shape = 1 / dispersion;
    LogLik = shape * LogLik + 
      xTemp.n_rows * (shape * log(shape) - lgamma(shape)) + 
      (shape - 1) * arma::accu(log(*Y));
  }
  
  // Returning previoys lower bound if log likelihood is nan
  if(std::isnan(LogLik)){
    return(LowerBound);
  }
  
  // Updating metric value
  if(cur > 0){
    Metrics->at(cur - 1) = GetMetric(&xTemp, LogLik, Dist, metric);
  }
  else{
    Metrics->at(0) = GetMetric(&xTemp, LogLik, Dist, metric);
  } 
  
  // Checking for failed convergence 
  if(Iter == -1){
    return(LowerBound);
  }
  
  // Getting bound if model converged
  double NewBound = BoundHelper(X, LogLik, Dist, metric, minsize);
  
  // Tightening bounds if possible
  double MetricVal = 0;
  if(metric == "AIC"){
    MetricVal = 2;
  }
  else if (metric == "BIC"){
    MetricVal = log(X->n_rows);
  }
  if(CurMetric - LowerBound > MetricVal){
    NewBound +=  MetricVal;
  }
  else{
    NewBound = arma::datum::inf;
  }
  
  return(NewBound);
}

// Function used to performing branching for branch and bound method
void Branch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
            std::string method, int m, std::string Link, std::string Dist,
            arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
            unsigned int* numchecked, arma::ivec* indices, double tol, 
            int maxit, 
            int maxsize, unsigned int cur, std::string metric, 
            double LowerBound, arma::uvec* NewOrder, Progress* p){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < *BestMetric && maxsize > 0){
    
    // Updating progress
    p->update(2);
    p->print();
    
    // Creating vectors to be used later on
    arma::uvec NewOrder2(NewOrder->n_elem - cur);
    arma::vec Metrics(NewOrder->n_elem - cur);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder->at(j + cur)) = 1;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      NewOrder2.at(j) = NewOrder->at(j + cur);
      Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, 
                 method, m, Link, Dist, 
                 tol, maxit, metric);
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += NewOrder2.n_elem;
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    if(Metrics.at(0) < *BestMetric){
      *BestMetric = Metrics.at(0);
      *BestModel = *CurModel;
      BestModel->at(NewOrder2.at(0)) = 1;
    }
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Only find bounds and perform branching if there is at least 1 element to branch on
    // and maxsize is greater than 1
    if(NewOrder2.n_elem > 1 && maxsize > 1){
      
      // Creating vector to store lower bounds
      arma::vec Bounds(NewOrder2.n_elem - 1, arma::fill::zeros);
      
      // Getting lower bounds
  #pragma omp parallel for schedule(dynamic)
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        CurModel2.at(NewOrder2.at(j)) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Bounds.at(j) = GetBound(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2,
                     indices, tol, maxit, metric, 
                     j + 1, xTemp.n_cols, &NewOrder2, LowerBound, 
                     Metrics.at(j), &Metrics);
      }
        
      // Updating numchecked
      (*numchecked) += NewOrder2.n_elem - 2;
      
      // Checking for user interrupt
      checkUserInterrupt();
      
      // Recursively calling this function for each new model
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        CurModel2.at(NewOrder2.at(j)) = 1;
        Branch(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
               BestMetric, numchecked, indices, tol, maxit, maxsize - 1, j + 1, metric, 
               Bounds.at(j), &NewOrder2, p);
      }
    }
  }
  else{
    // Updating progress since we have cut off part of the tree
    p->update(GetNum(NewOrder->n_elem - cur, maxsize));
    p->print();
  }
}


// Branch and bound method
// [[Rcpp::export]]
List BranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                       IntegerVector indices, IntegerVector num,
                       std::string method, int m,
                       std::string Link, std::string Dist,
                       unsigned int nthreads, double tol, int maxit, 
                       IntegerVector keep, int maxsize, std::string metric,
                       bool display_progress, double BestMetric){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  // Making sure that no variables are including in curmodel, except those kept in each model
  CurModel.replace(1, 0);
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  
  
  // Getting X'WX
  arma::mat XTWX;
  if(Dist == "gaussian" || Dist == "gamma"){
    XTWX = X.t() * X;
  }else{
    arma::vec beta(X.n_cols, arma::fill::zeros);
    arma::vec mu = LinkCpp(&X, &beta, &Offset, Link, Dist);
    arma::vec Deriv = DerivativeCpp(&X, &beta, &Offset, &mu, Link, Dist);
    arma::vec Var = Variance(&mu, Dist);
    XTWX = FisherInfoCpp(&X, &Deriv, &Var);
  }
  
  // Creating necessary scalars
  unsigned int numchecked = 1;
  unsigned int size = 0;
  
  
  // Setting number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Getting size of model space to check
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      size++;
    }
  }
  
  // Creating object to report progress
  Progress p(GetNum(size, maxsize), display_progress);
  p.print();
  
  arma::uvec NewOrder(size);
  unsigned int k = 0;
  
  // Making vector to store the order of the variables
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      NewOrder.at(k++) = j;
    }
  }
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Fitting initial model
  double CurMetric = MetricHelper(&xTemp, &XTWX, &Y, &Offset, &Indices, 
                                     &CurModel, method, m, Link, Dist, 
                                     tol, maxit, metric);
  
  // Updating BestMetric is CurMetric is better
  if(CurMetric < BestMetric){
    BestMetric = CurMetric;
    BestModel = CurModel;
  }
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  arma::vec Metrics(1);
  Metrics.at(0) = arma::datum::inf;
  LowerBound = GetBound(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel,
                        &Indices, tol, maxit, metric, 
                        0, sum(abs(CurModel)), &NewOrder, LowerBound, 
                        CurMetric, &Metrics);
  
  // Incrementing numchecked
  numchecked++;
  
  // Starting branching process
  Branch(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
            &BestMetric, &numchecked, &Indices, tol, maxit, maxsize, 0, metric, 
            LowerBound, &NewOrder, &p);
  
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Printing off final update
  p.finalprint();
  
  // Getting x matrix for best model found
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  // Fitting best model
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                                 nthreads, tol, maxit);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  // Resetting number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// When doing the process backwards the upper model is already fit, so we just 
// need to use that and minimum number of variables to get bound
double BackwardGetBound(const arma::mat* X, arma::ivec* indices, arma::ivec* CurModel,
                        arma::uvec* NewOrder, unsigned int cur, double metricVal, 
                        std::string metric, unsigned int maxsize){
  
  // If metricVal is infinite, then return -inf as the lower bound
  if(metricVal == arma::datum::inf){
    return(-metricVal);
  }
  
  // Getting the minimum size of this set of regressions
  arma::ivec CurModel2 = *CurModel;
  for(unsigned int i = 0; i <= cur; i++){
    CurModel2.at(NewOrder->at(i)) = 2;
  }
  
  unsigned int minsize = maxsize;
  double value = 0;
  for(unsigned int i = 0; i < indices->n_elem; i++){
    if(CurModel2.at(indices->at(i)) == 2){
      minsize--;
    }
  }
  
  // Calculating lower bound from metricVal and maxsize/minsize
  if(metric == "AIC"){
    value =  metricVal  - 2 * int (maxsize - minsize);
  }else if(metric == "AICc"){
    // Need to fix this method, but it's not currently used anyways
    int newk = -(maxsize - minsize);
    value = metricVal + 2 * maxsize - (2 * newk + 2 * pow(newk, 2)) / (X->n_rows - newk - 1) + 
      (2 * minsize + 2 * pow(minsize, 2)) / (X->n_rows - minsize - 1);
  }else if(metric == "BIC"){
    value = metricVal - log(X->n_rows) * int (maxsize - minsize);
  }
  
  return(value);
}

// Function used to performing branching for backward branch and bound method
void BackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                    std::string method, int m, std::string Link, std::string Dist,
                    arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
                    unsigned int* numchecked, arma::ivec* indices, double tol, 
                    int maxit, unsigned int cur, std::string metric, 
                    double LowerBound, arma::uvec* NewOrder, Progress* p){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < *BestMetric){
    // Updating progress
    p->update(2);
    p->print();
    
    // Creating vectors to be used later
    arma::uvec NewOrder2(cur + 1);
    arma::vec Metrics(cur + 1);
    
    //Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder->at(j)) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      NewOrder2.at(j) = NewOrder->at(j);
      Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
                                        method, m, Link, Dist, 
                                        tol, maxit, metric);
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += NewOrder2.n_elem;
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    if(Metrics.at(0) < *BestMetric){
      *BestMetric = Metrics.at(0);
      *BestModel = *CurModel;
      BestModel->at(NewOrder2.at(0)) = 0;
    }
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Getting lower bounds which are now stored in Metrics
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      Metrics.at(j) = BackwardGetBound(X, indices, &CurModel2, &NewOrder2, 
                 j - 1, Metrics.at(j), metric, 
                 xTemp.n_cols);
    }
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Recursively calling this function for each new model
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 0;
      BackwardBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
                     BestMetric, numchecked, indices, tol, maxit, j - 1, metric, 
                     Metrics.at(j), &NewOrder2, p);
    }
  }
  else{
    // Updating progress since we have cut off part of the tree
    p->update(GetNum(cur + 1, cur + 1));
    p->print();
  }
}


// Backward Branch and bound method
// [[Rcpp::export]]
List BackwardBranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                               IntegerVector indices, IntegerVector num,
                               std::string method, int m,
                               std::string Link, std::string Dist,
                               unsigned int nthreads, double tol, int maxit, 
                               IntegerVector keep, std::string metric,
                               bool display_progress, double BestMetric){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  
  // Making sure that CurModel includes all variables 
  CurModel.replace(0, 1);
  
  // Getting X'WX
  arma::mat XTWX;
  if(Dist == "gaussian" || Dist == "gamma"){
    XTWX = X.t() * X;
  }else{
    arma::vec beta(X.n_cols, arma::fill::zeros);
    arma::vec mu = LinkCpp(&X, &beta, &Offset, Link, Dist);
    arma::vec Deriv = DerivativeCpp(&X, &beta, &Offset, &mu, Link, Dist);
    arma::vec Var = Variance(&mu, Dist);
    XTWX = FisherInfoCpp(&X, &Deriv, &Var);
  }
  
  // Setting number of threads if OpenMP is defined
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  
  // Getting size of model space to check
  unsigned int size = 0;
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 1){
      size++;
    }
  }
  
  // Creating object to report progress
  Progress p(GetNum(size, size), display_progress);
  p.print();
  
  
  unsigned int k = 0;
  
  // Making vector of order to look at variables
  arma::uvec NewOrder(size);
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 1){
      NewOrder.at(k++) = j;
    }
  }
  
  // Fitting model with all variables included
  double CurMetric = MetricHelper(&X, &XTWX, &Y, &Offset, &Indices, &CurModel,
                                     method, m, Link, Dist, 
                                     tol, maxit, metric);
  
  // Updating BestMetric and BestModel if CurMetric is better than BestMetric
  if(CurMetric < BestMetric){
    BestMetric = CurMetric;
    BestModel = CurModel;
  }
  
  // Creating numchecked to keep track of the number of models fit
  unsigned int numchecked = 1;
  
  // Getting lower bound for all models
  double LowerBound = BackwardGetBound(&X, &Indices, &CurModel, &NewOrder, 
                                          NewOrder.n_elem - 1, CurMetric, metric, 
                                          X.n_cols);
  
  // Starting the branching process
  BackwardBranch(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
                    &BestMetric, &numchecked, &Indices, tol, maxit, NewOrder.n_elem - 1, metric, 
                    LowerBound, &NewOrder, &p);
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Printing off final update
  p.finalprint();
  
  // Getting x matrix for best model found
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  // Fitting best model
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                                 nthreads, tol, maxit);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  // Resetting number of threads if OpenMP is defined
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// Defining backward branching function for switch method
// Forward declaration so this can be called by the forward switch branch
void SwitchBackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                             std::string method, int m, std::string Link, std::string Dist,
                             arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
                             unsigned int* numchecked, arma::ivec* indices, double tol, 
                             int maxit, unsigned int cur, std::string metric, 
                             double LowerBound, arma::uvec* NewOrder, Progress* p, 
                             double LowerMetric);

// Function used to performing branching for forward part of switch branch
void SwitchForwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
               std::string method, int m, std::string Link, std::string Dist,
               arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
               unsigned int* numchecked, arma::ivec* indices, double tol, 
               int maxit, unsigned int cur, std::string metric, 
               double LowerBound, arma::uvec* NewOrder, Progress* p, double UpperMetric){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < *BestMetric){
    // Updating progress
    p->update(2);
    p->print();
    
    // Creating vectors to be used later
    arma::uvec NewOrder2(NewOrder->n_elem - cur);
    arma::vec Metrics(NewOrder->n_elem - cur);
    
     
     
    // Getting metric values
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder->at(j + cur)) = 1;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      NewOrder2(j) = NewOrder->at(j + cur);
      Metrics(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, 
                 method, m, Link, Dist, 
                 tol, maxit, metric);
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += NewOrder2.n_elem;
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    if(Metrics(0) < *BestMetric){
      *BestMetric = Metrics(0);
      *BestModel = *CurModel;
      BestModel->at(NewOrder2(0)) = 1;
    }
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Only need to calculate bounds and branch if NewOrder2.n_elem > 1
    if(NewOrder2.n_elem > 1){
      
      // Creating vectors using in finding the bounds
      /// Bounds stores the bounds
      /// Metrics2 stores the metric values from the upper models
      arma::vec Bounds(NewOrder2.n_elem - 1);
      arma::vec Metrics2(NewOrder2.n_elem - 1);
      Metrics2.fill(arma::datum::inf);
      Metrics2.at(0) = UpperMetric;
      
      // Finding lower bounds
#pragma omp parallel for schedule(dynamic)
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        CurModel2(NewOrder2(j)) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Bounds(j) = GetBound(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2,
                   indices, tol, maxit, metric, 
                   j + 1, xTemp.n_cols, &NewOrder2, LowerBound, Metrics.at(j), &Metrics2);
      }
      
      // Updating numchecked and potentially updating the best model based on upper models
      (*numchecked) += NewOrder2.n_elem - 2;
      sorted = sort_index(Metrics2);
      if(Metrics2(sorted(0)) < *BestMetric){
        *BestMetric = Metrics2(sorted(0));
        *BestModel = *CurModel;
        for(unsigned int i = sorted(0); i < NewOrder2.n_elem; i++){
          BestModel->at(NewOrder2(i)) = 1;
        }
      }
      
      // Checking for user interrupt
      checkUserInterrupt();
      
      // Defining upper model to be used for possible switch to backward branching
      /// Reversing new order for possible switch to backward branching
      arma::uvec revNewOrder2 = reverse(NewOrder2);
      arma::ivec UpperModel = *CurModel;
      for(unsigned int i = 0; i < revNewOrder2.n_elem; i++){
        UpperModel(revNewOrder2(i)) = 1;
      }
      
      // Reversing vectors for possible switch to backward branching
      Bounds = reverse(Bounds);
      Metrics2 = reverse(Metrics2);
      Metrics= reverse(Metrics);
      
      // Recursively calling this function for each new model
      for(unsigned int j = NewOrder2.n_elem - 1; j > 0; j--){
        if(j < revNewOrder2.n_elem - 1){
          UpperModel(revNewOrder2(j + 1)) = 0;
        }
        
        if(Metrics.at(j) > Metrics2.at(j - 1)){
          // If upper model is better than lower model then call backward
        SwitchBackwardBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &UpperModel, BestModel, 
                                BestMetric, numchecked, indices, tol, maxit, j - 1, metric, 
                                Bounds.at(j - 1), &revNewOrder2, p, Metrics.at(j));
        }else{
          // Creating new current model for next call to forward branch
          arma::ivec CurModel2 = *CurModel;
          CurModel2(revNewOrder2(j)) = 1;
          
          // If lower model is better than upper model then call forward
          SwitchForwardBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
                                  BestMetric, numchecked, indices, tol, maxit, NewOrder2.n_elem - j, metric, 
                                  Bounds.at(j - 1), &NewOrder2, p, Metrics2.at(j - 1));
        }
      }
    }
  }
  else{
    // Updating progress since we have cut off part of the tree
    p->update(GetNum(NewOrder->n_elem - cur, NewOrder->n_elem - cur));
    p->print();
  }
}


// Function used to performing branching for branch and bound method
void SwitchBackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                       std::string method, int m, std::string Link, std::string Dist,
                       arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
                       unsigned int* numchecked, arma::ivec* indices, double tol, 
                       int maxit, unsigned int cur, std::string metric, 
                       double LowerBound, arma::uvec* NewOrder, Progress* p, 
                       double LowerMetric){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < *BestMetric){
    // Updating progress
    p->update(2);
    p->print();
    
    // Creating vectors to be used later
    arma::uvec NewOrder2(cur + 1);
    arma::vec Metrics(cur + 1);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder->at(j)) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      NewOrder2(j) = NewOrder->at(j);
      Metrics(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
                 method, m, Link, Dist, 
                 tol, maxit, metric);
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += NewOrder2.n_elem;
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    if(Metrics(0) < *BestMetric){
      *BestMetric = Metrics(0);
      *BestModel = *CurModel;
      BestModel->at(NewOrder2(0)) = 0;
    }
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Defining Bounds to store the lower bounds
    arma::vec Bounds(Metrics.n_elem - 1);
    
  // Computing lower bounds
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder2(j)) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      Bounds(j - 1) = BackwardGetBound(X, indices, &CurModel2, &NewOrder2, 
                 j - 1, Metrics(j), metric, 
                 xTemp.n_cols);
    }
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Defining lower model for switch
    arma::uvec revNewOrder2 = reverse(NewOrder2);
    arma::ivec LowerModel = *CurModel;
    LowerModel(revNewOrder2(revNewOrder2.n_elem - 1)) = 0;
    
    // Reversing for possible switch to forward branching
    Bounds = reverse(Bounds);
    Metrics = reverse(Metrics);
    
    
    // Lower models and branching only need to be done if revNewOrder2.n_elem > 1
    if(revNewOrder2.n_elem > 1){
      // Creating vector to store metric values from lower models
      arma::vec Lower(Bounds.n_elem);
      
      // Fitting lower models
#pragma omp parallel for schedule(dynamic)
      for(int j = revNewOrder2.n_elem - 2; j >= 0; j--){
        if(j > 0){
          // Getting lower model
          /// Only need to fit lower model for j > 0
          arma::ivec NewLowerModel = LowerModel;
          NewLowerModel.elem(revNewOrder2.subvec(j, revNewOrder2.n_elem - 2)) = 
          arma::ivec(revNewOrder2.n_elem - 1 - j, arma::fill::zeros);
          arma::mat xTemp = GetMatrix(X, &NewLowerModel, indices);
          Lower.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &NewLowerModel,
                                         method, m, Link, Dist, 
                                         tol, maxit, metric);
   
           // Seeing if we can tighten bounds
          double MetricVal = 0;
          if(metric == "AIC"){
            MetricVal = 2;
          }
          else if(metric == "BIC"){
            MetricVal = log(X->n_rows);
          }
            
          // If lower metric is sufficiently far from lowerbound, then we can increase lowerbound
          if(Lower.at(j) - Bounds.at(j) > MetricVal){
            Bounds.at(j) += MetricVal;
          }else{
            Bounds.at(j) = arma::datum::inf;
          }
        }
          else{
            Lower.at(j) = LowerMetric;
        }
      }
      
      // Updating numchecked
      (*numchecked) += revNewOrder2.n_elem - 2;
      
      // Checking if we need to update bounds
      sorted = sort_index(Lower);
      if(Lower.at(sorted.at(0)) < *BestMetric){
        *BestMetric = Lower.at(sorted.at(0));
        *BestModel = LowerModel;
        for(unsigned int j = revNewOrder2.n_elem - 2; j >= sorted.at(0); j--){
          BestModel->at(revNewOrder2.at(j)) = 0;
        }
      }
      
      // Performing the branching
      for(int j = revNewOrder2.n_elem - 2; j >= 0; j--){
        // Updating lower model for current iteration
        LowerModel(revNewOrder2(j)) = 0;
        
        if(Metrics.at(j) > Lower.at(j)){
          // If Lower model has better metric value than upper model use forward
        SwitchForwardBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &LowerModel, BestModel, 
                          BestMetric, numchecked, indices, tol, maxit, j + 1, metric, 
                          Bounds.at(j), &revNewOrder2, p, Metrics.at(j));
        }
        else{
          // Creating new CurModel for next set of models
          arma::ivec CurModel2 = *CurModel;
          CurModel2.at(revNewOrder2.at(j)) = 0;
          
          // If upper model has better metric value than lower model use backward
          SwitchBackwardBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
                                 BestMetric, numchecked, indices, tol, maxit, 
                                 revNewOrder2.n_elem - 2 - j, metric, 
                                 Bounds.at(j), &NewOrder2, p, Lower.at(j));
        }
      }
    }
  }
  else{
    // Updating progress since we have cut off part of the tree
    p->update(GetNum(cur + 1, cur + 1));
    p->print();
  }
}


// Branch and bound method
// [[Rcpp::export]]
List SwitchBranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                          IntegerVector indices, IntegerVector num,
                          std::string method, int m,
                          std::string Link, std::string Dist,
                          unsigned int nthreads, double tol, int maxit, 
                          IntegerVector keep, std::string metric,
                          bool display_progress, double BestMetric){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  CurModel.replace(1, 0);
  
  
  // Getting X'WX
  arma::mat XTWX;
  if(Dist == "gaussian" || Dist == "gamma"){
    XTWX = X.t() * X;
  }else{
    arma::vec beta(X.n_cols, arma::fill::zeros);
    arma::vec mu = LinkCpp(&X, &beta, &Offset, Link, Dist);
    arma::vec Deriv = DerivativeCpp(&X, &beta, &Offset, &mu, Link, Dist);
    arma::vec Var = Variance(&mu, Dist);
    XTWX = FisherInfoCpp(&X, &Deriv, &Var);
  }
  
  // Creating necessary scalars
  unsigned int numchecked = 0;
  unsigned int size = 0;
  
  // Changing number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Getting size of model space to check
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      size++;
    }
  }
  
  // Creating object to report progress
  Progress p(GetNum(size, size), display_progress);
  p.print();
  
  
  // Making vector of order to look at variables
  arma::uvec NewOrder(size);
  unsigned int k = 0;
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      NewOrder.at(k++) = j;
    }
  }
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Fitting lower model
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  double CurMetric = MetricHelper(&xTemp, &XTWX, &Y, &Offset, &Indices, 
                                       &CurModel, method, m, Link, Dist, 
                                       tol, maxit, metric);
  
  // Updating BestMetric and BestModel if CurMetric is better than BestMetric
  if(CurMetric < BestMetric){
    BestMetric = CurMetric;
    BestModel = CurModel;
  }
  
  // Incrementing numchecked  
  numchecked++;
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  arma::vec Metrics(1);
  Metrics.at(0) = arma::datum::inf;
  LowerBound = GetBound(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel,
                           &Indices, tol, maxit, metric, 
                           0, sum(abs(CurModel)), &NewOrder, LowerBound, CurMetric, &Metrics);
  
  // Updating BestMetric and BestModel if metric from upper model is better than BestMetric
  if(Metrics.at(0) < BestMetric){
    BestMetric = Metrics.at(0);
    
    // Defining Upper model
    arma::ivec UpperModel = CurModel;
    for(unsigned int i = 0; i < NewOrder.n_elem; i++){
      UpperModel.at(NewOrder.at(i)) = 1;
    }
    BestModel = UpperModel;
  }
  
  // Incrementing numchecked
  numchecked++;
  
  // Starting branching process
  if(Metrics.at(0) < CurMetric){
    // Branching forward if lower model has better metric value than upper model
    SwitchForwardBranch(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
            &BestMetric, &numchecked, &Indices, tol, maxit, 0, metric, 
            LowerBound, &NewOrder, &p, Metrics.at(0));
  }else{
    // Branching backward if upper model has better metric value than lower model
    arma::ivec UpperModel = CurModel;
    for(unsigned int i = 0; i < NewOrder.n_elem; i++){
      UpperModel.at(NewOrder.at(i)) = 1;
    }
    
    SwitchBackwardBranch(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &UpperModel, &BestModel, 
                           &BestMetric, &numchecked, &Indices, tol, maxit, NewOrder.n_elem - 1, metric, 
                           LowerBound, &NewOrder, &p, CurMetric);
  }
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Printing off final update
  p.finalprint();
  
  // Getting x matrix for best model found
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  // Fitting best model
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                                 nthreads, tol, maxit);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  // Resetting number of threads if OpenMP is defined
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}
