#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
#include "EffBranchGLMHelpers.h"
#include "VariableSelection.h"
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

// Function used to fit models and calculate desired metric
double EffMetricHelper(const arma::mat* X, const arma::mat* XTWX, 
                       const arma::vec* Y, const arma::vec* Offset,
                       const arma::ivec* Indices, const arma::ivec* CurModel,
                    std::string method, 
                    int m, std::string Link, std::string Dist,
                    double tol, int maxit, std::string metric){
  
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
  bool UseXTWX = true;
  arma::vec beta(X->n_cols, arma::fill::zeros);
  EffgetInit(&beta, X, &NewXTWX, Y, Offset, Dist, Link, &UseXTWX);
  int Iter;
  
  if(Dist == "gaussian" && Link == "identity"){
    Iter = EffLinRegCppShort(&beta, X, &NewXTWX, Y, Offset);
  }else if(method == "BFGS"){
    Iter = EffBFGSGLMCpp(&beta, X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
    
  }
  else if(method == "LBFGS"){
    Iter = EffLBFGSGLMCpp(&beta, X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, m, UseXTWX);
  }
  else{
    Iter = EffFisherScoringGLMCpp(&beta, X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }
  if(Iter == -2){
    return(arma::datum::inf);
  }
  
  arma::vec mu = ParLinkCpp(X, &beta, Offset, Link, Dist);
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
    LogLik -=  ParLogFact(Y);
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
  return(GetMetric(X, LogLik, Dist, metric));
}

// Fits upper model for a set of models and calculates the bound for the desired metric
double EffGetBound(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                std::string method, int m, std::string Link, std::string Dist,
                arma::ivec* CurModel,  arma::ivec* indices, 
                double tol, int maxit,
                std::string metric, unsigned int cur, int minsize,
                arma::uvec* NewOrder, double LowerBound, 
                double MetricVal){
  int Iter;
  arma::ivec UpperModel = *CurModel;
  
  // Checking if we need to fit model for upper bound and updating bounds if we don't need to
  if(cur == 1){
    if(metric == "AIC"){
      if(MetricVal - LowerBound < 2){
        return(arma::datum::inf);
      }
      else{
        return(UpdateBound(X, indices, NewOrder->at(cur - 1), LowerBound, metric, minsize));
      }
    }
    else if(metric == "BIC"){
      if(MetricVal - LowerBound < log(X->n_rows)){
        return(arma::datum::inf);
      }
      else{
        return(UpdateBound(X, indices, NewOrder->at(cur - 1), LowerBound, metric, minsize));
      }
    }
  }
  
  // Creating vector for the upper model
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
  
  arma::mat NewXTWX = XTWX->submat(NewInd, NewInd);
  bool UseXTWX = true;
  
  // Creating matrix for upper model and fitting it
  arma::mat xTemp = GetMatrix(X, &UpperModel, indices);
  arma::vec beta(xTemp.n_cols, arma::fill::zeros);
  EffgetInit(&beta, &xTemp, &NewXTWX, Y, Offset, Dist, Link, &UseXTWX);
  
  // Fitting model
  if(Dist == "gaussian" && Link == "identity"){
    Iter = EffLinRegCppShort(&beta, &xTemp, &NewXTWX, Y, Offset);
  }else if(method == "BFGS"){
    Iter = EffBFGSGLMCpp(&beta, &xTemp, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }
  else if(method == "LBFGS"){
    Iter = EffLBFGSGLMCpp(&beta, &xTemp, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, m, UseXTWX);
  }
  else{
    Iter = EffFisherScoringGLMCpp(&beta, &xTemp, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
  }
  
  // Checking for convergence and returning previous bound if it doesn't
  if(Iter < 0){
    return(LowerBound);
  }
  
  // Calculating metric value
  arma::vec mu = ParLinkCpp(&xTemp, &beta, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(&xTemp, Y, &mu, Dist);
  double dispersion = GetDispersion(&xTemp, Y, &mu, LogLik, Dist, tol);
  
  if(dispersion <= 0){
    return(LowerBound);
  }
  if(Dist == "gaussian"){
    double temp = xTemp.n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  ParLogFact(Y);
  }
  else if(Dist == "gamma"){
    double shape = 1 / dispersion;
    LogLik = shape * LogLik + 
      xTemp.n_rows * (shape * log(shape) - lgamma(shape)) + 
      (shape - 1) * arma::accu(log(*Y));
  }
  
  if(std::isnan(LogLik)){
    return(LowerBound);
  }
  
  // Getting bound and checking if it can tightened or removed
  double NewBound = BoundHelper(X, LogLik, Dist, metric, minsize);
  if(metric == "AIC"){
    if(MetricVal - NewBound < 2){
      NewBound = arma::datum::inf;
    }else{
      NewBound += 2;
    }
  }else if(metric == "BIC"){
    if(MetricVal - NewBound < log(X->n_rows)){
      NewBound = arma::datum::inf;
    }else{
      NewBound += log(X->n_rows);
    }
  }
  return(NewBound);
}

// Function used to performing branching for branch and bound method
void EffBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
            std::string method, int m, std::string Link, std::string Dist,
            arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
            unsigned int* numchecked, arma::ivec* indices, double tol, 
            int maxit, 
            int maxsize, unsigned int cur, std::string metric, 
            double LowerBound, arma::uvec* NewOrder, Progress* p){
  
  checkUserInterrupt();
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < *BestMetric && maxsize > 0){
    p->update(2);
    p->print();
    arma::uvec NewOrder2(NewOrder->n_elem - cur);
    arma::vec Metrics(NewOrder->n_elem - cur);
    
    //Getting metric values
#pragma omp parallel for 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder->at(j + cur)) = 1;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      NewOrder2.at(j) = NewOrder->at(j + cur);
      Metrics.at(j) = EffMetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, 
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
      Rcout << "Found better metric value "<< *BestMetric << std::endl;
    }
    
    // Getting lower bounds
    checkUserInterrupt();
    arma::uvec Counts(NewOrder2.n_elem - 1, arma::fill::zeros);
    if(maxsize > 1){
#pragma omp parallel for
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        CurModel2.at(NewOrder2.at(j)) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Counts.at(j) = 1;
        Metrics.at(j) = EffGetBound(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2,
                     indices, tol, maxit, metric, 
                     j + 1, xTemp.n_cols, &NewOrder2, LowerBound, 
                     Metrics.at(j));
      }
      (*numchecked) += sum(Counts);
    }
    checkUserInterrupt();
    
    // Recursively calling this function for each new model
    for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 1;
      EffBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
             BestMetric, numchecked, indices, tol, maxit, maxsize - 1, j + 1, metric, 
             Metrics.at(j), &NewOrder2, p);
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
List EffBranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
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
  
  // Making vector of order to look at variables
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      NewOrder.at(k++) = j;
    }
  }
  
  checkUserInterrupt();
  
  // Fitting initial model
  double CurMetric = EffMetricHelper(&xTemp, &XTWX, &Y, &Offset, &Indices, 
                                     &CurModel, method, m, Link, Dist, 
                                     tol, maxit, metric);
  
  if(CurMetric < BestMetric){
    BestMetric = CurMetric;
    BestModel = CurModel;
  }
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  LowerBound = EffGetBound(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel,
                        &Indices, tol, maxit, metric, 
                        0, sum(abs(CurModel)), &NewOrder, LowerBound, 
                        CurMetric);
  
  numchecked++;
  
  // Starting branching process
  EffBranch(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
            &BestMetric, &numchecked, &Indices, tol, maxit, maxsize, 0, metric, 
            LowerBound, &NewOrder, &p);
  
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
  
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// When doing the process backwards the upper model is already fit, so we just 
// need to use that and minimum number of variables to get bound
double EffBackwardGetBound(const arma::mat* X, arma::ivec* indices, arma::ivec* CurModel,
                        arma::uvec* NewOrder, unsigned int cur, double metricVal, 
                        std::string metric, unsigned int maxsize){
  
  if(metricVal == arma::datum::inf){
    return(-metricVal);
  }
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

// Function used to performing branching for branch and bound method
void EffBackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                    std::string method, int m, std::string Link, std::string Dist,
                    arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
                    unsigned int* numchecked, arma::ivec* indices, double tol, 
                    int maxit, unsigned int cur, std::string metric, 
                    double LowerBound, arma::uvec* NewOrder, Progress* p){
  
  checkUserInterrupt();
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < *BestMetric){
    p->update(2);
    p->print();
    arma::uvec NewOrder2(cur + 1);
    arma::vec Metrics(cur + 1);
    
    //Getting metric values
#pragma omp parallel for 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder->at(j)) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      NewOrder2.at(j) = NewOrder->at(j);
      Metrics.at(j) = EffMetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
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
      Rcout << "Found better metric value "<< *BestMetric << std::endl;
    }
    
    
    checkUserInterrupt();
    
#pragma omp parallel for
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      Metrics.at(j) = EffBackwardGetBound(X, indices, &CurModel2, &NewOrder2, 
                 j - 1, Metrics.at(j), metric, 
                 xTemp.n_cols);
    }
    
    checkUserInterrupt();
    
    // Recursively calling this function for each new model
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 0;
      EffBackwardBranch(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
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


// Branch and bound method
// [[Rcpp::export]]
List EffBackwardBranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
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
  
  
  
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  unsigned int size = 0;
  
  // Getting size of model space to check
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 1){
      size++;
    }
  }
  
  // Creating object to report progress
  Progress p(GetNum(size, size), display_progress);
  p.print();
  
  arma::uvec NewOrder(size);
  arma::vec Metrics(size);
  unsigned int k = 0;
  
  // Making vector of order to look at variables
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 1){
      NewOrder.at(k++) = j;
    }
  }
  
  // Fitting model with all variables included
  double CurMetric = EffMetricHelper(&X, &XTWX, &Y, &Offset, &Indices, &CurModel,
                                     method, m, Link, Dist, 
                                     tol, maxit, metric);
  if(CurMetric < BestMetric){
    BestMetric = CurMetric;
    BestModel = CurModel;
  }
  
  unsigned int numchecked = 1;
  
  // Getting lower bound for all models
  double LowerBound = EffBackwardGetBound(&X, &Indices, &CurModel, &NewOrder, 
                                          NewOrder.n_elem - 1, CurMetric, metric, 
                                          X.n_cols);
  
  // Starting the branching process
  EffBackwardBranch(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
                    &BestMetric, &numchecked, &Indices, tol, maxit, NewOrder.n_elem - 1, metric, 
                    LowerBound, &NewOrder, &p);
  
  
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
  
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

