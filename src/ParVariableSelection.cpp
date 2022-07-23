#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
#include "VariableSelection.h"
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

// Function used to fit models and calculate desired metric
double MetricHelper(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                       std::string method, 
                       int m, std::string Link, std::string Dist,
                       double tol, int maxit, std::string metric){
  
  arma::vec beta(X->n_cols, arma::fill::zeros);
  PargetInit(&beta, X, Y, Offset, Dist, Link);
  int Iter;
  
  if(Dist == "gaussian" && Link == "identity"){
    Iter = ParLinRegCppShort(&beta, X, Y, Offset);
  }else if(method == "BFGS"){
    Iter = ParBFGSGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, maxit);
    
  }
  else if(method == "LBFGS"){
    Iter = ParLBFGSGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, maxit, m);
  }
  else{
    Iter = ParFisherScoringGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, maxit);
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

// Given a current model, this finds the best variable to add to the model
void add1(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
          std::string method, int m, std::string Link, std::string Dist,
          arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
          unsigned int* numchecked, bool* flag, arma::ivec* order, unsigned int i,
          arma::ivec* indices, double tol, int maxit, std::string metric){
  
  arma::vec Metrics(CurModel->n_elem, arma::fill::zeros);
  Metrics.fill(arma::datum::inf);
  arma::ivec Counts(CurModel->n_elem, arma::fill::zeros);
  checkUserInterrupt();
  
  // Adding each variable one at a time and calculating metric for each model
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int j = 0; j < CurModel->n_elem; j++){
    if(CurModel->at(j) == 0){
      Counts.at(j) = 1;
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 1;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      Metrics.at(j) = MetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
                 tol, maxit, metric);
    }
  }
  
  // Updating numchecked
  (*numchecked) += arma::accu(Counts);
  
  // Updating best model
  unsigned int BestVar = Metrics.index_min();
  double NewMetric = Metrics.at(BestVar);
  checkUserInterrupt();
  if(NewMetric < *BestMetric){
    arma::ivec CurModel2 = *CurModel;
    CurModel2.at(BestVar) = 1;
    *BestModel = CurModel2;
    *BestMetric = NewMetric;
    *flag = false;
    order->at(i) = BestVar;
  }
}

// Performs forward selection
// [[Rcpp::export]]
List ForwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                IntegerVector indices, IntegerVector num,
                std::string method, int m,
                std::string Link, std::string Dist,
                unsigned int nthreads, double tol, int maxit,
                IntegerVector keep, 
                unsigned int steps, std::string metric){
  
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  IntegerVector order(CurModel.n_elem, -1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
  // Creating necessary scalars
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
                               tol, maxit, metric);
  unsigned int numchecked = 0;
  
  // Performing forward selection
  for(unsigned int i = 0; i < steps; i++){
    checkUserInterrupt();
    bool flag = true;
    CurModel = BestModel;
    add1(&X, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
         &BestMetric, &numchecked, &flag, &Order, i, &Indices, tol, maxit, metric);
    
    // Stopping process if no better model is found
    if(flag){
      break;
    }
  }
  // Getting x matrix for best model found
  checkUserInterrupt();
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  // Fitting best model
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                                 nthreads, tol, maxit);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("order") = order,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// Given a current model, this finds the best variable to remove
void drop1(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
           std::string method, int m, std::string Link, std::string Dist,
           arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
           unsigned int* numchecked, bool* flag, arma::ivec* order, unsigned int i,
           arma::ivec* indices, double tol, int maxit, std::string metric){
  
  arma::vec Metrics(CurModel->n_elem);
  arma::ivec Counts(CurModel->n_elem, arma::fill::zeros);
  Metrics.fill(arma::datum::inf);
  
  // Removing each variable one at a time and calculating metric for each model
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int j = 0; j < CurModel->n_elem; j++){
    if(CurModel->at(j) == 1){
      Counts.at(j) = 1;
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      Metrics.at(j) = MetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
                 tol, maxit, metric);
      
      
    }
  }
  
  // Updating numchecked
  (*numchecked) += arma::accu(Counts);
  
  // Updating best model
  unsigned int BestVar = Metrics.index_min();
  double NewMetric = Metrics.at(BestVar);
  if(NewMetric < *BestMetric){
    arma::ivec CurModel2 = *CurModel;
    CurModel2.at(BestVar) = 0;
    *BestModel = CurModel2;
    *BestMetric = NewMetric;
    *flag = false;
    order->at(i) = BestVar;
  }
}

// Performs backward elimination
// [[Rcpp::export]]
List BackwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                 IntegerVector indices, IntegerVector num,
                 std::string method, int m,
                 std::string Link, std::string Dist,
                 unsigned int nthreads, double tol, int maxit,
                 IntegerVector keep, unsigned int steps, std::string metric){
  
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Creating neccessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  IntegerVector order(CurModel.n_elem, - 1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
  // Creating necessary scalars
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, tol, maxit,
                               metric);
  
  unsigned int numchecked = 0;
  
  // Performing Backward elimination
  for(unsigned int i = 0; i < steps; i++){
    
    bool flag = true;
    CurModel = BestModel;
    drop1(&X, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
          &BestMetric, &numchecked, &flag, &Order, i, &Indices, tol, maxit, metric);
    
    // Stopping the process if no better model is found
    if(flag){
      break;
    }
  }
  
  // Getting x matrix for best model found
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  
  // Fitting best model
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                                 nthreads, tol, maxit);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("order") = order,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// Fits upper model for a set of models and calculates the bound for the desired metric
double GetBound(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                   std::string method, int m, std::string Link, std::string Dist,
                   arma::ivec* CurModel,  arma::ivec* indices, 
                   double tol, int maxit,
                   std::string metric, unsigned int cur, int minsize,
                   arma::uvec* NewOrder, double LowerBound){
  int Iter;
  arma::ivec UpperModel = *CurModel;
  
  // Creating vector for the upper model
  for(unsigned int i = cur; i < NewOrder->n_elem; i++){
    UpperModel.at(NewOrder->at(i)) = 1;
  }
  
  // Creating matrix for upper model and fitting it
  arma::mat xTemp = GetMatrix(X, &UpperModel, indices);
  arma::vec beta(xTemp.n_cols, arma::fill::zeros);
  PargetInit(&beta, &xTemp, Y, Offset, Dist, Link);
  
  if(Dist == "gaussian" && Link == "identity"){
    Iter = ParLinRegCppShort(&beta, &xTemp, Y, Offset);
  }else if(method == "BFGS"){
    Iter = ParBFGSGLMCpp(&beta, &xTemp, Y, Offset, Link, Dist, tol, maxit);
  }
  else if(method == "LBFGS"){
    Iter = ParLBFGSGLMCpp(&beta, &xTemp, Y, Offset, Link, Dist, tol, maxit, m);
  }
  else{
    Iter = ParFisherScoringGLMCpp(&beta, &xTemp, Y, Offset, Link, Dist, tol, maxit);
  }
  if(Iter < 0){
    return(LowerBound);
  }
  
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
  return(BoundHelper(X, LogLik, Dist, metric, minsize));
}

// Function used to performing branching for branch and bound method
void Branch(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
               std::string method, int m, std::string Link, std::string Dist,
               arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
               unsigned int* numchecked, arma::ivec* indices, double tol, 
               int maxit, 
               int maxsize, unsigned int cur, std::string metric, 
               double LowerBound, arma::uvec* NewOrder, Progress* p){
  
  checkUserInterrupt();
  // Continuing branching process if lower bound is smaler than the best observed metric
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
      Metrics.at(j) = MetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
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
    
    //Getting lower bounds
    checkUserInterrupt();
    arma::uvec Counts(NewOrder->n_elem, arma::fill::zeros);
    if(maxsize > 1){
#pragma omp parallel for
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        CurModel2.at(NewOrder2.at(j)) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics.at(j) = UpdateBound(X, indices, NewOrder2.at(j), LowerBound, metric, xTemp.n_cols);
        if(j > 0 && Metrics.at(j) < *BestMetric){
          Counts.at(j) = 1;
          Metrics.at(j) = GetBound(X, Y, Offset, method, m, Link, Dist, &CurModel2,
                     indices, tol, maxit, metric, 
                     j + 1, xTemp.n_cols, &NewOrder2, Metrics.at(j));
        }
      }
      (*numchecked) += sum(Counts);
    }
    
    checkUserInterrupt();
    
    // Recursively calling this function for each new model
    for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 1;
      Branch(X, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
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
List BranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                          IntegerVector indices, IntegerVector num,
                          std::string method, int m,
                          std::string Link, std::string Dist,
                          unsigned int nthreads, double tol, int maxit, 
                          IntegerVector keep, int maxsize, std::string metric,
                          bool display_progress){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  
  // Creating necessary scalars
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
                               tol, maxit, metric);
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
  p.update(1);
  
  arma::uvec NewOrder(size);
  arma::vec Metrics(size);
  unsigned int k = 0;
  
  // Making vector of order to look at variables
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      NewOrder.at(k++) = j;
    }
  }
  
  checkUserInterrupt();
  // Fitting all models with 1 additional variable
#pragma omp parallel for
  for(unsigned int j = 0; j < NewOrder.n_elem; j++){
    arma::ivec CurModel2 = CurModel;
    CurModel2.at(NewOrder.at(j)) = 1;
    arma::mat xTemp = GetMatrix(&X, &CurModel2, &Indices);
    Metrics.at(j) = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
               tol, maxit, metric);
  }
  
  checkUserInterrupt();
  
  // Updating numchecked
  numchecked += NewOrder.n_elem;
  
  // Ordering variables based upon metric
  arma::uvec sorted = sort_index(Metrics);
  NewOrder = NewOrder(sorted);
  Metrics = Metrics(sorted);
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  LowerBound = GetBound(&X, &Y, &Offset, method, m, Link, Dist, &CurModel,
                           &Indices, tol, maxit, metric, 
                           0, sum(abs(CurModel)), &NewOrder, LowerBound);
  
  numchecked++;
  if(Metrics.at(0) < BestMetric){
    BestMetric = Metrics.at(0);
    BestModel.at(NewOrder.at(0)) = 1;
  }
  
  // Finding lower bounds
  arma::uvec Counts(NewOrder.n_elem, arma::fill::zeros);
  if(maxsize > 1){
#pragma omp parallel for
    for(unsigned int j = 0; j < NewOrder.n_elem; j++){
      arma::ivec CurModel2 = CurModel;
      CurModel2.at(NewOrder.at(j)) = 1;
      arma::mat xTemp = GetMatrix(&X, &CurModel2, &Indices);
      Metrics.at(j) = UpdateBound(&X, &Indices, NewOrder.at(j), LowerBound, metric, xTemp.n_cols);
      if(j > 0 && Metrics.at(j) < BestMetric){
        Counts.at(j) = 1;
        Metrics.at(j) = GetBound(&X, &Y, &Offset, method, m, Link, Dist, &CurModel2,
                   &Indices, tol, maxit, metric, 
                   j + 1, xTemp.n_cols, &NewOrder, Metrics.at(j));
      }
    }
  }
  
  // Updating numchecked
  numchecked += sum(Counts);
  if(NewOrder.n_elem > 1){
    p.update(1);
  }
  
  // Branching for each of the variables
  for(unsigned int j = 0; j < NewOrder.n_elem - 1; j++){
    arma::ivec CurModel2 = CurModel;
    CurModel2.at(NewOrder.at(j)) = 1;
    Branch(&X, &Y, &Offset, method, m, Link, Dist, &CurModel2, &BestModel, 
              &BestMetric, &numchecked, &Indices, tol, maxit, maxsize - 1, j + 1, metric, 
              Metrics.at(j), &NewOrder, &p);
    
  }
  
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
