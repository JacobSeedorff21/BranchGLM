#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
#include "VariableSelection.h"
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

// Given a current model, this finds the best variable to add to the model
void add1(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
          const arma::imat* Interactions, std::string method, int m, std::string Link, std::string Dist,
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
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 1;
      if(CheckModel(&CurModel2, Interactions)){
        // This model is valid, so we fit it
        Counts.at(j) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, method, m, Link, Dist, 
                   tol, maxit, metric);
      }
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
                IntegerMatrix interactions,
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
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  IntegerVector order(CurModel.n_elem, -1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
  
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
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &XTWX, &Y, &Offset, &Indices, &CurModel, method, m, Link, Dist, 
                               tol, maxit, metric);
  unsigned int numchecked = 0;
  
  
  // Performing forward selection
  for(unsigned int i = 0; i < steps; i++){
    checkUserInterrupt();
    bool flag = true;
    CurModel = BestModel;
    add1(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModel, 
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
void drop1(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
           const arma::imat* Interactions, std::string method, int m, std::string Link, std::string Dist,
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
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 0;
      if(CheckModel(&CurModel2, Interactions)){
        Counts.at(j) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, method, m, Link, Dist, 
                   tol, maxit, metric);
      }
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
                 IntegerMatrix interactions, 
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
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  IntegerVector order(CurModel.n_elem, - 1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
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
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &XTWX, &Y, &Offset, &Indices, &CurModel, method, m, Link, Dist, tol, maxit,
                               metric);
  
  unsigned int numchecked = 0;
  
  // Performing Backward elimination
  for(unsigned int i = 0; i < steps; i++){
    
    bool flag = true;
    CurModel = BestModel;
    drop1(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModel, 
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