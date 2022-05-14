#include <RcppArmadillo.h>
#include <cmath>
#include "ParBranchGLMHelpers.h"
#include "BranchGLMHelpers.h"
#include "VariableSelection.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

double ParMetricHelper(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                       std::string method, 
                       int m, std::string Link, std::string Dist,
                       double tol, int maxit, std::string metric){
  
  arma::vec beta(X->n_cols, arma::fill::zeros);
  int Iter;
  if(method == "BFGS"){
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
  double dispersion = 1;
  arma::vec mu = ParLinkCpp(X, &beta, Offset, Link, Dist);
  
  if(Dist == "gaussian"){
    dispersion = arma::accu(pow(*Y - mu, 2)) / (X->n_rows - X->n_cols);
  }
  
  double LogLik = -ParLogLikelihoodCpp(X, Y, &mu, Dist);
  if(Dist == "gaussian"){
    double temp = X->n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  ParLogFact(Y);
  }
  return(GetMetric(X, LogLik, Dist, metric));
}
double ParGetBound(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                        std::string method, int m, std::string Link, std::string Dist,
                        arma::ivec* CurModel,  arma::ivec* indices, 
                        double tol, int maxit,
                        std::string metric, unsigned int cur, int minsize,
                        arma::uvec* NewOrder, double LowerBound){
  int Iter;
  arma::ivec UpperModel = *CurModel;
  
  for(unsigned int i = cur; i < NewOrder->n_elem; i++){
    UpperModel.at(NewOrder->at(i)) = 1;
  }
  arma::mat xTemp = GetMatrix(X, &UpperModel, indices);
  
  arma::vec beta(xTemp.n_cols, arma::fill::zeros);
  
  if(method == "BFGS"){
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
  
  double dispersion = 1;
  arma::vec mu = ParLinkCpp(&xTemp, &beta, Offset, Link, Dist);
  
  if(Dist == "gaussian"){
    dispersion = arma::accu(pow(*Y - mu, 2)) / (xTemp.n_rows - xTemp.n_cols);
  }
  
  double LogLik = -ParLogLikelihoodCpp(&xTemp, Y, &mu, Dist);
  if(Dist == "gaussian"){
    double temp = xTemp.n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  ParLogFact(Y);
  }
  return(BoundHelper(X, LogLik, Dist, metric, minsize));
}

void ParBranch(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                      std::string method, int m, std::string Link, std::string Dist,
                      arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
                      unsigned int* numchecked,arma::ivec* indices, double tol, 
                      int maxit, 
                      int maxsize, unsigned int cur, std::string metric, 
                      double LowerBound, arma::uvec* NewOrder, Progress* p){
  
  checkUserInterrupt();
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
        Metrics.at(j) = ParMetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
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
            Metrics.at(j) = ParGetBound(X, Y, Offset, method, m, Link, Dist, &CurModel2,
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
        ParBranch(X, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
                          BestMetric, numchecked, indices, tol, maxit, maxsize - 1, j + 1, metric, 
                          Metrics.at(j), &NewOrder2, p);
      }
    }
  else{
    p->update(GetNum(NewOrder->n_elem - cur, maxsize));
    p->print();
  }
}


// [[Rcpp::export]]

List ParBranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                                 IntegerVector indices, IntegerVector num,
                                 std::string method, int m,
                                 std::string Link, std::string Dist,
                                 unsigned int nthreads, double tol, int maxit, 
                                 IntegerVector keep, int maxsize, std::string metric,
                                 bool display_progress){
  
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  double BestMetric = arma::datum::inf;
  BestMetric = ParMetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
                               tol, maxit, metric);
  unsigned int numchecked = 1;
  unsigned int size = 0;
  omp_set_num_threads(nthreads);
  
  // Getting size of model space to check
  
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      size++;
    }
  }
  
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
  // Fitting all models with 1 variable
#pragma omp parallel for
  for(unsigned int j = 0; j < NewOrder.n_elem; j++){
    arma::ivec CurModel2 = CurModel;
    CurModel2.at(NewOrder.at(j)) = 1;
    arma::mat xTemp = GetMatrix(&X, &CurModel2, &Indices);
    Metrics.at(j) = ParMetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
               tol, maxit, metric);
  }
  checkUserInterrupt();
  numchecked += NewOrder.n_elem;
  // Ordering variables based upon metric
  arma::uvec sorted = sort_index(Metrics);
  NewOrder = NewOrder(sorted);
  Metrics = Metrics(sorted);
  double LowerBound = -arma::datum::inf;
  LowerBound = ParGetBound(&X, &Y, &Offset, method, m, Link, Dist, &CurModel,
                                       &Indices, tol, maxit, metric, 
                                       0, xTemp.n_cols, &NewOrder, LowerBound);
  numchecked++;
  if(Metrics.at(0) < BestMetric){
    BestMetric = Metrics.at(0);
    BestModel.at(NewOrder.at(0)) = 1;
  }
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
        Metrics.at(j) = ParGetBound(&X, &Y, &Offset, method, m, Link, Dist, &CurModel2,
                   &Indices, tol, maxit, metric, 
                   j + 1, xTemp.n_cols, &NewOrder, Metrics.at(j));
      }
    }
  }
  numchecked += sum(Counts);
  if(NewOrder.n_elem > 1){
    p.update(1);
  }
  for(unsigned int j = 0; j < NewOrder.n_elem - 1; j++){
    arma::ivec CurModel2 = CurModel;
    CurModel2.at(NewOrder.at(j)) = 1;
    ParBranch(&X, &Y, &Offset, method, m, Link, Dist, &CurModel2, &BestModel, 
                     &BestMetric, &numchecked, &Indices, tol, maxit, maxsize - 1, j + 1, metric, 
                     Metrics.at(j), &NewOrder, &p);
    
  }
  checkUserInterrupt();
  p.finalprint();
  
  // Getting x matrix for best model found
  
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                                  nthreads, tol, maxit);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  omp_set_num_threads(1);
  
  return(FinalList);
}
