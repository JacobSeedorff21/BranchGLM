#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
#include "VariableSelection.h"
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

// Given a current model, this finds the best variable to remove
void Fastdrop1(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
               const arma::imat* Interactions, std::string method, int m, std::string Link, std::string Dist,
               arma::ivec* CurModel, arma::vec* BestModel, double* BestMetric, 
               unsigned int* numchecked, bool* flag, arma::ivec* order, unsigned int i,
               arma::ivec* indices, double tol, int maxit, const arma::vec* pen, 
               arma::vec* bounds, unsigned int nthreads){
  
  arma::vec Metrics(CurModel->n_elem);
  arma::ivec Counts(CurModel->n_elem, arma::fill::zeros);
  Metrics.fill(arma::datum::inf);
  arma::mat NewModels(X->n_cols, CurModel->n_elem, arma::fill::zeros);
  arma::uvec fitOrder = arma::sort_index(*bounds); 
  
  // Removing each variable one at a time and calculating metric for each model
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int i = 0; i < fitOrder.n_elem; i++){
    unsigned int j = fitOrder.at(i);
    double tempBestMetric;
    if(*BestMetric < Metrics.min()){
      tempBestMetric = *BestMetric;
    }else{
      tempBestMetric = Metrics.min();
    }
    if(CurModel->at(j) == 1 && (bounds->at(j) + 1e-6) < tempBestMetric){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 0;
      if(CheckModel(&CurModel2, Interactions)){
        Counts.at(j) = 1;
        Metrics.at(j) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel2, method, m, Link, Dist, 
                   tol, maxit, pen, j, &NewModels);
        if(!std::isinf(Metrics.at(j))){
          bounds->at(j) = Metrics.at(j);
        }
      }
    }
  }
  
  // Updating best metric (probably can't do this in parallel)
  unsigned int BestVar = Metrics.index_min();
  double NewMetric = Metrics.at(BestVar);
  if(NewMetric < *BestMetric){
    *BestMetric = NewMetric;
    *flag = false;
  }
  
  // Updating numchecked
  (*numchecked) += arma::accu(Counts);
  
  // Updating best model
  if(!(*flag)){
    CurModel->at(BestVar) = 0;
    *BestModel = NewModels.col(BestVar);
    *bounds -= pen->at(BestVar);
    order->at(i) = BestVar;
  }
}

// Performs backward elimination
// [[Rcpp::export]]
List FastBackwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                     IntegerVector indices, IntegerVector num,
                     IntegerMatrix interactions, 
                     std::string method, int m,
                     std::string Link, std::string Dist,
                     unsigned int nthreads, double tol, int maxit,
                     IntegerVector keep, unsigned int steps, NumericVector pen){
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
  
  // Creating neccessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  arma::imat BestModels(CurModel.n_elem, CurModel.n_elem + 1, arma::fill::zeros);
  arma::vec BestMetrics(CurModel.n_elem + 1, 1);
  BestMetrics.fill(arma::datum::inf);
  arma::vec BestModel(X.n_cols, 1, arma::fill::zeros);
  arma::mat BestBetas(X.n_cols, CurModel.n_elem + 1, arma::fill::zeros);
  arma::vec bounds(CurModel.n_elem);
  bounds.fill(-arma::datum::inf);
  CurModel.replace(0, 1);
  IntegerVector order(CurModel.n_elem, -1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
  // Creating necessary scalars
  double BestMetric = arma::datum::inf;
  arma::mat betaMat(X.n_cols, 1, arma::fill::zeros);
  BestMetric = MetricHelper(&X, &XTWX, &Y, &Offset, &Indices, &CurModel, method, m, Link, Dist, tol, maxit,
                            &Pen, 0, &betaMat);
  BestModel = betaMat.col(0);
  BestMetrics.at(0) = BestMetric;
  BestModels.col(0) = CurModel;
  BestBetas.col(0) = BestModel;
  unsigned int numchecked = 1;
  
  // Performing Backward elimination
  for(unsigned int i = 0; i < steps; i++){
    checkUserInterrupt();
    bool flag = true;
    Fastdrop1(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModel, 
              &BestMetric, &numchecked, &flag, &Order, i, &Indices, tol, maxit, &Pen, &bounds, 
              nthreads);
    
    // Stopping the process if no better model is found
    if(flag){
      break;
    }else{
      BestModels.col(i + 1) = CurModel;
      BestBetas.col(i + 1) = BestModel;
      BestMetrics.at(i + 1) = BestMetric;
    }
  }
  
  List FinalList = List::create(Named("order") = order,
                                Named("numchecked") = numchecked,
                                Named("bestmetrics") = BestMetrics,
                                Named("bestmodels") = BestModels,
                                Named("betas") = BestBetas);
  
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// Given a current model, this finds the best variable to remove
void Fastdrop2(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
               const arma::imat* Interactions, std::string method, int m, std::string Link, std::string Dist,
               arma::ivec* CurModel, arma::vec* BestModel, double* BestMetric, 
               unsigned int* numchecked, bool* flag, arma::imat* order, unsigned int i1,
               arma::ivec* indices, double tol, int maxit, const arma::vec* pen, 
               arma::vec* bounds, unsigned int nthreads, 
               arma::vec* AllMetrics, arma::mat* AllBetas, unsigned int* k){
  
  arma::vec Metrics(CurModel->n_elem);
  arma::ivec Counts(CurModel->n_elem, arma::fill::zeros);
  Metrics.fill(arma::datum::inf);
  arma::mat NewModels(X->n_cols, CurModel->n_elem, arma::fill::zeros);
  arma::uvec fitOrder = arma::sort_index(*bounds); 
  arma::ivec tempCurModel;
  
  // Removing each variable one at a time and calculating metric for each model
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int i = 0; i < fitOrder.n_elem; i++){
    unsigned int j = fitOrder.at(i);
    double tempBestMetric;
    if(*BestMetric < Metrics.min()){
      tempBestMetric = *BestMetric;
    }else{
      tempBestMetric = Metrics.min();
    }
    if(CurModel->at(j) == 1 && (bounds->at(j) + 1e-6) < tempBestMetric){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 0;
      if(CheckModel(&CurModel2, Interactions)){
        Counts.at(j)++;
        Metrics.at(j) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel2, method, m, Link, Dist, 
                   tol, maxit, pen, j, &NewModels);
        if(!std::isinf(Metrics.at(j))){
          bounds->at(j) = Metrics.at(j);
        }
      }
    } 
  }
  
// Updating best metric
unsigned int BestVar = Metrics.index_min();
double NewMetric = Metrics.at(BestVar);
double singleMetric = NewMetric;
arma::vec singleBeta = NewModels.col(BestVar);
if(NewMetric < *BestMetric){
  *BestMetric = NewMetric;
  tempCurModel = *CurModel;
  tempCurModel.at(BestVar) = 0;
  *BestModel = NewModels.col(BestVar);
  order->at(i1, 0) = BestVar;
  *flag = false;
}
  
  // Removing 2-variables at a time
  fitOrder = arma::sort_index(*bounds);
  arma::mat doubleMetrics(CurModel->n_elem, CurModel->n_elem);
  doubleMetrics.fill(arma::datum::inf);
  arma::mat doubleBounds(CurModel->n_elem, CurModel->n_elem);
  doubleBounds.fill(-arma::datum::inf);
  bool doubleFlag = true;
  unsigned int var1 = 0;
  unsigned int var2 = 0;
  for(unsigned int k = 0; k < CurModel->n_elem; k++){
    unsigned int i = fitOrder.at(k);
    arma::ivec CurModel2 = *CurModel;
    if(CurModel2.at(i) == 1){
      checkUserInterrupt();
      CurModel2.at(i) = 0;
      NewModels.fill(0);
#pragma omp parallel for schedule(dynamic, 1)
      for(unsigned int l = 0; l < k; l++){
        unsigned int j = fitOrder.at(l);
        if(CurModel2.at(j) == 1){
          double bound1 = bounds->at(i) - pen->at(j);
          double bound2 = bounds->at(j) - pen->at(i);
          double bound = std::max(bound1, bound2) + 1e-6;
          
          arma::ivec CurModel3 = CurModel2;
          CurModel3.at(j) = 0;
          
          double tempBestMetric;
          if(*BestMetric < doubleMetrics.col(i).min()){
            tempBestMetric = *BestMetric;
          }else{
            tempBestMetric = doubleMetrics.col(i).min();
          }
          
          if(CheckModel(&CurModel3, Interactions) && bound < tempBestMetric){
            Counts.at(j)++;
            doubleMetrics.at(j, i) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel3, method, m, Link, Dist, 
                       tol, maxit, pen, j, &NewModels);
            if(!std::isinf(doubleMetrics.at(j, i))){
              doubleBounds.at(i, j) = doubleMetrics.at(j, i);
              doubleBounds.at(j, i) = doubleMetrics.at(j, i);
            }
          }else{
            doubleBounds.at(i, j) = bound - 1e-6;
            doubleBounds.at(j, i) = bound - 1e-6;
          }
        }
      }
      // Updating best metric (probably can't do this in parallel)
      unsigned int BestVar = doubleMetrics.col(i).index_min();
      double NewMetric = doubleMetrics.at(BestVar, i);
      if(NewMetric < *BestMetric){
        *BestMetric = NewMetric;
        tempCurModel = *CurModel;
        tempCurModel.at(BestVar) = 0;
        tempCurModel.at(i) = 0;
        *BestModel = NewModels.col(BestVar);
        var1 = i;
        var2 = BestVar;
        order->at(i1, 0) = i;
        order->at(i1, 1) = BestVar;
        *flag = false;
        doubleFlag = false;
      }
    }
  }
  
  // Updating numchecked
  (*numchecked) += arma::accu(Counts);
  
  // Updating best model
  if(!(*flag)){
    *CurModel = tempCurModel;
    AllMetrics->at(*k) = singleMetric;
    AllBetas->col(*k) = singleBeta;
    AllMetrics->at(*k + 1) = *BestMetric;
    AllBetas->col(*k + 1) = *BestModel;
    
    // Getting bounds
    if(!doubleFlag){
      (*k) += 2;
      arma::vec bounds1 = max(doubleBounds.col(var1) - pen->at(var2), 
                              doubleBounds.col(var2) - pen->at(var1));
      *bounds = max(*bounds - pen->at(var1) - pen->at(var2), bounds1);
    }else{ 
      (*k)++;
      *bounds = doubleBounds.col(Metrics.index_min());
    }
  }
} 

// Performs backward elimination
// [[Rcpp::export]]
List FastDoubleBackwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                     IntegerVector indices, IntegerVector num,
                     IntegerMatrix interactions, 
                     std::string method, int m,
                     std::string Link, std::string Dist,
                     unsigned int nthreads, double tol, int maxit,
                     IntegerVector keep, unsigned int steps, NumericVector pen){
  
#ifdef _OPENMP 
  omp_set_num_threads(nthreads);
#endif 
  
  // Creating neccessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  arma::imat BestModels(CurModel.n_elem, CurModel.n_elem + 1, arma::fill::zeros);
  arma::vec BestMetrics(CurModel.n_elem + 1, 1);
  BestMetrics.fill(arma::datum::inf);
  arma::vec AllMetrics = BestMetrics;
  arma::vec BestModel(X.n_cols, 1, arma::fill::zeros);
  arma::mat BestBetas(X.n_cols, CurModel.n_elem + 1, arma::fill::zeros);
  arma::mat AllBetas(X.n_cols, CurModel.n_elem + 1, arma::fill::zeros);
  arma::vec bounds(CurModel.n_elem);
  bounds.fill(-arma::datum::inf);
  CurModel.replace(0, 1);
  IntegerMatrix order(CurModel.n_elem, 2);
  arma::imat Order(order.begin(), order.rows(), order.cols(), false, true);
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
  // Creating necessary scalars
  double BestMetric = arma::datum::inf;
  arma::mat betaMat(X.n_cols, 1, arma::fill::zeros);
  BestMetric = MetricHelper(&X, &XTWX, &Y, &Offset, &Indices, &CurModel, method, m, Link, Dist, tol, maxit,
                            &Pen, 0, &betaMat);
  BestModel = betaMat.col(0);
  BestMetrics.at(0) = BestMetric;
  AllMetrics.at(0) = BestMetric;
  BestModels.col(0) = CurModel;
  BestBetas.col(0) = BestModel;
  AllBetas.col(0) = BestModel;
  unsigned int numchecked = 1;
  
  // Performing Backward elimination
  unsigned int k = 1;
  for(unsigned int i = 0; i < steps; i++){
    checkUserInterrupt();
    bool flag = true;
    Fastdrop2(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModel, 
              &BestMetric, &numchecked, &flag, &Order, i, &Indices, tol, maxit, &Pen, &bounds, 
              nthreads, &AllMetrics, &AllBetas, &k);
    
    // Stopping the process if no better model is found
    if(flag){
      break;
    }else{
      BestModels.col(i + 1) = CurModel;
      BestBetas.col(i + 1) = BestModel;
      BestMetrics.at(i + 1) = BestMetric;
    }
  }
  
  List FinalList = List::create(Named("order") = order,
                                Named("numchecked") = numchecked,
                                Named("bestmetrics") = BestMetrics,
                                Named("bestmodels") = BestModels,
                                Named("betas") = BestBetas, 
                                Named("allbetas") = AllBetas, 
                                Named("allmetrics") = AllMetrics);
  
#ifdef _OPENMP 
  omp_set_num_threads(1);
#endif 
  
  return(FinalList);
}

// Given a current model, this finds the best variable to remove
void Slowdrop2(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
               const arma::imat* Interactions, std::string method, int m, std::string Link, std::string Dist,
               arma::ivec* CurModel, arma::vec* BestModel, double* BestMetric, 
               unsigned int* numchecked, bool* flag, arma::imat* order, unsigned int i1,
               arma::ivec* indices, double tol, int maxit, const arma::vec* pen, 
               arma::vec* bounds, unsigned int nthreads, 
               arma::vec* AllMetrics, arma::mat* AllBetas, unsigned int* k){
  
  arma::vec Metrics(CurModel->n_elem);
  arma::ivec Counts(CurModel->n_elem, arma::fill::zeros);
  Metrics.fill(arma::datum::inf);
  arma::mat NewModels(X->n_cols, CurModel->n_elem, arma::fill::zeros);
  arma::uvec fitOrder = arma::sort_index(*bounds); 
  arma::ivec tempCurModel;
  
  // Removing each variable one at a time and calculating metric for each model
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int i = 0; i < fitOrder.n_elem; i++){
    unsigned int j = fitOrder.at(i);
    if(CurModel->at(j) == 1){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 0;
      if(CheckModel(&CurModel2, Interactions)){
        Counts.at(j)++;
        Metrics.at(j) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel2, method, m, Link, Dist, 
                   tol, maxit, pen, j, &NewModels);
        if(!std::isinf(Metrics.at(j))){
          bounds->at(j) = Metrics.at(j);
        }
      } 
    } 
  }
  
  // Updating best metric
  unsigned int BestVar = Metrics.index_min();
  double NewMetric = Metrics.at(BestVar);
  double singleMetric = NewMetric;
  arma::vec singleBeta = NewModels.col(BestVar);
  if(NewMetric < *BestMetric){
    *BestMetric = NewMetric; 
    tempCurModel = *CurModel;
    tempCurModel.at(BestVar) = 0;
    *BestModel = NewModels.col(BestVar);
    order->at(i1, 0) = BestVar;
    *flag = false;
  }
  
  // Removing 2-variables at a time
  fitOrder = arma::sort_index(*bounds);
  arma::mat doubleMetrics(CurModel->n_elem, CurModel->n_elem);
  doubleMetrics.fill(arma::datum::inf);
  arma::mat doubleBounds(CurModel->n_elem, CurModel->n_elem);
  doubleBounds.fill(-arma::datum::inf);
  bool doubleFlag = true;
  unsigned int var1 = 0;
  unsigned int var2 = 0;
  for(unsigned int k = 0; k < CurModel->n_elem; k++){
    unsigned int i = fitOrder.at(k);
    arma::ivec CurModel2 = *CurModel;
    if(CurModel2.at(i) == 1){
      checkUserInterrupt();
      CurModel2.at(i) = 0;
      NewModels.fill(0);
#pragma omp parallel for schedule(dynamic, 1) 
      for(unsigned int l = 0; l < k; l++){
        unsigned int j = fitOrder.at(l);
        if(CurModel2.at(j) == 1){
          double bound1 = bounds->at(i) - pen->at(j);
          double bound2 = bounds->at(j) - pen->at(i);
          double bound = std::max(bound1, bound2) + 1e-6;
          
          arma::ivec CurModel3 = CurModel2;
          CurModel3.at(j) = 0;
          
          if(CheckModel(&CurModel3, Interactions)){
            Counts.at(j)++;
            doubleMetrics.at(j, i) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel3, method, m, Link, Dist, 
                             tol, maxit, pen, j, &NewModels); 
            if(!std::isinf(doubleMetrics.at(j, i))){
              doubleBounds.at(i, j) = doubleMetrics.at(j, i);
              doubleBounds.at(j, i) = doubleMetrics.at(j, i);
            } 
          }else{
            doubleBounds.at(i, j) = bound - 1e-6;
            doubleBounds.at(j, i) = bound - 1e-6;
          } 
        }
      }
      // Updating best metric (probably can't do this in parallel)
      unsigned int BestVar = doubleMetrics.col(i).index_min();
      double NewMetric = doubleMetrics.at(BestVar, i);
      if(NewMetric < *BestMetric){
        *BestMetric = NewMetric;
        tempCurModel = *CurModel;
        tempCurModel.at(BestVar) = 0;
        tempCurModel.at(i) = 0;
        *BestModel = NewModels.col(BestVar);
        var1 = i;
        var2 = BestVar;
        order->at(i1, 0) = i;
        order->at(i1, 1) = BestVar;
        *flag = false;
        doubleFlag = false;
      }
    } 
  }
  
  // Updating numchecked
  (*numchecked) += arma::accu(Counts);
  
  // Updating best model
  if(!(*flag)){
    *CurModel = tempCurModel;
    AllMetrics->at(*k) = singleMetric;
    AllBetas->col(*k) = singleBeta;
    AllMetrics->at(*k + 1) = *BestMetric;
    AllBetas->col(*k + 1) = *BestModel;
    
    // Getting bounds
    if(!doubleFlag){
      (*k) += 2;
      arma::vec bounds1 = max(doubleBounds.col(var1) - pen->at(var2), 
                              doubleBounds.col(var2) - pen->at(var1));
      *bounds = max(*bounds - pen->at(var1) - pen->at(var2), bounds1);
    }else{  
      (*k)++;
      *bounds = doubleBounds.col(Metrics.index_min());
    }
  } 
} 

// Performs backward elimination
// [[Rcpp::export]]
List DoubleBackwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                           IntegerVector indices, IntegerVector num,
                           IntegerMatrix interactions, 
                           std::string method, int m,
                           std::string Link, std::string Dist,
                           unsigned int nthreads, double tol, int maxit,
                           IntegerVector keep, unsigned int steps, NumericVector pen){
  
#ifdef _OPENMP 
  omp_set_num_threads(nthreads);
#endif 
  
  // Creating neccessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  arma::imat BestModels(CurModel.n_elem, CurModel.n_elem + 1, arma::fill::zeros);
  arma::vec BestMetrics(CurModel.n_elem + 1, 1);
  BestMetrics.fill(arma::datum::inf);
  arma::vec AllMetrics = BestMetrics;
  arma::vec BestModel(X.n_cols, 1, arma::fill::zeros);
  arma::mat BestBetas(X.n_cols, CurModel.n_elem + 1, arma::fill::zeros);
  arma::mat AllBetas(X.n_cols, CurModel.n_elem + 1, arma::fill::zeros);
  arma::vec bounds(CurModel.n_elem);
  bounds.fill(-arma::datum::inf);
  CurModel.replace(0, 1);
  IntegerMatrix order(CurModel.n_elem, 2);
  arma::imat Order(order.begin(), order.rows(), order.cols(), false, true);
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
  // Creating necessary scalars
  double BestMetric = arma::datum::inf;
  arma::mat betaMat(X.n_cols, 1, arma::fill::zeros);
  BestMetric = MetricHelper(&X, &XTWX, &Y, &Offset, &Indices, &CurModel, method, m, Link, Dist, tol, maxit,
                            &Pen, 0, &betaMat);
  BestModel = betaMat.col(0);
  BestMetrics.at(0) = BestMetric;
  AllMetrics.at(0) = BestMetric;
  BestModels.col(0) = CurModel;
  BestBetas.col(0) = BestModel;
  AllBetas.col(0) = BestModel;
  unsigned int numchecked = 1;
  
  // Performing Backward elimination
  unsigned int k = 1;
  for(unsigned int i = 0; i < steps; i++){
    checkUserInterrupt();
    bool flag = true;
    Slowdrop2(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModel, 
              &BestMetric, &numchecked, &flag, &Order, i, &Indices, tol, maxit, &Pen, &bounds, 
              nthreads, &AllMetrics, &AllBetas, &k);
    
    // Stopping the process if no better model is found
    if(flag){
      break;
    }else{
      BestModels.col(i + 1) = CurModel;
      BestBetas.col(i + 1) = BestModel;
      BestMetrics.at(i + 1) = BestMetric;
    }
  }
  
  List FinalList = List::create(Named("order") = order,
                                Named("numchecked") = numchecked,
                                Named("bestmetrics") = BestMetrics,
                                Named("bestmodels") = BestModels,
                                Named("betas") = BestBetas, 
                                Named("allbetas") = AllBetas, 
                                Named("allmetrics") = AllMetrics);
  
#ifdef _OPENMP 
  omp_set_num_threads(1);
#endif 
  
  return(FinalList);
}
