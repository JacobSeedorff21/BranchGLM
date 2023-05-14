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

void UpdateBestMetrics(arma::imat* BestModels, 
                       arma::vec* BestMetrics, 
                       arma::imat* Models, 
                       arma::vec* Metrics, 
                       double cutoff){
  
  // Defining matrix to store all models
  arma::imat NewModels(Models->n_rows, BestModels->n_cols + Models->n_cols);
  NewModels.cols(0, BestModels->n_cols - 1) = *BestModels;
  NewModels.cols(BestModels->n_cols, NewModels.n_cols - 1) = *Models;
  
  // Defining vector to order metrics
  arma::vec tempMetrics(BestMetrics->n_elem + Metrics->n_elem);
  tempMetrics.subvec(0, BestMetrics->n_elem - 1) = *BestMetrics;
  tempMetrics.subvec(BestMetrics->n_elem, tempMetrics.n_elem - 1) = *Metrics;
  
  // Sorting by metric values
  arma::uvec sorted = sort_index(tempMetrics);
  tempMetrics = tempMetrics(sorted);
  NewModels = NewModels.cols(sorted);
  
  // Only getting first k metrics
  if(cutoff == 0){
  *BestModels = NewModels.cols(0, BestModels->n_cols - 1);
  *BestMetrics = tempMetrics.subvec(0, BestMetrics->n_elem - 1);
  }
  else{
    // Only keeping models with metric values within cutoff of the best metric
    arma::uvec ind = find(tempMetrics < tempMetrics.at(0) + cutoff);
    *BestModels = NewModels.cols(ind);
    *BestMetrics = tempMetrics.elem(ind);
  }
}

// Function used to performing branching for branch and bound method
void Branch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
            const arma::imat* Interactions, 
            std::string method, int m, std::string Link, std::string Dist,
            arma::ivec* CurModel, arma::imat* BestModels, arma::vec* BestMetrics, 
            unsigned int* numchecked, arma::ivec* indices, double tol, 
            int maxit, 
            int maxsize, unsigned int cur, const arma::vec* pen, 
            double LowerBound, arma::uvec* NewOrder, Progress* p, double cutoff){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Getting metric value used to cutoff branches
  double metricCutoff;
  if(cutoff == 0){
    metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
  }
  else{
    metricCutoff = BestMetrics->at(0) + cutoff;
  }
  
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < metricCutoff && maxsize > 0){
    
    // Updating progress
    p->update(2);
    p->print();
    
    // Creating vectors to be used later on
    arma::uvec NewOrder2 = NewOrder->subvec(cur, NewOrder->n_elem - 1);
    arma::vec Metrics(NewOrder2.n_elem);
    arma::uvec Counts(NewOrder2.n_elem, arma::fill::zeros);
    arma::imat NewModels(CurModel->n_elem, NewOrder2.n_elem, arma::fill::zeros);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 1;
      NewModels.col(j) = CurModel2;
      if(CheckModel(&CurModel2, Interactions)){
        // Only fitting model if it is valid
        Counts.at(j) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, 
                   method, m, Link, Dist, 
                   tol, maxit, pen);
      }
      else{
        // If model is not valid then set metric value to infinity
        Metrics.at(j) = arma::datum::inf;
      }
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += arma::accu(Counts);
    UpdateBestMetrics(BestModels, BestMetrics, &NewModels, &Metrics, cutoff);
    
    // Getting cutoff for new best metric
    if(cutoff == 0){
      metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
    }
    else{
      metricCutoff = BestMetrics->at(0) + cutoff;
    }
    
    // Updating best metrics must be done before sorting
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Only find bounds and perform branching if there is at least 1 element to branch on
    // and maxsize is greater than 1
    if(NewOrder2.n_elem > 1 && maxsize > 1){
      
      // Creating vector to store lower bounds
      arma::vec Bounds(NewOrder2.n_elem - 1);
      Bounds.fill(arma::datum::inf);
      bool flag = false;
      arma::uvec Counts2(NewOrder2.n_elem - 1, arma::fill::zeros);
      
      // Getting lower bounds
    #pragma omp parallel for schedule(dynamic)
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        if(!flag && CheckModels(CurModel, &NewOrder2, Interactions, j + 1)){
          // Only need to calculate bounds if this set of models is valid
          if(j > 0){
            Counts2.at(j) = 1;
          
            arma::mat xTemp = GetMatrix(X, CurModel, indices);
          
            // Getting lower bound of model without current variable necessarily included
            Bounds.at(j) = GetBound(X, XTWX, Y, Offset, method, m, Link, Dist, CurModel,
                         indices, tol, maxit, pen, j, xTemp.n_cols, &NewOrder2, LowerBound, &Metrics);
            Bounds.at(j) += min(*pen);
            }
          else{
            // Don't need to recalculate this
            Bounds.at(0) = LowerBound;
          }
          if(Bounds.at(j) > metricCutoff){
            // None of the next models can have a better lower bound
            flag = true;
          }
          else{
            // Updating bound
            Bounds.at(j) += pen->at(NewOrder2.at(j));
          }
        }
      }
      
      // Updating numchecked
      (*numchecked) += arma::accu(Counts2);
      
      // Checking for user interrupt
      checkUserInterrupt();
      
      // Recursively calling this function for each new model
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        CurModel2.at(NewOrder2.at(j)) = 1;
        Branch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &CurModel2, BestModels, 
               BestMetrics, numchecked, indices, tol, maxit, maxsize - 1, j + 1, pen, 
               Bounds.at(j), &NewOrder2, p, cutoff);
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
                       IntegerMatrix interactions,
                       std::string method, int m,
                       std::string Link, std::string Dist,
                       unsigned int nthreads, double tol, int maxit, 
                       IntegerVector keep, int maxsize, NumericVector pen,
                       bool display_progress, unsigned int NumBest, double cutoff){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::imat BestModels(keep.size(), NumBest, arma::fill::zeros);
  arma::vec BestMetrics(NumBest);
  BestMetrics.fill(arma::datum::inf);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  
  // Making sure that no variables are including in curmodel, except those kept in each model
  CurModel.replace(1, 0);
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
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
                                     tol, maxit, &Pen);
  
  // Updating BestMetric is CurMetric is better
  if(CurMetric < BestMetrics.at(0)){
    BestMetrics.at(0) = CurMetric;
    BestModels.col(0) = CurModel;
  }
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  arma::vec Metrics(1);
  Metrics.at(0) = arma::datum::inf;
  LowerBound = GetBound(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel,
                        &Indices, tol, maxit, &Pen, 
                        0, sum(abs(CurModel)), &NewOrder, LowerBound, &Metrics, true);
  
  // Incrementing numchecked
  numchecked++;
  
  // Starting branching process
  Branch(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModels, 
            &BestMetrics, &numchecked, &Indices, tol, maxit, maxsize, 0, &Pen, 
            LowerBound, &NewOrder, &p, cutoff);
  
  // Printing off final update
  p.finalprint();
  
  List FinalList = List::create(Named("bestmodels") = BestModels,
                                Named("numchecked") = numchecked,
                                Named("bestmetrics") = BestMetrics);
  
  // Resetting number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// Function used to performing branching for backward branch and bound method
void BackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                    const arma::imat* Interactions, 
                    std::string method, int m, std::string Link, std::string Dist,
                    arma::ivec* CurModel, arma::imat* BestModels, arma::vec* BestMetrics, 
                    unsigned int* numchecked, arma::ivec* indices, double tol, 
                    int maxit, unsigned int cur, const arma::vec* pen, 
                    double LowerBound, arma::uvec* NewOrder, Progress* p, double cutoff){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Getting metric value used to cutoff branches
  double metricCutoff;
  if(cutoff == 0){
    metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
  }
  else{
    metricCutoff = BestMetrics->at(0) + cutoff;
  }
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < metricCutoff){
    // Updating progress
    p->update(2);
    p->print();
    
    // Creating vectors to be used later
    arma::uvec NewOrder2(cur + 1);
    arma::vec Metrics(cur + 1);
    Metrics.fill(arma::datum::inf);
    arma::uvec Counts(cur + 1, arma::fill::zeros);
    arma::imat NewModels(CurModel->n_elem, NewOrder2.n_elem, arma::fill::zeros);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder->at(j)) = 0;
      NewOrder2.at(j) = NewOrder->at(j);
      NewModels.col(j) = CurModel2;
      if(CheckModel(&CurModel2, Interactions)){
        // Only fitting model if it is valid
        Counts.at(j) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
                                          method, m, Link, Dist, 
                                          tol, maxit, pen);
      }
    }
    
    
    
    // Updating numchecked and potentially updating the best model
    *numchecked += arma::accu(Counts);
    UpdateBestMetrics(BestModels, BestMetrics, &NewModels, &Metrics, cutoff);
    
    // Updating best metrics must be done before sorting
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Creating vector to store Counts
    arma::uvec Counts2(Metrics.n_elem - 1, arma::fill::zeros);
    
    // Getting lower bounds which are now stored in Metrics
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 0;
      if(BackwardCheckModels(&CurModel2, &NewOrder2, Interactions, j - 1)){
        // If this set of models is valid then find lower bound
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        
        if(!CheckModel(&CurModel2, Interactions)){
          // Fitting model for upper bound since it wasn't fit earlier
          // Only done when the upper model isn't valid, but the set is valid
          Counts2(j - 1) = 1;
          Metrics.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
                  method, m, Link, Dist, 
                  tol, maxit, pen);
        }
        
        Metrics.at(j) = BackwardGetBound(X, indices, &CurModel2, &NewOrder2, 
                   j, Metrics.at(j), pen, xTemp.n_cols);
      }
      else{
        // If this set of models is invalid then set bound to infinity so no branching is done
        Metrics.at(j) = arma::datum::inf;
      }
    }
    
    // Updating numchecked
    (*numchecked) += arma::accu(Counts2);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Recursively calling this function for each new model
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 0;
      BackwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &CurModel2, BestModels, 
                     BestMetrics, numchecked, indices, tol, maxit, j - 1, pen, 
                     Metrics.at(j), &NewOrder2, p, cutoff);
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
                               IntegerMatrix interactions,
                               std::string method, int m,
                               std::string Link, std::string Dist,
                               unsigned int nthreads, double tol, int maxit, 
                               IntegerVector keep, NumericVector pen,
                               bool display_progress, unsigned int NumBest, double cutoff){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::imat BestModels(keep.size(), NumBest, arma::fill::zeros);
  arma::vec BestMetrics(NumBest);
  BestMetrics.fill(arma::datum::inf);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  
  // Making sure that CurModel includes all variables 
  CurModel.replace(0, 1);
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
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
                                     tol, maxit, &Pen);
  
  // Updating BestMetric and BestModel if CurMetric is better than BestMetric
  if(CurMetric < BestMetrics.at(0)){
    BestMetrics.at(0) = CurMetric;
    BestModels.col(0) = CurModel;
  }
  
  // Creating numchecked to keep track of the number of models fit
  unsigned int numchecked = 1;
  
  // Getting lower bound for all models
  double LowerBound = BackwardGetBound(&X, &Indices, &CurModel, &NewOrder, 
                                          NewOrder.n_elem, CurMetric, &Pen, 
                                          X.n_cols);
  
  // Starting the branching process
  BackwardBranch(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModels, 
                    &BestMetrics, &numchecked, &Indices, tol, maxit, NewOrder.n_elem - 1, &Pen, 
                    LowerBound, &NewOrder, &p, cutoff);
  
  // Printing off final update
  p.finalprint();
  
  List FinalList = List::create(Named("bestmodels") = BestModels,
                                Named("numchecked") = numchecked,
                                Named("bestmetrics") = BestMetrics);
  
  // Resetting number of threads if OpenMP is defined
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}

// Defining backward branching function for switch method
// Forward declaration so this can be called by the forward switch branch
void SwitchBackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                             const arma::imat* Interactions,
                             std::string method, int m, std::string Link, std::string Dist,
                             arma::ivec* CurModel, arma::imat* BestModels, arma::vec* BestMetrics, 
                             unsigned int* numchecked, arma::ivec* indices, double tol, 
                             int maxit, unsigned int cur, const arma::vec* pen, 
                             double LowerBound, arma::uvec* NewOrder, Progress* p, 
                             double LowerMetric, double cutoff);

// Function used to performing branching for forward part of switch branch
void SwitchForwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
               const arma::imat* Interactions,
               std::string method, int m, std::string Link, std::string Dist,
               arma::ivec* CurModel, arma::imat* BestModels, arma::vec* BestMetrics, 
               unsigned int* numchecked, arma::ivec* indices, double tol, 
               int maxit, unsigned int cur, const arma::vec* pen, 
               double LowerBound, arma::uvec* NewOrder, Progress* p, 
               double UpperMetric, double cutoff){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Getting metric value used to cutoff branches
  double metricCutoff;
  if(cutoff == 0){
    metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
  }
  else{
    metricCutoff = BestMetrics->at(0) + cutoff;
  }
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < metricCutoff){
    // Updating progress
    p->update(4);
    p->print();
    
    // Creating vectors to be used later
    arma::uvec NewOrder2(NewOrder->n_elem - cur);
    arma::vec Metrics(NewOrder->n_elem - cur);
    arma::uvec Counts(NewOrder->n_elem - cur, arma::fill::zeros);
    arma::imat NewModels(CurModel->n_elem, NewOrder2.n_elem, arma::fill::zeros);
     
    // Getting metric values
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder->at(j + cur)) = 1;
      NewModels.col(j) = CurModel2;
      NewOrder2(j) = NewOrder->at(j + cur);
      if(CheckModel(&CurModel2, Interactions)){
        // Only fitting model if it is valid
        Counts.at(j) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2, 
                   method, m, Link, Dist, 
                   tol, maxit, pen);
      }
      else{
        // If model is not valid then set metric value to infinity
        Metrics(j) = arma::datum::inf;
      }
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += arma::accu(Counts);
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    UpdateBestMetrics(BestModels, BestMetrics, &NewModels, &Metrics, cutoff);
    
    // Updating metric cutoff
    if(cutoff == 0){
      metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
    }
    else{
      metricCutoff = BestMetrics->at(0) + cutoff;
    }
    
    // Updating metrics must be done before sorting
    Metrics = Metrics(sorted);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Only need to calculate bounds and branch if NewOrder2.n_elem > 1
    if(NewOrder2.n_elem > 1){
      
      // Creating vectors using in finding the bounds
      /// Bounds stores the bounds
      /// Metrics2 stores the metric values from the upper models
      arma::vec Bounds(NewOrder2.n_elem - 1);
      Bounds.fill(arma::datum::inf);
      arma::vec Metrics2(NewOrder2.n_elem - 1);
      arma::uvec Counts2(NewOrder2.n_elem - 1, arma::fill::zeros);
      Metrics2.fill(arma::datum::inf);
      arma::imat NewModels(CurModel->n_elem, Bounds.n_elem, arma::fill::zeros);
      bool flag = false;
      
#pragma omp parallel for schedule(dynamic)
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        arma::ivec CurModel3 = CurModel2;
        for(unsigned int i = j; i < NewOrder2.n_elem; i++){
          CurModel3.at(NewOrder2(i)) = 1;
        }
        NewModels.col(j) = CurModel3;
        
        if(CheckModels(&CurModel2, &NewOrder2, Interactions, j + 1) && !flag){
          // Only need to calculate bounds if this set of models is valid
          if(j > 0){
            Counts2.at(j) = 1;
          
            arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
          
            // Getting lower bound of model without current variable necessarily included
            Bounds.at(j) = GetBound(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2,
                      indices, tol, maxit, pen, 
                      j, xTemp.n_cols, &NewOrder2, LowerBound, &Metrics2);
            Bounds.at(j) += min(*pen);
          }
          else{
            // Don't need to recalculate this
            Bounds.at(0) = LowerBound;
          }
          
          if(Bounds.at(j) > metricCutoff){
            // None of the next models can have a better lower bound
            flag = true;
          }
          else{
            // Updating bound
            Bounds.at(j) += pen->at(NewOrder2.at(j));
          }
        }
      }
      
      // Updating numchecked and potentially updating the best model based on upper models
      (*numchecked) += arma::accu(Counts2);
      UpdateBestMetrics(BestModels, BestMetrics, &NewModels, &Metrics2, cutoff);
      Metrics2.at(0) = UpperMetric;
      
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
      Metrics = reverse(Metrics);
      
      // Recursively calling this function for each new model
      for(unsigned int j = NewOrder2.n_elem - 1; j > 1; j--){
        if(j < revNewOrder2.n_elem - 1){
          UpperModel(revNewOrder2(j + 1)) = 0;
        }
        
        if(Metrics.at(j) > Metrics2.at(j - 1)){
          // If upper model is better than lower model then call backward
        SwitchBackwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &UpperModel, BestModels, 
                                BestMetrics, numchecked, indices, tol, maxit, j - 1, pen, 
                                Bounds.at(j - 1), &revNewOrder2, p, Metrics.at(j), cutoff);
        }else{
          // Creating new current model for next call to forward branch
          arma::ivec CurModel2 = *CurModel;
          CurModel2(revNewOrder2(j)) = 1;
          
          // If lower model is better than upper model then call forward
          SwitchForwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &CurModel2, BestModels, 
                                  BestMetrics, numchecked, indices, tol, maxit, NewOrder2.n_elem - j, pen, 
                                  Bounds.at(j - 1), &NewOrder2, p, Metrics2.at(j - 1), cutoff);
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
                       const arma::imat* Interactions,
                       std::string method, int m, std::string Link, std::string Dist,
                       arma::ivec* CurModel, arma::imat* BestModels, arma::vec* BestMetrics, 
                       unsigned int* numchecked, arma::ivec* indices, double tol, 
                       int maxit, unsigned int cur, const arma::vec* pen, 
                       double LowerBound, arma::uvec* NewOrder, Progress* p, 
                       double LowerMetric, double cutoff){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Getting metric value used to cutoff branches
  double metricCutoff;
  if(cutoff == 0){
    metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
  }
  else{
    metricCutoff = BestMetrics->at(0) + cutoff;
  }
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < metricCutoff){
    // Updating progress
    p->update(4);
    p->print();
    
    // Creating vectors to be used later
    arma::uvec NewOrder2(cur + 1);
    arma::vec Metrics(cur + 1);
    arma::uvec Counts(cur + 1, arma::fill::zeros);
    arma::imat NewModels(CurModel->n_elem, NewOrder2.n_elem, arma::fill::zeros);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder->at(j)) = 0;
      NewModels.col(j) = CurModel2;
      NewOrder2(j) = NewOrder->at(j);
      if(CheckModel(&CurModel2, Interactions)){
        // Only fitting model if it is valid
        Counts.at(j) = 1;
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        Metrics(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
                   method, m, Link, Dist, 
                   tol, maxit, pen);
      }
      else{
        // Assigning infinity to metric value if model is not valid
        Metrics(j) = arma::datum::inf;
      }
    }
    
    // Updating numchecked and potentially updating the best model
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    UpdateBestMetrics(BestModels, BestMetrics, &NewModels, &Metrics, cutoff);
    
    // Getting metric value used to cutoff branches
    if(cutoff == 0){
      metricCutoff = BestMetrics->at(BestMetrics->n_elem - 1);
    }
    else{
      metricCutoff = BestMetrics->at(0) + cutoff;
    }
    
    // Updating best metrics must be done before sorting
    Metrics = Metrics(sorted);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Defining Bounds to store the lower bounds
    arma::vec Bounds(Metrics.n_elem - 1);
    
  // Computing lower bounds
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder2(j)) = 0;
      if(BackwardCheckModels(&CurModel2, &NewOrder2, Interactions, j - 1)){
        // If this set of models is valid then find lower bound
        arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
        
        if(!CheckModel(&CurModel2, Interactions)){
          // Fitting model for upper bound since it wasn't fit earlier
          // Only done when the upper model isn't valid, but the set is valid
          Counts.at(j - 1) = 1;
          Metrics(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &CurModel2,
                  method, m, Link, Dist, 
                  tol, maxit, pen);
        }
        Bounds(j - 1) = BackwardGetBound(X, indices, &CurModel2, &NewOrder2, 
               j, Metrics(j), pen, xTemp.n_cols);
      }
      else{
        // If this set of models is invalid then set bound to infinity so no branching is done
        Bounds.at(j - 1) = arma::datum::inf;
      }
    }
    
    // Updating numchecked
    (*numchecked) += arma::accu(Counts);
    
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
      Lower.fill(arma::datum::inf);
      arma::uvec Counts2(Bounds.n_elem, arma::fill::zeros);
      arma::imat NewModels(CurModel->n_elem, Bounds.n_elem, arma::fill::zeros);
      
      // Fitting lower models
#pragma omp parallel for schedule(dynamic)
      for(int j = revNewOrder2.n_elem - 2; j >= 0; j--){
        if(j > 0 && Bounds.at(j) < metricCutoff){
          // Getting lower model
          /// Only need to fit lower model for j > 0
          arma::ivec NewLowerModel = LowerModel;
          NewLowerModel.elem(revNewOrder2.subvec(j, revNewOrder2.n_elem - 2)) = 
          arma::ivec(revNewOrder2.n_elem - 1 - j, arma::fill::zeros);
          NewModels.col(j) = NewLowerModel;
          
          if(CheckModel(&NewLowerModel, Interactions)){
            // Only fitting model if it is valid
            Counts2.at(j) = 1;
            arma::mat xTemp = GetMatrix(X, &NewLowerModel, indices);
            Lower.at(j) = MetricHelper(&xTemp, XTWX, Y, Offset, indices, &NewLowerModel,
                                         method, m, Link, Dist, 
                                         tol, maxit, pen);
          }
          
          // Tightening lower bound since we fit lower model
          Bounds.at(j) += min(*pen);
        }
      }
      
      // Updating numchecked
      (*numchecked) += arma::accu(Counts2);
      
      // Checking if we need to update bounds
      sorted = sort_index(Lower);
      UpdateBestMetrics(BestModels, BestMetrics, &NewModels, &Lower, cutoff);
      Lower.at(0) = LowerMetric;
      
      // Performing the branching
      if(revNewOrder2.n_elem > 2){
        LowerModel(revNewOrder2(revNewOrder2.n_elem - 2)) = 0;
        for(int j = revNewOrder2.n_elem - 3; j >= 0; j--){
          // Updating lower model for current iteration
          LowerModel(revNewOrder2(j)) = 0;
          
          if(Metrics.at(j) > Lower.at(j)){
            // If Lower model has better metric value than upper model use forward
          SwitchForwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &LowerModel, BestModels, 
                            BestMetrics, numchecked, indices, tol, maxit, j + 1, pen, 
                            Bounds.at(j), &revNewOrder2, p, Metrics.at(j), cutoff);
          }
          else{
            // Creating new CurModel for next set of models
            arma::ivec CurModel2 = *CurModel;
            CurModel2.at(revNewOrder2.at(j)) = 0;
            
            // If upper model has better metric value than lower model use backward
            SwitchBackwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &CurModel2, BestModels, 
                                   BestMetrics, numchecked, indices, tol, maxit, 
                                   revNewOrder2.n_elem - 2 - j, pen, 
                                   Bounds.at(j), &NewOrder2, p, Lower.at(j), cutoff);
          }
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


// Switch Branch and bound method
// [[Rcpp::export]]
List SwitchBranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                             IntegerVector indices, IntegerVector num,
                             IntegerMatrix interactions,
                             std::string method, int m,
                             std::string Link, std::string Dist,
                             unsigned int nthreads, double tol, int maxit, 
                             IntegerVector keep, NumericVector pen,
                             bool display_progress, unsigned int NumBest, 
                             double cutoff){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  arma::imat BestModels(keep.size(), NumBest, arma::fill::zeros);
  arma::vec BestMetrics(NumBest);
  BestMetrics.fill(arma::datum::inf);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  CurModel.replace(1, 0);
  
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  
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
                                       tol, maxit, &Pen);
  
  // Updating BestMetric and BestModel if CurMetric is better than BestMetric
  if(CurMetric < BestMetrics.at(0)){
    BestMetrics.at(0) = CurMetric;
    BestModels.col(0) = CurModel;
  }
  
  // Incrementing numchecked  
  numchecked++;
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  arma::vec Metrics(1);
  Metrics.at(0) = arma::datum::inf;
  LowerBound = GetBound(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel,
                           &Indices, tol, maxit, &Pen, 
                           0, sum(abs(CurModel)), &NewOrder, LowerBound, 
                           &Metrics, true);
  // Defining Upper model
  arma::ivec UpperModel = CurModel;
  for(unsigned int i = 0; i < NewOrder.n_elem; i++){
    UpperModel.at(NewOrder.at(i)) = 1;
  }
  
  // Updating BestMetric and BestModel if metric from upper model is better than BestMetric
  UpdateBestMetrics(&BestModels, &BestMetrics, &UpperModel, &Metrics, cutoff);
  
  // Incrementing numchecked
  numchecked++;
  
  // Starting branching process
  if(Metrics.at(0) < CurMetric && NewOrder.n_elem > 1){
    // Branching forward if lower model has better metric value than upper model
    SwitchForwardBranch(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &CurModel, &BestModels, 
            &BestMetrics, &numchecked, &Indices, tol, maxit, 0, &Pen, 
            LowerBound, &NewOrder, &p, Metrics.at(0), cutoff);
  }else if(NewOrder.n_elem > 1){
    // Branching backward if upper model has better metric value than lower model
    arma::ivec UpperModel = CurModel;
    for(unsigned int i = 0; i < NewOrder.n_elem; i++){
      UpperModel.at(NewOrder.at(i)) = 1;
    }
    
    SwitchBackwardBranch(&X, &XTWX, &Y, &Offset, &Interactions, method, m, Link, Dist, &UpperModel, &BestModels, 
                           &BestMetrics, &numchecked, &Indices, tol, maxit, NewOrder.n_elem - 1, &Pen, 
                           LowerBound, &NewOrder, &p, CurMetric, cutoff);
  }else{
    p.update(2);
  }
  
  // Printing off final update
  p.finalprint();
  
  List FinalList = List::create(Named("bestmodels") = BestModels,
                                Named("numchecked") = numchecked,
                                Named("bestmetrics") = BestMetrics);
  
  // Resetting number of threads if OpenMP is defined
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}


