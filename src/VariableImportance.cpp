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

void ImportanceUpdateBestMetrics(arma::imat* WithModels,
                                 arma::imat* WithoutModels,
                                 arma::vec* WithMetrics,
                                 arma::vec* WithoutMetrics,
                                 const arma::uvec* Vars,
                                 arma::imat* Models, 
                                 arma::vec* Metrics){
 // Updating metrics
 for(unsigned int i = 0; i < Metrics->n_elem; i++){
   // Updating withmetrics
   for(unsigned int j = 0; j < Vars->n_elem; j++){
     if(Metrics->at(i) < WithMetrics->at(j) && Models->at(Vars->at(j), i) != 0){
       WithModels->col(j) = Models->col(i);
       WithMetrics->at(j) = Metrics->at(i);
     }
   }
   
   // Updating withoutmetrics
   for(unsigned int j = 0; j < Vars->n_elem; j++){
     if(Metrics->at(i) < WithoutMetrics->at(j) && Models->at(Vars->at(j), i) == 0){
       WithoutModels->col(j) = Models->col(i);
       WithoutMetrics->at(j) = Metrics->at(i);
     }
   }
 }
}

// Defining backward branching function for Importance method
// Forward declaration so this can be called by the forward Importance branch
void ImportanceBackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                              const arma::imat* Interactions, 
                              std::string method, int m, std::string Link, std::string Dist,
                              arma::ivec* CurModel, 
                              arma::imat* WithModels, arma::vec* WithMetrics, 
                              arma::imat* WithoutModels, arma::vec* WithoutMetrics, 
                              const arma::uvec* Vars, 
                              unsigned int* numchecked, arma::ivec* indices, double tol, 
                              int maxit, unsigned int cur, const arma::vec* pen, 
                              double LowerBound, double LowerMetric,
                              arma::uvec* NewOrder, Progress* p);

// Function used to performing branching for forward part of Importance branch
void ImportanceForwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                             const arma::imat* Interactions, 
                             std::string method, int m, std::string Link, std::string Dist,
                             arma::ivec* CurModel, 
                             arma::imat* WithModels, arma::vec* WithMetrics, 
                             arma::imat* WithoutModels, arma::vec* WithoutMetrics, 
                             const arma::uvec* Vars, 
                             unsigned int* numchecked, arma::ivec* indices, double tol, 
                             int maxit, unsigned int cur, const arma::vec* pen, 
                             double LowerBound, double UpperMetric, 
                             arma::uvec* NewOrder, Progress* p){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Getting metric value used to cut off branches
  double metricCutoff = -arma::datum::inf;
  for(unsigned int i = 0; i < Vars->n_elem; i++){
    if(CurModel->at(Vars->at(i)) == 1){
      metricCutoff = std::max(metricCutoff, WithMetrics->at(i));
    }else{
      metricCutoff = std::max(metricCutoff, WithoutMetrics->at(i));
      if(any(NewOrder->subvec(cur, NewOrder->n_elem - 1) == Vars->at(i))){
        metricCutoff = std::max(metricCutoff, WithMetrics->at(i));
      }
    }
  }
  
  // Continuing branching process if lower bound is smaller than the best observed metric
  if(LowerBound < metricCutoff){
    
    // Updating progress
    p->update(4);
    p->print();
    
    // Creating vectors to be used later on
    arma::uvec NewOrder2(NewOrder->n_elem - cur);
    arma::vec Metrics(NewOrder->n_elem - cur);
    arma::uvec Counts(NewOrder->n_elem - cur, arma::fill::zeros);
    arma::mat NewModels(X->n_cols, NewOrder2.n_elem, arma::fill::zeros);
    arma::imat NewModels2(CurModel->n_elem, NewOrder2.n_elem, arma::fill::zeros);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder->at(j + cur)) = 1;
      NewOrder2.at(j) = NewOrder->at(j + cur);
      NewModels2.col(j) = CurModel2;
      if(CheckModel(&CurModel2, Interactions)){
        // Only fitting model if it is valid
        Counts.at(j) = 1;
        Metrics.at(j) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel2, 
                   method, m, Link, Dist, tol, maxit, pen, j, &NewModels);
      }
      else{
        // If model is not valid then set metric value to infinity
        Metrics.at(j) = arma::datum::inf;
      }
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += arma::accu(Counts);
    ImportanceUpdateBestMetrics(WithModels, WithoutModels, 
                                WithMetrics, WithoutMetrics,
                                Vars,
                                &NewModels2, &Metrics);
    
    // Updating best metrics must be done before sorting
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Getting metric value used to cut off branches
    
    // Only find bounds and perform branching if there is at least 1 element to branch on
    if(NewOrder2.n_elem > 1){
      
      // Creating vector to store lower bounds
      arma::vec Bounds(NewOrder2.n_elem - 1);
      Bounds.fill(-arma::datum::inf);
      arma::vec Bounds2 = Bounds;
      Bounds.fill(arma::datum::inf);
      arma::uvec Counts2(NewOrder2.n_elem - 1, arma::fill::zeros);
      arma::vec Metrics2(Bounds.n_elem);
      Metrics2.fill(arma::datum::inf);
      NewModels = arma::mat(X->n_cols, Bounds.n_elem, arma::fill::zeros);
      NewModels2 = arma::imat(CurModel->n_elem, Bounds.n_elem, arma::fill::zeros);
      
      // Getting lower bounds
#pragma omp parallel for schedule(dynamic)
      for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
        arma::ivec CurModel2 = *CurModel;
        bool flag = false;
        for(unsigned int i = 0; i < j; i++){
          if(Bounds2.at(i) + min(*pen) > metricCutoff){
            flag = true;
            Bounds2.at(j) = Bounds2.at(i);
            break;
          }
        }
        
        if(!flag && CheckModels(&CurModel2, &NewOrder2, Interactions, j + 1)){
          // Only need to calculate bounds if this set of models is valid
          if(j > 0){
            Counts2.at(j) = 1;
            
            // Creating vector for the upper model
            arma::ivec UpperModel = CurModel2;
            for(unsigned int i = j; i < NewOrder2.n_elem; i++){
              UpperModel.at(NewOrder2.at(i)) = 1;
            }
            
            // Getting lower bound of model without current variable necessarily included
            NewModels2.col(j) = UpperModel;
            Bounds.at(j) = GetBound(X, XTWX, Y, Offset, method, m, Link, Dist, &CurModel2,
                      indices, tol, maxit, pen, j, &NewOrder2, 
                      LowerBound, &Metrics2, &NewModels);
            Bounds.at(j) += min(*pen);
            if(std::isinf(Bounds.at(j))){
              Bounds.at(j) = LowerBound;
            } 
            Bounds2.at(j) = Bounds.at(j);
          } 
          else{
            // Don't need to recalculate this
            Bounds.at(0) = LowerBound;
            Bounds2.at(0) = Bounds.at(0);
          } 
          // Updating bound
          Bounds.at(j) += pen->at(NewOrder2.at(j));
        }else{ 
          Bounds.at(j) = arma::datum::inf;
        } 
      }
      
      // Updating numchecked
      (*numchecked) += arma::accu(Counts2);
      ImportanceUpdateBestMetrics(WithModels, WithoutModels, 
                                  WithMetrics, WithoutMetrics,
                                  Vars,
                                  &NewModels2, &Metrics2);
      Metrics2.at(0) = UpperMetric;
      
      // Checking for user interrupt
      checkUserInterrupt();
      
      // Defining upper model to be used for possible Importance to backward branching
      /// Reversing new order for possible Importance to backward branching
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
          ImportanceBackwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &UpperModel, 
                                   WithModels, WithMetrics, WithoutModels, WithoutMetrics,
                                   Vars, numchecked, indices, tol, maxit, j - 1, pen, 
                                   Bounds.at(j - 1), Metrics.at(j), &revNewOrder2, p);
        }else{
          // Creating new current model for next call to forward branch
          arma::ivec CurModel2 = *CurModel;
          CurModel2(revNewOrder2(j)) = 1;
          
          // If lower model is better than upper model then call forward
          ImportanceForwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &CurModel2, 
                                  WithModels, WithMetrics, WithoutModels, WithoutMetrics,
                                  Vars, numchecked, indices, tol, maxit, NewOrder2.n_elem - j, pen, 
                                  Bounds.at(j - 1), Metrics2.at(j - 1), &NewOrder2, p);
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
void ImportanceBackwardBranch(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                              const arma::imat* Interactions, 
                              std::string method, int m, std::string Link, std::string Dist,
                              arma::ivec* CurModel, 
                              arma::imat* WithModels, arma::vec* WithMetrics, 
                              arma::imat* WithoutModels, arma::vec* WithoutMetrics, 
                              const arma::uvec* Vars, 
                              unsigned int* numchecked, arma::ivec* indices, double tol, 
                              int maxit, unsigned int cur, const arma::vec* pen, 
                              double LowerBound, double LowerMetric,
                              arma::uvec* NewOrder, Progress* p){
  
  // Checking for user interrupt
  checkUserInterrupt();
  
  // Getting metric value used to cut off branches
  double metricCutoff = -arma::datum::inf;
  for(unsigned int i = 0; i < Vars->n_elem; i++){
    if(CurModel->at(Vars->at(i)) == 1){
      metricCutoff = std::max(metricCutoff, WithMetrics->at(i));
      if(any(NewOrder->subvec(0, cur) == Vars->at(i))){
        metricCutoff = std::max(metricCutoff, WithoutMetrics->at(i));
      }
    }else{
      metricCutoff = std::max(metricCutoff, WithoutMetrics->at(i));
    }
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
    arma::mat NewModels(X->n_cols, NewOrder2.n_elem, arma::fill::zeros);
    arma::imat NewModels2(CurModel->n_elem, NewOrder2.n_elem, arma::fill::zeros);
    
    // Getting metric values
#pragma omp parallel for schedule(dynamic) 
    for(unsigned int j = 0; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder->at(j)) = 0;
      NewOrder2(j) = NewOrder->at(j);
      NewModels2.col(j) = CurModel2;
      if(CheckModel(&CurModel2, Interactions)){
        // Only fitting model if it is valid
        Counts.at(j) = 1;
        Metrics(j) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel2,
                method, m, Link, Dist, 
                tol, maxit, pen, j, &NewModels);
      }
      else{
        // Assigning infinity to metric value if model is not valid
        Metrics(j) = arma::datum::inf;
      }
    }
    
    // Updating numchecked and potentially updating the best model
    *numchecked += arma::accu(Counts);
    ImportanceUpdateBestMetrics(WithModels, WithoutModels, 
                                WithMetrics, WithoutMetrics,
                                Vars,
                                &NewModels2, &Metrics);
    
    // Updating best metrics must be done before sorting
    arma::uvec sorted = sort_index(Metrics);
    NewOrder2 = NewOrder2(sorted);
    Metrics = Metrics(sorted);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Defining Bounds to store the lower bounds
    arma::vec Bounds(Metrics.n_elem - 1);
    arma::uvec Counts2(Metrics.n_elem - 1, arma::fill::zeros);
    
    // Computing lower bounds
#pragma omp parallel for schedule(dynamic)
    for(unsigned int j = 1; j < NewOrder2.n_elem; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2(NewOrder2(j)) = 0;
      if(BackwardCheckModels(&CurModel2, &NewOrder2, Interactions, j - 1)){
        // If this set of models is valid then find lower bound
        
        if(!CheckModel(&CurModel2, Interactions)){
          // Fitting model for upper bound since it wasn't fit earlier
          // Only done when the upper model isn't valid, but the set is valid
          Counts.at(j - 1) = 1;
          Metrics(j) = MetricHelper(X, XTWX, Y, Offset, indices, &CurModel2,
                  method, m, Link, Dist, 
                  tol, maxit, pen, j, &NewModels);
        }
        Bounds(j - 1) = BackwardGetBound(X, indices, &CurModel2, &NewOrder2, 
               j, Metrics(j), pen);
      }
      else{
        // If this set of models is invalid then set bound to infinity so no branching is done
        Bounds.at(j - 1) = arma::datum::inf;
      }
    }
    
    // Updating numchecked
    (*numchecked) += arma::accu(Counts2);
    
    // Checking for user interrupt
    checkUserInterrupt();
    
    // Defining lower model for Importance
    arma::uvec revNewOrder2 = reverse(NewOrder2);
    arma::ivec LowerModel = *CurModel;
    LowerModel(revNewOrder2(revNewOrder2.n_elem - 1)) = 0;
    
    // Reversing for possible Importance to forward branching
    Bounds = reverse(Bounds);
    Metrics = reverse(Metrics);
    
    
    // Lower models and branching only need to be done if revNewOrder2.n_elem > 1
    if(revNewOrder2.n_elem > 1){
      // Creating vector to store metric values from lower models
      arma::vec Lower(Bounds.n_elem);
      Lower.fill(arma::datum::inf);
      arma::uvec Counts2(Bounds.n_elem, arma::fill::zeros);
      arma::mat NewModels(X->n_cols, Bounds.n_elem, arma::fill::zeros);
      arma::imat NewModels2(CurModel->n_elem, Bounds.n_elem, arma::fill::zeros);
      
      // Fitting lower models
#pragma omp parallel for schedule(dynamic)
      for(int j = revNewOrder2.n_elem - 2; j >= 0; j--){
        if(j > 0 && Bounds.at(j) < metricCutoff){
          // Getting lower model
          /// Only need to fit lower model for j > 0
          arma::ivec NewLowerModel = LowerModel;
          NewLowerModel.elem(revNewOrder2.subvec(j, revNewOrder2.n_elem - 2)) = 
            arma::ivec(revNewOrder2.n_elem - 1 - j, arma::fill::zeros);
          NewModels2.col(j) = NewLowerModel;
          
          if(CheckModel(&NewLowerModel, Interactions)){
            // Only fitting model if it is valid
            Counts2.at(j) = 1;
            Lower.at(j) = MetricHelper(X, XTWX, Y, Offset, indices, &NewLowerModel,
                     method, m, Link, Dist, tol, maxit, pen, j, &NewModels);
          }
          
          // Tightening lower bound since we fit lower model
          Bounds.at(j) += min(*pen);
        }
      }
      
      // Updating numchecked
      (*numchecked) += arma::accu(Counts2);
      
      // Checking if we need to update bounds
      ImportanceUpdateBestMetrics(WithModels, WithoutModels, 
                                  WithMetrics, WithoutMetrics,
                                  Vars,
                                  &NewModels2, &Lower);
      Lower.at(0) = LowerMetric;
      
      // Performing the branching
      if(revNewOrder2.n_elem > 2){
        LowerModel(revNewOrder2(revNewOrder2.n_elem - 2)) = 0;
        for(int j = revNewOrder2.n_elem - 3; j >= 0; j--){
          // Updating lower model for current iteration
          LowerModel(revNewOrder2(j)) = 0;
          
          if(Metrics.at(j) > Lower.at(j)){
            // If Lower model has better metric value than upper model use forward
            ImportanceForwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &LowerModel, 
                                    WithModels, WithMetrics, WithoutModels, WithoutMetrics,
                                    Vars, numchecked, indices, tol, maxit, j + 1, pen, 
                                    Bounds.at(j), Metrics.at(j), &revNewOrder2, p);
          }
          else{
            // Creating new CurModel for next set of models
            arma::ivec CurModel2 = *CurModel;
            CurModel2.at(revNewOrder2.at(j)) = 0;
            
            // If upper model has better metric value than lower model use backward
            ImportanceBackwardBranch(X, XTWX, Y, Offset, Interactions, method, m, Link, Dist, &CurModel2, 
                                     WithModels, WithMetrics, WithoutModels, WithoutMetrics, 
                                     Vars, numchecked, indices, tol, maxit, 
                                     revNewOrder2.n_elem - 2 - j, pen, 
                                     Bounds.at(j), Lower.at(j), &NewOrder2, p);
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

// Branch and bound method
// [[Rcpp::export]]
List SwitchVariableImportanceCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                           IntegerVector indices, IntegerVector num,
                           IntegerMatrix interactions,
                           IntegerMatrix withmodels, NumericVector withmetrics, 
                           IntegerMatrix withoutmodels, NumericVector withoutmetrics,
                           std::string method, int m,
                           std::string Link, std::string Dist,
                           unsigned int nthreads, double tol, int maxit, 
                           IntegerVector keep, NumericVector pen,
                           bool display_progress){
  
  // Creating necessary vectors/matrices
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  const arma::vec Pen(pen.begin(), pen.size(), false, true);
  const arma::imat Interactions(interactions.begin(), interactions.rows(), 
                                interactions.cols(), false, true);
  
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel(keep.begin(), keep.size(), false, true);
  CurModel.replace(1, 0);
  const arma::uvec Vars = find(CurModel == 0);
  
  // Making stuff for models with variables included
  arma::imat WithModels(withmodels.begin(), withmodels.rows(), 
                        withmodels.cols(), false, true);
  arma::vec WithMetrics(withmetrics.begin(), withmetrics.size(), false, true);
  
  // Making stuff for models without variables included
  arma::imat WithoutModels(withoutmodels.begin(), withoutmodels.rows(), 
                           withoutmodels.cols(), false, true);
  arma::vec WithoutMetrics(withoutmetrics.begin(), withoutmetrics.size(), false, true);
  
  // Getting X'WX
  arma::mat XTWX = X.t() * X;
  arma::mat XY = X.t() * (Y - Offset);
  
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
  Progress p(GetNum(size, size), display_progress);
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
  arma::mat matModel(X.n_cols, 1, arma::fill::zeros);
  double CurMetric = MetricHelper(&X, &XTWX, &Y, &Offset, &Indices, 
                                  &CurModel, method, m, Link, Dist, 
                                  tol, maxit, &Pen, 0, &matModel);
  
  // Updating BestMetric is CurMetric is better
  for(unsigned int i = 0; i < Vars.n_elem; i++){
    if(WithoutMetrics.at(i) > CurMetric){
      WithoutMetrics.at(i) = CurMetric;
      WithoutModels.col(i) = CurModel;
    }
  }
  
  // Finding initial lower bound
  double LowerBound = -arma::datum::inf;
  arma::vec Metrics(1);
  Metrics.at(0) = arma::datum::inf;
  LowerBound = GetBound(&X, &XTWX, &Y, &Offset, method, m, Link, Dist, &CurModel,
                        &Indices, tol, maxit, &Pen, 0, &NewOrder, LowerBound, 
                        &Metrics, &matModel, true);
  
  // Incrementing numchecked
  numchecked++;
  
  // Updating BestMetric is Metrics.at(0) is better
  if(Metrics.is_finite()){
    arma::ivec CurModel2 = CurModel;
    CurModel2.replace(0, 1);
    for(unsigned int i = 0; i < Vars.n_elem; i++){
      if(WithMetrics.at(i) > Metrics.at(0)){
        WithMetrics.at(i) = Metrics.at(0);
        WithModels.col(i) = CurModel2;
      }
    }
  }
  
  // Starting branching process
  if(Metrics.at(0) < CurMetric && NewOrder.n_elem > 1){
    // Branching forward if lower model has better metric value than upper model
    ImportanceForwardBranch(&X, &XTWX, &Y, &Offset, &Interactions, method, 
                                  m, Link, Dist, &CurModel, 
                                  &WithModels, &WithMetrics, 
                                  &WithoutModels, &WithoutMetrics,
                                  &Vars, &numchecked, &Indices, tol, maxit, 0, &Pen, 
                                  LowerBound, Metrics.at(0), &NewOrder, &p);
    
  }else if(NewOrder.n_elem > 1){
    // Branching backward if upper model has better metric value than lower model
    arma::ivec UpperModel = CurModel;
    for(unsigned int i = 0; i < NewOrder.n_elem; i++){
      UpperModel.at(NewOrder.at(i)) = 1;
    }
    
    ImportanceBackwardBranch(&X, &XTWX, &Y, &Offset, &Interactions, method, 
                                   m, Link, Dist, &UpperModel, 
                                   &WithModels, &WithMetrics, 
                                   &WithoutModels, &WithoutMetrics,
                                   &Vars, &numchecked, &Indices, tol, 
                                   maxit, NewOrder.n_elem - 1, &Pen, 
                                   LowerBound, CurMetric, &NewOrder, &p);
  }
  
  // Printing off final update
  p.finalprint();
  
  List FinalList = List::create(Named("withmodels") = WithModels,
                                Named("withoutmodels") = WithoutModels,
                                Named("numchecked") = numchecked,
                                Named("withmetrics") = WithMetrics,
                                Named("withoutmetrics") = WithoutMetrics);
  
  // Resetting number of threads if OpenMP is available
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  
  return(FinalList);
}
