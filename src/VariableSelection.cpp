#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
#include "ParBranchGLMHelpers.h"
using namespace Rcpp;

// Function used to get number of models given a certain maxsize and the number 
// of variables
unsigned long long GetNum(unsigned long long size, unsigned long long max){
  double temp = 0;
  if(max >= size){
    temp = pow(2, size);
  }else{
    double helper = 1;
    temp = 1;
    for(unsigned int i = 1; i <= max; i++){
      helper *= (double)(size - i + 1) /(i);
      temp += round(helper);
    }
  }
  return(temp);
}

// Class to display progress for branch and bound method
class Progress{
private:
  unsigned long long max_size,cur_size;
  double last_print = -0.0000000001; 
  double diff = 0.0000000001;
  bool display_progress;
public:
  Progress(unsigned long long maxnum, bool display):max_size(maxnum), cur_size(0), 
  display_progress(display){}
  void update(unsigned long long num = 1){
    cur_size += num;
  };
  void print(){
    double next_print = 100 * (float)cur_size / (float)max_size;
    if(display_progress && next_print - last_print >= diff){
      Rcout << "Checked " << next_print << "% of all possible models"  << std::endl;
      while(diff <= (next_print - last_print) && diff <= 1.0){
        diff *= 10;
      }
      last_print = next_print;
    }
  }
    void finalprint(){
      double next_print = 100 * (float)cur_size / (float)max_size;
      if(display_progress){
        Rcout << "Checked " << next_print << "% of all possible models"  << std::endl;
        Rcout << "Found best model"  << std::endl << std::endl;
        }
  }
};
// Gets metric for given log likelihood and x matrix
double GetMetric(const arma::mat* X, double logLik, 
                 std::string Dist, std::string metric){
  
  double value = 0;
  unsigned int k = X->n_cols;
  if(Dist == "gaussian" || Dist == "gamma"){
    k++;
  }
  
  if(metric == "AIC"){
    value = -2 * logLik + 2 * k;
  }else if(metric == "AICc"){
    value = -2 * logLik + 2 * k + (2 * k + 2 * pow(k, 2)) / (X->n_rows - k - 1);
  }else if(metric == "BIC"){
    value = -2 * logLik + log(X->n_rows) * k;
  }
  
  return(value);
}
// Gets matrix for a given model
arma::mat GetMatrix(const arma::mat* X, arma::ivec* CurModel, 
                    arma::ivec* Indices){
  
  // Getting size of matrix and the desired columns
  double Size = 0;
  unsigned int k = 0;
  arma::ivec CurCols(Indices->n_elem, arma::fill::zeros);
  for(unsigned int i = 0; i < Indices->n_elem; i++){
    if(CurModel->at(Indices->at(i)) != 0){
      CurCols.at(i) = 1;
      Size++;
    }
  }
  
  // Making desired matrix
  arma::mat xTemp(X->n_rows, Size);
  k = 0;
  for(unsigned int j = 0; j < Indices->n_elem; j++){
    if(CurCols.at(j) == 1){
      xTemp.col(k) = X->col(j);
      k++;
    }
  }
  
  return(xTemp);
}

double BoundHelper(const arma::mat* X, double logLik, 
                   std::string Dist, std::string metric, int minsize){
  
  double value = 0;
  unsigned int k = minsize;
  if(Dist == "gaussian" || Dist == "gamma"){
    k++;
  }
  
  if(metric == "AIC"){
    value = -2 * logLik + 2 * k;
  }else if(metric == "AICc"){
    value = -2 * logLik + 2 * k + (2 * k + 2 * pow(k, 2)) / (X->n_rows - k - 1);
  }else if(metric == "BIC"){
    value = -2 * logLik + log(X->n_rows) * k;
  }
  
  return(value);
}

// Updates bound if upper model is the same as it was for the previous lower bound, 
// is also a naive bound checked before calculating better lower bound
double UpdateBound(const arma::mat* X, arma::ivec* indices, int cur, double LowerBound, 
                   std::string metric, int minsize){
  
  double value = 0;
  unsigned int k = 0;
  for(unsigned int i = 0; i < indices->n_elem; i++){
    if(indices->at(i) == (cur - 1)){
      k++;
    }
  }
  
  if(metric == "AIC"){
    value =  LowerBound  + 2 * k;
  }else if(metric == "AICc"){
    unsigned int newk = minsize - k;
    value = LowerBound + 2 * k - (2 * newk + 2 * pow(newk, 2)) / (X->n_rows - newk - 1) + 
      (2 * minsize + 2 * pow(minsize, 2)) / (X->n_rows - minsize - 1);
  }else if(metric == "BIC"){
    value = LowerBound + log(X->n_rows) * k;
  }
  return(value);
}

// Function used to fit models and calculate desired metric
double MetricHelper(const arma::mat* X, const arma::mat* XTWX, 
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
  PargetInit(&beta, X, &NewXTWX, Y, Offset, Dist, Link, &UseXTWX);
  int Iter;
  
  if(Dist == "gaussian" && Link == "identity"){
    Iter = ParLinRegCppShort(&beta, X, &NewXTWX, Y, Offset);
  }else if(method == "BFGS"){
    Iter = ParBFGSGLMCpp(&beta, X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
    
  }
  else if(method == "LBFGS"){
    Iter = ParLBFGSGLMCpp(&beta, X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, m, UseXTWX);
  }
  else{
    Iter = ParFisherScoringGLMCpp(&beta, X, &NewXTWX, Y, Offset, Link, Dist, tol, maxit, UseXTWX);
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
  return(GetMetric(X, LogLik, Dist, metric));
}

// Function used to check if given model is valid, i.e. if lower order terms are 
// in the model while an interaction term is present
bool CheckModel(const arma::ivec* CurModel, const arma::imat* Interactions){
  for(unsigned int i = 0; i < CurModel->n_elem; i++){
    if(CurModel->at(i) != 0){
      // This variable is included in the current model, so we need to check for 
      // lower order terms if it is an interaction term
      
      for(unsigned int j = 0; j < Interactions->n_rows; j++){
        if(Interactions->at(j, i) != 0 && CurModel->at(j) == 0){
          // Interaction term found without lower order terms included
          return(false);
        }
      }
    }
  }
  return(true);
}

// Function used to check if a set of models is valid, i.e. if any of the models in this 
// set are valid
bool CheckModels(const arma::ivec* CurModel, arma::uvec* NewOrder, 
                 const arma::imat* Interactions, 
                 unsigned int cur){
  
  
  // Getting order for this set of models
  arma::uvec NewOrder2 = NewOrder->subvec(cur, NewOrder->n_elem - 1);
  
  for(unsigned int i = 0; i < CurModel->n_elem; i++){
    if(CurModel->at(i) != 0){
      // This variable is included in the current model, so we need to check for 
      // lower order terms if it is an interaction term
      for(unsigned int j = 0; j < Interactions->n_rows; j++){
        if(Interactions->at(j, i) != 0 && CurModel->at(j) == 0 && all(NewOrder2 != j)){
          // Interaction term found without lower order terms included and they cannot be included
          return(false);
        }
      }
    }
  }
  return(true);
}

// Function used to check if a set of models is valid for backward methods
bool BackwardCheckModels(const arma::ivec* CurModel, arma::uvec* NewOrder, 
                         const arma::imat* Interactions, 
                         unsigned int cur){
  
  arma::uvec NewOrder2 = NewOrder->subvec(0, cur);
  
  for(unsigned int i = 0; i < CurModel->n_elem; i++){
    if(CurModel->at(i) != 0){
      // This variable is included in the current model, so we need to check for 
      // lower order terms if it is an interaction term
      
      for(unsigned int j = 0; j < Interactions->n_rows; j++){
        if(Interactions->at(j, i) != 0 && CurModel->at(j) == 0 && all(NewOrder2 != i)){
          // Interaction term found without lower order terms included
          return(false);
        }
      }
    }
  }
  return(true);
}