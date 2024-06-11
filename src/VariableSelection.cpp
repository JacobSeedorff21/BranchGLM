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
        Rcout << "Found best models"  << std::endl << std::endl;
        }
  }
};

// Gets matrix for a given model
arma::mat GetMatrix(const arma::mat* X, arma::ivec* CurModel, 
                    const arma::ivec* Indices){
  
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

// Updates bound if upper model is the same as it was for the previous lower bound, 
// is also a naive bound checked before calculating better lower bound
double UpdateBound(const arma::mat* X, arma::ivec* indices, int cur, double LowerBound, 
                   const arma::vec* pen){
  
  unsigned int k = 0;
  for(unsigned int i = 0; i < indices->n_elem; i++){
    if(indices->at(i) == (cur)){
      k++;
    }
  }
  
  return(LowerBound + pen->at(cur));
}

// Function used to fit models and calculate desired metric
double MetricHelper(const arma::mat* OldX, const arma::mat* XTWX, 
                    const arma::vec* Y, const arma::vec* Offset,
                    const arma::ivec* Indices, const arma::ivec* CurModel,
                    std::string method, 
                    int m, std::string Link, std::string Dist,
                    double tol, int maxit, const arma::vec* pen, 
                    unsigned int cur, arma::mat* betaMat){
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
  arma::mat X = OldX->cols(NewInd);
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
  
  if(Iter < 0){
    return(arma::datum::inf);
  }
  
  arma::vec mu = ParLinkCpp(&X, &beta, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(&X, Y, &mu, Dist);
  double dispersion = GetDispersion(&X, Y, &mu, LogLik, Dist, tol);
  if(dispersion <= 0 || std::isnan(LogLik) || std::isinf(dispersion)){
    return(arma::datum::inf);
  }
  
  if(Dist == "gaussian"){
    double temp = X.n_rows/2. * log(2*M_PI*dispersion);
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
  // Getting beta
  betaMat->submat(NewInd, arma::uvec(1, arma::fill::value(cur))) = beta;
  return(-2 * LogLik + arma::accu(pen->elem(find(*CurModel != 0))));
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

// When doing the process backwards the upper model is already fit, so we just 
// need to use that and minimum number of variables to get bound
double BackwardGetBound(const arma::mat* X, arma::ivec* indices, arma::ivec* CurModel,
                        arma::uvec* NewOrder, unsigned int cur, double metricVal, 
                        const arma::vec* pen){
  
  // If metricVal is infinite, then return -inf as the lower bound
  if(metricVal == arma::datum::inf){
    return(-arma::datum::inf);
  }
  
  // Getting lower model
  arma::ivec lowerModel = *CurModel;
  for(unsigned int i = 0; i < cur; i++){
    lowerModel(NewOrder->at(i)) = 0;
  }
  
  // arma::mat xLower = GetMatrix(X, &lowerModel, indices);
  return(metricVal - arma::accu(pen->elem(find(*CurModel != 0))) + 
         arma::accu(pen->elem(find(lowerModel != 0))));
}


// Fits upper model for a set of models and calculates the bound for the desired metric
double GetBound(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                std::string method, int m, std::string Link, std::string Dist,
                arma::ivec* CurModel, arma::ivec* indices, 
                double tol, int maxit,
                const arma::vec* pen, unsigned int cur,
                arma::uvec* NewOrder, double LowerBound,
                arma::vec* Metrics, arma::mat* betaMat, bool DoAnyways = false){
  
  // Checking if we need to fit model for upper bound and updating bounds if we don't need to
  if(cur == 0 && !DoAnyways){
    return(LowerBound + pen->at(NewOrder->at(cur)));
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
  // Defining Iter
  int Iter;
  
  // Creating matrices for upper model and fitting it
  arma::mat NewXTWX = XTWX->submat(NewInd, NewInd);
  bool UseXTWX = true;
  arma::mat xTemp = X->cols(NewInd);
  arma::vec beta(xTemp.n_cols, arma::fill::zeros);
  
  // Getting initial values
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
  if(Iter < 0){
    return(LowerBound);
  }
  
  // Calculating metric value
  arma::vec mu = ParLinkCpp(&xTemp, &beta, Offset, Link, Dist);
  double LogLik = -ParLogLikelihoodCpp(&xTemp, Y, &mu, Dist);
  double dispersion = GetDispersion(&xTemp, Y, &mu, LogLik, Dist, tol);
  
  // Checking for non-positive dispersion
  if(dispersion <= 0 || std::isinf(dispersion)){
    return(LowerBound);
  }
  
  // Final computation of log-likelihood
  if(Dist == "gaussian"){
    double temp = xTemp.n_rows/2. * log(2*M_PI*dispersion);
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
  
  // Returning previous lower bound if log likelihood is nan
  if(std::isnan(LogLik)){
    return(LowerBound);
  }
  
  // Updating metric value
  Metrics->at(cur) = -2 * LogLik + arma::accu(pen->elem(find(UpperModel != 0)));
  betaMat->submat(NewInd, arma::uvec(1, arma::fill::value(cur))) = beta;
  
  // Getting bound if model converged
  double NewBound = -2 * LogLik + arma::accu(pen->elem(find(*CurModel != 0)));
  return(NewBound);
}