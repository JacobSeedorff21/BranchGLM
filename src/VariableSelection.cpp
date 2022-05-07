#include <RcppArmadillo.h>
#include <cmath>
#include "BranchGLMHelpers.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

// Function used to get number of models given a certain maxsize and the number 
// of variables
// [[Rcpp::export]]
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
  if(Dist == "gaussian"){
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
// Fits models and returns metric from them
double MetricHelper(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                    std::string method, 
                    int m, std::string Link, std::string Dist,
                    double tol, std::string metric){
  int Iter;
  
  arma::vec beta(X->n_cols, arma::fill::zeros);
  if(method == "BFGS"){
    Iter = BFGSGLMCpp(&beta, X, Y, Offset, Link, Dist, tol);
    
  }
  else if(method == "LBFGS"){
    Iter = LBFGSGLMCpp(&beta, X, Y, Offset, Link, Dist, tol, m);
  }
  else{
    Iter = FisherScoringGLMCpp(&beta, X, Y, Offset, Link, Dist, tol);
  }
  if(Iter == -2){
    return(arma::datum::inf);
  }
  double dispersion = 1;
  arma::vec mu = LinkCpp(X, &beta, Offset, Link, Dist);
  
  if(Dist == "gaussian"){
    dispersion = arma::accu(pow(*Y - mu, 2)) / (X->n_rows - X->n_cols);
  }
  
  double LogLik = -LogLikelihoodCpp(X, Y, &mu, Dist);
  if(Dist == "gaussian"){
    double temp = X->n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  LogFact(Y);
  }
  return(GetMetric(X, LogLik, Dist, metric));
}

// Given a current model, this finds the best variable to add to the model
void add1(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
          std::string method, int m, std::string Link, std::string Dist,
          arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
          unsigned int* numchecked, bool* flag, arma::ivec* order, unsigned int i,
          arma::ivec* indices, double tol, std::string metric){
  
  for(unsigned int j = 0; j < CurModel->n_elem; j++){
    if(CurModel->at(j) == 0){
      (*numchecked)++;
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 1;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      double NewMetric = MetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
                                      tol, metric);
      
      if(NewMetric < *BestMetric){
        *BestModel = CurModel2;
        *BestMetric = NewMetric;
        *flag = false;
        order->at(i) = j;
      }
    }
  }
}

// Performs forward selection
// [[Rcpp::export]]
List ForwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                IntegerVector indices, IntegerVector num,
                std::string method, int m,
                std::string Link, std::string Dist,
                unsigned int nthreads, double tol, 
                bool intercept, IntegerVector keep, 
                unsigned int steps,std::string metric){
  
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
                                   tol, metric);
  unsigned int numchecked = 0;
  
  IntegerVector order(CurModel.n_elem, -1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
  omp_set_num_threads(nthreads);
  
  for(unsigned int i = 0; i < steps; i++){
    
    bool flag = true;
    CurModel = BestModel;
    add1(&X, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
         &BestMetric, &numchecked, &flag, &Order, i, &Indices, tol, metric);
    
    if(flag){
      break;
    }
  }
  // Getting x matrix for best model found
  
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                               nthreads, tol);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("order") = order,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  omp_set_num_threads(1);
  
  return(FinalList);
}

// Given a current model, this finds the best variable to remove
void drop1(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
           std::string method, int m, std::string Link, std::string Dist,
           arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
           unsigned int* numchecked, bool* flag, arma::ivec* order, unsigned int i,
           bool intercept, arma::ivec* indices, double tol, std::string metric){
  
  unsigned int j = 0;
  
  if(intercept){
    j++;
  }
  
  for(; j < CurModel->n_elem; j++){
    if(CurModel->at(j) == 1){
      (*numchecked)++;
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(j) = 0;
      arma::mat xTemp = GetMatrix(X, &CurModel2, indices);
      double NewMetric = MetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
                                      tol, metric);
      
      if(NewMetric < *BestMetric){
        *BestModel = CurModel2;
        *BestMetric = NewMetric;
        *flag = false;
        order->at(i) = j;
      }
    }
  }
}

// Performs backward elimination
// [[Rcpp::export]]
List BackwardCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                 IntegerVector indices, IntegerVector num,
                 std::string method, int m,
                 std::string Link, std::string Dist,
                 unsigned int nthreads, double tol, bool intercept,
                 IntegerVector keep, unsigned int steps, std::string metric){
  
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, tol, 
                            metric);
  
  unsigned int numchecked = 0;
  IntegerVector order(CurModel.n_elem, -1);
  arma::ivec Order(order.begin(), order.size(), false, true);
  
  
  omp_set_num_threads(nthreads);
  
  for(unsigned int i = 0; i < steps; i++){
    
    bool flag = true;
    CurModel = BestModel;
    drop1(&X, &Y, &Offset, method, m, Link, Dist, &CurModel, &BestModel, 
          &BestMetric, &numchecked, &flag, &Order, i, intercept, &Indices, tol, metric);
    
    if(flag){
      break;
    }
  }
  
  // Getting x matrix for best model found
  
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                               nthreads, tol);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("order") = order,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  omp_set_num_threads(1);
  
  return(FinalList);
}


double BoundHelper(const arma::mat* X, double logLik, 
                   std::string Dist, std::string metric, int minsize){
  
  double value = 0;
  unsigned int k = minsize;
  if(Dist == "gaussian"){
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

double GetBound(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                std::string method, int m, std::string Link, std::string Dist,
                arma::ivec* CurModel,  arma::ivec* indices, double tol, 
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
    Iter = BFGSGLMCpp(&beta, &xTemp, Y, Offset, Link, Dist, tol);
    
  }
  else if(method == "LBFGS"){
    Iter = LBFGSGLMCpp(&beta, &xTemp, Y, Offset, Link, Dist, tol, m);
  }
  else{
    Iter = FisherScoringGLMCpp(&beta, &xTemp, Y, Offset, Link, Dist, tol);
  }
  if(Iter < 0){
    return(LowerBound);
  }
  
  double dispersion = 1;
  arma::vec mu = LinkCpp(&xTemp, &beta, Offset, Link, Dist);
  
  if(Dist == "gaussian"){
    dispersion = arma::accu(pow(*Y - mu, 2)) / (xTemp.n_rows - xTemp.n_cols);
  }
  
  double LogLik = -LogLikelihoodCpp(&xTemp, Y, &mu, Dist);
  if(Dist == "gaussian"){
    double temp = xTemp.n_rows/2 * log(2*M_PI*dispersion);
    LogLik = LogLik / dispersion - temp;
  }
  else if(Dist == "poisson"){
    LogLik -=  LogFact(Y);
  }
  return(BoundHelper(X, LogLik, Dist, metric, minsize));
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

void Branch(const arma::mat* X, const arma::vec* Y, const arma::vec* Offset,
                 std::string method, int m, std::string Link, std::string Dist,
                 arma::ivec* CurModel, arma::ivec* BestModel, double* BestMetric, 
                 unsigned int* numchecked,arma::ivec* indices, double tol, 
                 int maxsize, unsigned int cur, std::string metric, 
                 double LowerBound, arma::uvec* NewOrder, Progress* p,
                 bool update = false){
  
  arma::mat xTemp = GetMatrix(X, CurModel, indices);
  // Updating lower bound, first checking naive bound, then calculating better bound
  LowerBound = UpdateBound(X, indices, NewOrder->at(cur), LowerBound, metric, xTemp.n_cols);
  if(update && LowerBound < *BestMetric){
    (*numchecked)++;
    LowerBound = GetBound(X, Y, Offset, method, m, Link, Dist, CurModel,
                                indices, tol, metric, 
                                cur, xTemp.n_cols, NewOrder, LowerBound);
  }
  if(LowerBound < *BestMetric){
    if(maxsize > 0){ 
      if(NewOrder->n_elem - cur > 1){
        p->update(1);
      }
      p->update(1);
      p->print();
      arma::uvec NewOrder2(NewOrder->n_elem - cur);
      arma::vec Metrics(NewOrder->n_elem - cur);
      unsigned int k = 0;
      // Finding metrics of all current models
      for(unsigned int j = cur; j < NewOrder->n_elem; j++){
        (*numchecked)++;
        arma::ivec CurModel2 = *CurModel;
        CurModel2.at(NewOrder->at(j)) = 1;
        xTemp = GetMatrix(X, &CurModel2, indices);
        NewOrder2.at(k) = NewOrder->at(j);
        Metrics.at(k) = MetricHelper(&xTemp, Y, Offset, method, m, Link, Dist, 
                   tol, metric);
          
        if(Metrics.at(k) < *BestMetric){
          *BestModel = CurModel2;
          *BestMetric = Metrics.at(k);
        }
        k++;
    }
      // Recursively calling this function for each model
    NewOrder2 = NewOrder2(sort_index(Metrics));
    for(unsigned int j = 0; j < NewOrder2.n_elem - 1; j++){
      arma::ivec CurModel2 = *CurModel;
      CurModel2.at(NewOrder2.at(j)) = 1;
      if(j == 0){
        Branch(X, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
                      BestMetric, numchecked, indices, tol, maxsize - 1, j + 1, metric, 
                      LowerBound, &NewOrder2, p);
      }
      else{
        Branch(X, Y, Offset, method, m, Link, Dist, &CurModel2, BestModel, 
                    BestMetric, numchecked, indices, tol, maxsize - 1, j + 1, metric, 
                    LowerBound, &NewOrder2, p, true);
        }         
      }
    }
  }
  else{
    p->update(GetNum(NewOrder->n_elem - cur, maxsize));
    p->print();
  }
}


// [[Rcpp::export]]

List BranchAndBoundCpp(NumericMatrix x, NumericVector y, NumericVector offset, 
                            IntegerVector indices, IntegerVector num,
                            std::string method, int m,
                            std::string Link, std::string Dist,
                            unsigned int nthreads, double tol, 
                            IntegerVector keep, int maxsize, std::string metric,
                            bool display_progress = true){
  
  const arma::mat X(x.begin(), x.rows(), x.cols(), false, true);
  const arma::vec Y(y.begin(), y.size(), false, true);
  const arma::vec Offset(offset.begin(), offset.size(), false, true);
  arma::ivec BestModel(keep.begin(), keep.size(), false, true);
  arma::ivec Indices(indices.begin(), indices.size(), false, true);
  arma::ivec CurModel = BestModel;
  arma::mat xTemp = GetMatrix(&X, &CurModel, &Indices);
  double BestMetric = arma::datum::inf;
  BestMetric = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, tol, 
                            metric);
  unsigned int numchecked = 1;
  unsigned int size = 0;
  omp_set_num_threads(nthreads);
  
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
  for(unsigned int j = 0; j < CurModel.n_elem; j++){
    if(CurModel.at(j) == 0){
      numchecked++;
      arma::ivec CurModel2 = CurModel;
      CurModel2.at(j) = 1;
      xTemp = GetMatrix(&X, &CurModel2, &Indices);
      NewOrder.at(k) = j;
      Metrics.at(k) = MetricHelper(&xTemp, &Y, &Offset, method, m, Link, Dist, 
                 tol, metric);
      
      if(Metrics.at(k) < BestMetric){
        BestModel = CurModel2;
        BestMetric = Metrics.at(k);
      }
      k++;
    }
  }
  
  NewOrder = NewOrder(sort_index(Metrics));
  double LowerBound = -arma::datum::inf;
  LowerBound = GetBound(&X, &Y, &Offset, method, m, Link, Dist, &CurModel,
                                    &Indices, tol, metric, 
                                    0, 0, &NewOrder, LowerBound);
  if(NewOrder.n_elem > 1){
    p.update(1);
  }
  for(unsigned int j = 0; j < NewOrder.n_elem - 1; j++){
    arma::ivec CurModel2 = CurModel;
    CurModel2.at(NewOrder.at(j)) = 1;
    if(j == 0){
      Branch(&X, &Y, &Offset, method, m, Link, Dist, &CurModel2, &BestModel, 
             &BestMetric, &numchecked, &Indices, tol, maxsize - 1, j + 1, metric, 
             LowerBound, &NewOrder, &p);
    }
    else{
      Branch(&X, &Y, &Offset, method, m, Link, Dist, &CurModel2, &BestModel, 
                  &BestMetric, &numchecked, &Indices, tol, maxsize - 1, j + 1, metric, 
                  LowerBound, &NewOrder, &p, true);
    }
                
  }
  
  p.finalprint();
  
  // Getting x matrix for best model found
  
  const arma::mat Finalx = GetMatrix(&X, &BestModel, &Indices);
  
  List helper =  BranchGLMFitCpp(&Finalx, &Y, &Offset, method, m, Link, Dist, 
                               nthreads, tol);
  
  List FinalList = List::create(Named("fit") = helper,
                                Named("model") = keep,
                                Named("numchecked") = numchecked,
                                Named("bestmetric") = BestMetric);
  
  omp_set_num_threads(1);
  
  return(FinalList);
}
