#ifndef VariableSelection_H
#define VariableSelection_H

#include <RcppArmadillo.h>
using namespace Rcpp;

unsigned long long GetNum(unsigned long long size, unsigned long long max);

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

double GetMetric(const arma::mat* X, double logLik, 
                 std::string Dist, std::string metric);

arma::mat GetMatrix(const arma::mat* X, arma::ivec* CurModel, 
                    arma::ivec* Indices);

double BoundHelper(const arma::mat* X, double logLik, 
                   std::string Dist, std::string metric, int minsize);

double UpdateBound(const arma::mat* X, arma::ivec* indices, int cur, double LowerBound, 
                   std::string metric, int minsize);

double MetricHelper(const arma::mat* X, const arma::mat* XTWX, 
                    const arma::vec* Y, const arma::vec* Offset,
                    const arma::ivec* Indices, const arma::ivec* CurModel,
                    std::string method, 
                    int m, std::string Link, std::string Dist,
                    double tol, int maxit, std::string metric);

bool CheckModel(const arma::ivec* CurModel, const arma::imat* Interactions);

bool CheckModels(const arma::ivec* CurModel, arma::uvec* NewOrder, 
                 const arma::imat* Interactions, 
                 unsigned int cur);

bool BackwardCheckModels(const arma::ivec* CurModel, arma::uvec* NewOrder, 
                         const arma::imat* Interactions, 
                         unsigned int cur);

#endif