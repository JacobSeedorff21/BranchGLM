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
        Rcout << "Found best models"  << std::endl << std::endl;
        }
  }
};

arma::mat GetMatrix(const arma::mat* X, arma::ivec* CurModel, 
                    const arma::ivec* Indices);

double UpdateBound(const arma::mat* X, arma::ivec* indices, int cur, double LowerBound, 
                   const arma::vec* pen);

double MetricHelper(const arma::mat* X, const arma::mat* XTWX, 
                    const arma::vec* Y, const arma::vec* Offset,
                    const arma::ivec* Indices, const arma::ivec* CurModel,
                    std::string method, 
                    int m, std::string Link, std::string Dist,
                    double tol, int maxit, const arma::vec* pen);

bool CheckModel(const arma::ivec* CurModel, const arma::imat* Interactions);

bool CheckModels(const arma::ivec* CurModel, arma::uvec* NewOrder, 
                 const arma::imat* Interactions, 
                 unsigned int cur);

bool BackwardCheckModels(const arma::ivec* CurModel, arma::uvec* NewOrder, 
                         const arma::imat* Interactions, 
                         unsigned int cur);

double BackwardGetBound(const arma::mat* X, arma::ivec* indices, arma::ivec* CurModel,
                        arma::uvec* NewOrder, unsigned int cur, double metricVal, 
                        const arma::vec* pen, unsigned int maxsize);

double GetBound(const arma::mat* X, const arma::mat* XTWX, const arma::vec* Y, const arma::vec* Offset,
                std::string method, int m, std::string Link, std::string Dist,
                arma::ivec* CurModel,  arma::ivec* indices, 
                double tol, int maxit,
                const arma::vec* pen, unsigned int cur, int minsize,
                arma::uvec* NewOrder, double LowerBound,
                arma::vec* Metrics, bool DoAnyways = false);

#endif