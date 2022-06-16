#include <RcppArmadillo.h>
#include <cmath>
#ifdef _OPENMP
# include <omp.h>
#endif
using namespace Rcpp;

// Use this for parallel linear regression
arma::mat ParXTX(const arma::mat* x){
  
  arma::mat FinalMat(x->n_cols, x->n_cols);
  
  // Finding X'X
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int i = 0; i < x->n_cols; i+=2){
    if(i == x->n_cols - 1){
      arma::vec temp = x->col(i);
      for(unsigned int j = i; j < x->n_cols; j++){
        double tempnum = 0;
        if(j == i){
          for(unsigned int k = 0; k < x->n_rows; k++){
            tempnum += temp.at(k) * temp.at(k);
          }
          FinalMat.at(i, i) = tempnum;
        }
        else{
          for(unsigned int k = 0; k < x->n_rows; k++){
            tempnum += temp.at(k) * x->at(k, j);
          }
          FinalMat.at(i, j) = tempnum;
          FinalMat.at(j, i) = tempnum;
        }
      }
    }
    else{
      arma::vec temp = x->col(i);
      arma::vec temp2 = x->col(i + 1);
      for(unsigned int j = i; j < x->n_cols; j++){
        double tempnum1 = 0;
        double tempnum2 = 0;
        if(j == i){
          double tempnum3 = 0;
          for(unsigned int k = 0; k < x->n_rows; k++){
            tempnum1 += temp.at(k) * temp.at(k);
            tempnum2 += temp2.at(k) * temp2.at(k);
            tempnum3 += temp.at(k) * temp2.at(k);
          }
          FinalMat.at(i, i) = tempnum1;
          FinalMat.at(i + 1, i) = tempnum3;
          FinalMat.at(i, i + 1) = tempnum3;
          FinalMat.at(i + 1, i + 1) = tempnum2;
          j++;
        }else{
          for(unsigned int k = 0; k < x->n_rows; k++){
            tempnum1 += temp.at(k) * x->at(k, j);
            tempnum2 += temp2.at(k) * x->at(k, j);
          }
          FinalMat(i, j) = tempnum1;
          FinalMat(i + 1, j) = tempnum2;
          FinalMat(j, i) = tempnum1;
          FinalMat(j, i + 1) = tempnum2;
        }
      }
    }
  }
  return(FinalMat);
}

// Use this for non-parallel linear regression
arma::mat XTX(const arma::mat* x, unsigned int B = 16){
  
  arma::mat FinalMat(x->n_cols, x->n_cols, arma::fill::zeros);
  

  for(unsigned int i = 0; i < x->n_cols; i+= B){
    for(unsigned int k = 0; k < x->n_rows; k+=B){
      for(unsigned int ii = i; ii < std::min(x->n_cols, i + B); ii+=2){
        if(ii == std::min(x->n_cols, i + B) - 1){
          for(unsigned int j = ii; j < x->n_cols; j+=B){
            for(unsigned int jj = j; jj < std::min(x->n_cols, j + B); jj++){
              double temp = 0;
              for(unsigned int kk = k; kk < std::min(x->n_rows, k + B); kk++){
                temp += x->at(kk, ii) * x->at(kk, jj);
              }
              FinalMat.at(ii, jj) += temp;
              if(jj != ii && k >= x->n_rows - B){
                FinalMat.at(jj, ii) = FinalMat.at(ii, jj);
              }
            }
          }
        }
        else{
          for(unsigned int j = ii; j < x->n_cols; j+=B){
            for(unsigned int jj = j; jj < std::min(x->n_cols, j + B); jj++){
              if(jj == ii){
                double temp = 0;
                for(unsigned int kk = k; kk < std::min(x->n_rows, k + B); kk++){
                  temp += x->at(kk, ii) * x->at(kk, jj);
                }
                FinalMat.at(ii, jj) += temp;
              }
              else{
                double temp1 = 0;
                double temp2 = 0;
                for(unsigned int kk = k; kk < std::min(x->n_rows, k + B); kk++){
                  temp1 += x->at(kk, ii) * x->at(kk, jj);
                  temp2 += x->at(kk, ii + 1) * x->at(kk, jj);
                }
                FinalMat.at(ii, jj) += temp1;
                FinalMat.at(ii + 1, jj) += temp2;
                if(k >= x->n_rows - B){
                  FinalMat.at(jj, ii) = FinalMat.at(ii, jj);
                  if(jj != ii + 1){
                    FinalMat.at(jj, ii + 1) = FinalMat.at(ii + 1, jj);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return(FinalMat);
}

// Use this for parallel GLM fitting
arma::mat ParXTWX(const arma::mat* x, const arma::vec* w){
  
  arma::mat FinalMat(x->n_cols, x->n_cols);
  
#pragma omp parallel for schedule(dynamic, 1)
  for(unsigned int i = 0; i < x->n_cols; i++){
    
    FinalMat(i, i) = arma::dot((x->col(i) % *w), x->col(i));
    
    for(unsigned int j = i + 1; j < x->n_cols; j++){
      
      FinalMat(i, j) = arma::dot((x->col(j) % *w), x->col(i));
      FinalMat(j, i) = FinalMat(i, j);
      
    } 
    
  }
  return(FinalMat);
}

// Use this for non-parallel GLM fitting
arma::mat XTWX(const arma::mat* x, const arma::vec* w, 
               unsigned int B = 16){
  
  arma::mat FinalMat(x->n_cols, x->n_cols, arma::fill::zeros);
  

  for(unsigned int i = 0; i < x->n_cols; i+= B){
    for(unsigned int k = 0; k < x->n_rows; k+=B){
      for(unsigned int ii = i; ii < std::min(x->n_cols, i + B); ii+=2){
        if(ii == std::min(x->n_cols, i + B) - 1){
          for(unsigned int j = ii; j < x->n_cols; j+=B){
            for(unsigned int jj = j; jj < std::min(x->n_cols, j + B); jj++){
              double temp = 0;
              for(unsigned int kk = k; kk < std::min(x->n_rows, k + B); kk++){
                temp += x->at(kk, ii) * x->at(kk, jj) * w->at(kk);
              }
              FinalMat.at(ii, jj) += temp;
              if(jj != ii && k >= x->n_rows - B){
                FinalMat.at(jj, ii) = FinalMat.at(ii, jj);
              }
            }
          }
        }
        else{
          for(unsigned int j = ii; j < x->n_cols; j+=B){
            for(unsigned int jj = j; jj < std::min(x->n_cols, j + B); jj++){
              if(jj == ii){
                double temp = 0;
                for(unsigned int kk = k; kk < std::min(x->n_rows, k + B); kk++){
                  temp += x->at(kk, ii) * x->at(kk, jj) * w->at(kk);
                }
                FinalMat.at(ii, jj) += temp;
              }
              else{
                double temp1 = 0;
                double temp2 = 0;
                for(unsigned int kk = k; kk < std::min(x->n_rows, k + B); kk++){
                  temp1 += x->at(kk, ii) * x->at(kk, jj) * w->at(kk);
                  temp2 += x->at(kk, ii + 1) * x->at(kk, jj) * w->at(kk);
                }
                FinalMat.at(ii, jj) += temp1;
                FinalMat.at(ii + 1, jj) += temp2;
                if(k >= x->n_rows - B){
                  FinalMat.at(jj, ii) = FinalMat.at(ii, jj);
                  if(jj != ii + 1){
                    FinalMat.at(jj, ii + 1) = FinalMat.at(ii + 1, jj);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return(FinalMat);
}
