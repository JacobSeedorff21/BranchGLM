#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// [[Rcpp::export]]

NumericMatrix MakeTable(NumericVector preds, NumericVector y, double cutoff){
  
  NumericMatrix values(2, 2);
  
  for(unsigned int i = 0; i < y.size(); i++){
    if(preds(i) >= cutoff){
      if(y(i) == 0){
        values(0, 1)++;
      }
      else{
        values(1, 1)++;
      }
      
    }
    else{
      if(y(i) == 0){
        values(0, 0)++;
      }
      else{values(1, 0)++;}
    }
  }
  return(values);
}

// Creates confusion matrix for factor preds and y

// [[Rcpp::export]]

NumericMatrix MakeTableFactor(CharacterVector preds, CharacterVector y, 
                                  CharacterVector levels){
  
  NumericMatrix values(2, 2);
  
  for(unsigned int i = 0; i < y.size(); i++){
    if(preds(i) == levels(1)){
      if(y(i) == levels(0)){
        values(0, 1)++;
      }
      else{
        values(1, 1)++;
      }
      
    }
    else{
      if(y(i) == levels(0)){
        values(0, 0)++;
      }
      else{values(1, 0)++;}
    }
  }
  return(values);
}

// Creates confusion matrix for numeric preds and factor y

// [[Rcpp::export]]

NumericMatrix MakeTableFactor2(NumericVector preds, CharacterVector y, 
                              CharacterVector levels, double cutoff){
  
  NumericMatrix values(2, 2);
  
  for(unsigned int i = 0; i < y.size(); i++){
    if(preds(i) >= cutoff){
      if(y(i) == levels(0)){
        values(0, 1)++;
      }
      else{
        values(1, 1)++;
      }
      
    }
    else{
      if(y(i) == levels(0)){
        values(0, 0)++;
      }
      else{values(1, 0)++;}
    }
  }
  return(values);
}


// Calculates c-index for numeric preds and y

// [[Rcpp::export]]

double CindexCpp(NumericVector preds, NumericVector y){
  
  double Concordant = 0;
  double Total = 0;
  
  for(unsigned int i = 0; i < y.size(); i++){
    if(y(i) == 0){
      for(unsigned int j = i; j < y.size(); j++){
        if(y(j) == 1){
          Total++;
          if(preds(i) < preds(j)){
            Concordant++;
          }
          else if(preds(i) == preds(j)){
            Concordant += .5;
          }
        }
      }
    }
    else{
      for(unsigned int j = i; j < y.size(); j++){
        if(y(j) == 0){
          Total++;
          if(preds(i) > preds(j)){
            Concordant++;
          }
          else if(preds(i) == preds(j)){
            Concordant += .5;
          }
        }
      }
    }
  }
  return(Concordant / Total);
}
// Calculates c-index via trapezoidal rule

// [[Rcpp::export]]

double CindexTrap(NumericVector Sens, NumericVector Spec){
  
  double Area = 0;
  
  for(unsigned int i = 1; i < Sens.size(); i++){
    Area += ((Sens(i - 1) + Sens(i)) / 2) * (Spec(i) - Spec(i - 1));
  }
  return(Area);
}

// Creates ROC curve

// [[Rcpp::export]]

DataFrame ROCCpp(NumericVector preds, NumericVector y, NumericVector Cutoffs){
  
  NumericVector Sens(Cutoffs.size());
  NumericVector Spec(Cutoffs.size());
  double TotP = sum(y);
  double TotN = y.size() - TotP;
  double TN = 0;
  double TP = TotP;
  unsigned int j = 0;
  
  for(unsigned int i = 0; i < Cutoffs.size(); i++){
    while(j < y.size() && preds(j) == Cutoffs(i)){
      if(y(j) == 1){
        TP--;
      }
      else{
        TN++;
      }
      j++;
    }
    
    Sens(i) = TP / TotP;
    Spec(i) = TN / TotN;
  }
  return(DataFrame::create(Named("Sensitivity") = Sens, 
                      Named("Specificity") = Spec,
                      Named("Cutoffs") = Cutoffs));
}

