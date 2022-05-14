#' Confusion Matrix
#' @param preds a numeric vector of predictions between 0 and 1 or a two-level factor vector.
#' @param y observed values, can be a numeric vector of 0s and 1s, a two-level factor vector, or 
#' a logical vector. Must be a two-level factor vector if preds is also a two-level factor vector.
#' @param cutoff cutoff for predicted values, default is 0.5.
#' @param fit a \code{BranchGLM} object.
#' @name Table
#' @return a \code{BranchGLMTable} object.
#' @description Creates confusion matrix and calculates related measures.
#' @examples 
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' Table(Fit)
#' @export

Table <- function(x, ...) {
  UseMethod("Table")
}

#' @rdname Table
#' @export

Table.numeric <- function(preds, y, cutoff = .5){
  if((!is.numeric(y)) && (!is.factor(y)) && (!is.logical(y))){
    stop("y must be a numeric, two-level factor, or logical vector")
  }else if(length(y) !=  length(preds)){
    stop("Length of y must be the same as the length of preds")
  }else if((any(preds > 1) || any(preds < 0))){
    stop("preds must be between 0 and 1")
  }else if(any(is.na(preds)) || any(is.na(y))){
    stop("preds and y must not have any missing values")
  }else if(is.factor(y) && nlevels(y) != 2){
    stop("If y is a factor vector it must have exactly two levels")
  }else if(is.numeric(y) && any((y != 1) & (y != 0))){
    stop("If y is numeric it must only contain 0s and 1s.")
  }
  if(is.numeric(y)){
    
    Table <- MakeTable(preds, y, cutoff)
    
    List <- list("Table" = Table, 
                 "Accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
                 "Sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
                 "Specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
                 "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
                 "Levels" = c(0, 1))
    
  }else if(is.factor(y)){
    
    Table <- MakeTableFactor2(preds, as.character(y), levels(y), cutoff)
    
    List <- list("Table" = Table, 
                 "Accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
                 "Sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
                 "Specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
                 "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
                 "Levels" = levels(y))
    
  }else{
    
    Table <- MakeTable(preds, y * 1, cutoff)
    
    List <- list("Table" = Table, 
                 "Accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
                 "Sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
                 "Specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
                 "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
                 "Levels" = c(FALSE, TRUE))
    
  }
  return(structure(List, class = "BranchGLMTable"))
}

#' @rdname Table
#' @export

Table.factor <- function(preds, y){
  if(!is.factor(y)){
    stop("y must be a factor vector")
  }else if(length(y) !=  length(preds)){
    stop("Length of y must be the same as the length of preds")
  }else if((nlevels(y) != 2) || (nlevels(preds) != 2) || 
           any(levels(preds)!= levels(y))){
    stop("preds and y must be factors with exactly 2 levels and must 
         share the same levels")
  }
  
  Table <- MakeTableFactor(as.character(preds), as.character(y), levels(preds))
  
  List <- list("Table" = Table, 
               "Accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
               "Sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
               "Specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
               "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
               "Levels" = levels(y))
  
  return(structure(List, class = "BranchGLMTable"))
  
}

#' @rdname Table
#' @export

Table.BranchGLM <- function(fit, cutoff = .5){
  if(fit$family != "binomial"){
    stop("This method is only valid for BranchGLM models in the binomial family")
  }
  
  preds <- predict(fit, type = "response")
  
  Table <- MakeTable(preds, fit$y, cutoff)
  
  List <- list("Table" = Table, 
               "Accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
               "Sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
               "Specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
               "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
               "Levels" = fit$ylevel)
  
  return(structure(List, class = "BranchGLMTable"))
  
}

#' Print Method for BranchGLMTable
#' @param Table A BranchGLMTable object.
#' @param digits Number of digits to display.
#' @export

print.BranchGLMTable <- function(Table, digits = 4){
  Numbers <- apply(Table$Table, 2, FUN = function(x){max(nchar(x))})
  
  Numbers <- pmax(Numbers, c(4, 4)) |>
             pmax(nchar(Table$Levels))
  
  LeftSpace <- 10 + max(nchar(Table$Levels))
  
  cat("Confusion matrix:\n")
  
  cat(paste0(rep("-", LeftSpace + sum(Numbers) + 2), collapse = ""))
  cat("\n")
  
  cat(paste0(paste0(rep(" ", LeftSpace + Numbers[1] - 4), collapse = ""), 
      "Predicted\n",
      paste0(rep(" ", LeftSpace + floor((Numbers[1] - nchar(Table$Levels[1])) / 2)), 
             collapse = ""), 
      Table$Levels[1],
      paste0(rep(" ", ceiling((Numbers[1] - nchar(Table$Levels[1])) / 2) + 1 + 
                      floor((Numbers[2] - nchar(Table$Levels[2])) / 2)), 
                 collapse = ""), Table$Levels[2], "\n\n", 
      paste0(rep(" ", 9), collapse = ""), Table$Levels[1], 
      paste0(rep(" ", 1 + max(nchar(Table$Levels)) - nchar(Table$Levels[1]) + 
                        floor((Numbers[1] - nchar(Table$Table[1, 1])) / 2)), 
               collapse = ""), 
      Table$Table[1, 1], 
      paste0(rep(" ", ceiling((Numbers[1] - nchar(Table$Table[1, 1])) / 2) + 1 + 
                      floor((Numbers[2] - nchar(Table$Table[1, 2])) / 2)), 
             collapse = ""),
      Table$Table[1, 2], 
      "\n", "Observed\n",
      paste0(rep(" ", 9), collapse = ""), Table$Levels[2], 
      paste0(rep(" ", 1 + max(nchar(Table$Levels)) - nchar(Table$Levels[2]) + 
                   floor((Numbers[1] - nchar(Table$Table[2, 1])) / 2)), 
             collapse = ""), 
      Table$Table[2, 1], 
      paste0(rep(" ", ceiling((Numbers[1] - nchar(Table$Table[2, 1])) / 2) + 1 + 
                      floor((Numbers[2] - nchar(Table$Table[2, 2])) / 2)), 
             collapse = ""),
      Table$Table[2, 2], "\n\n"))
  
  cat(paste0(rep("-", LeftSpace + sum(Numbers) + 2), collapse = ""))
  cat("\n")
  
  cat("Measures:\n")
  
  cat(paste0(rep("-", LeftSpace + sum(Numbers) + 2), collapse = ""))
  cat("\n")
  
  cat("Accuracy: ", round(Table$Accuracy, digits = digits), "\n")
  cat("Sensitivity: ", round(Table$Sensitivity, digits = digits), "\n")
  cat("Specificity: ", round(Table$Specificity, digits = digits), "\n")
  cat("PPV: ", round(Table$PPV, digits = digits), "\n")
  
  invisible(Table)
  
}

#' Cindex/AUC
#' @param preds A numeric vector of predictions between 0 and 1.
#' @param y Observed values, can be a numeric vector of 0s and 1s, a two-level 
#' factor vector, or a logical vector.
#' @param fit A BranchGLM object.
#' @param ROC A BranchGLMROC object.
#' @name Cindex
#' @return The c-index or AUC.
#' @description Calculates c-index.
#' @details Uses trapezoidal rule to calculate AUC when given a BranchGLMROC object.
#' Uses Mann-Whitney U to calculate it otherwise.
#' @examples 
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' Cindex(Fit)
#' AUC(Fit)
#' @export

Cindex <- function(x, ...) {
  UseMethod("Cindex")
}

#' @rdname Cindex
#' @export

AUC <- Cindex

#' @rdname Cindex
#' @export

Cindex.numeric <- function(preds, y){
  if((!is.numeric(y)) && (!is.factor(y)) && (!is.logical(y))){
    stop("y must be a numeric, two-level factor, or logical vector")
  }else if(length(y) !=  length(preds)){
    stop("Length of y must be the same as the length of preds")
  }else if((any(preds > 1) || any(preds < 0))){
    stop("preds must be between 0 and 1")
  }else if(any(is.na(preds)) || any(is.na(y))){
    stop("preds and y must not have any missing values")
  }else if(is.factor(y) && nlevels(y) != 2){
    stop("If y is a factor vector it must have exactly two levels")
  }else if(is.numeric(y) && any((y != 1) & (y != 0))){
    stop("If y is numeric it must only contain 0s and 1s.")
  }
  if(is.numeric(y)){
    cindex <- CindexU(preds, y)
  }
  else if(is.factor(y)){
    y <- (y == levels(y)[2])
    cindex <- CindexU(preds, y)
  }
  else{
    y <- y * 1
    cindex <- CindexU(preds, y)
  }
  cindex
}

#' @rdname Cindex
#' @export

Cindex.BranchGLM <- function(fit){
  if(fit$family != "binomial"){
    stop("This method is only valid for BranchGLM models in the binomial family")
  }
  
  preds <- predict(fit, type = "response")
  
  cindex <- CindexU(preds, fit$y)
  
  cindex

}


#' @rdname Cindex
#' @export

Cindex.BranchGLMROC <- function(ROC){
  cindex <- CindexTrap(ROC$Info$Sensitivity, 
                       ROC$Info$Specificity)
  
  cindex
  
}

CindexU <- function(preds, y){
  y1 <- which(y == 1)
  
  Ranks <- rank(preds, ties.method = "average")
  
  U <- sum(Ranks[y1]) - (length(y1) * (length(y1) + 1))/(2)
  
  return(U / (length(y1) * as.double(length(y) - length(y1))))
}


#' ROC Curve
#' @param preds A numeric vector of predictions between 0 and 1.
#' @param y Observed values, can be a numeric vector of 0s and 1s, a two-level 
#' factor vector, or a logical vector. 
#' @param fit A \code{BranchGLM} object.
#' @name ROC
#' @return A \code{BranchGLMROC} object which can be plotted with \code{plot()}. The AUC can also 
#' be calculated using \code{AUC()}.
#' @description Creates an ROC curve.
#' @examples 
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' MyROC <- ROC(Fit)
#' plot(MyROC)
#' @export

ROC <- function(x, ...) {
  UseMethod("ROC")
}

#' @rdname ROC
#' @export

ROC.numeric <- function(preds, y){
  if((!is.numeric(y)) && (!is.factor(y)) && (!is.logical(y))){
    stop("y must be a numeric, two-level factor, or logical vector")
  }else if(length(y) !=  length(preds)){
    stop("Length of y must be the same as the length of preds")
  }else if((any(preds > 1) || any(preds < 0))){
    stop("preds must be between 0 and 1")
  }else if(any(is.na(preds)) || any(is.na(y))){
    stop("preds and y must not have any missing values")
  }else if(is.factor(y) && nlevels(y) != 2){
    stop("If y is a factor vector it must have exactly two levels")
  }else if(is.numeric(y) && any((y != 1) & (y != 0))){
    stop("If y is numeric it must only contain 0s and 1s.")
  }
  if(is.numeric(y)){
    SortOrder <- order(preds)
    preds <- preds[SortOrder]
    ROC <- ROCCpp(preds, y[SortOrder], unique(preds))
  }else if(is.factor(y)){
    y <- (y == levels(y)[2])
    SortOrder <- order(preds)
    preds <- preds[SortOrder]
    ROC <- ROCCpp(preds, y[SortOrder], unique(preds))
  }else{
    y <- y * 1
    SortOrder <- order(preds)
    preds <- preds[SortOrder]
    ROC <- ROCCpp(preds, y[SortOrder], unique(preds))
  }
  ROC <- list("NumObs" = length(preds),
              "Info" = ROC)
  return(structure(ROC, class = "BranchGLMROC"))
}

#' @rdname ROC
#' @export

ROC.BranchGLM <- function(fit){
  if(fit$family != "binomial"){
    stop("This method is only valid for BranchGLM models in the binomial family")
  }
  
  preds <- predict(fit, type = "response")
  SortOrder <- order(preds)
  preds <- preds[SortOrder]
  
  ROC <- ROCCpp(preds, fit$y[SortOrder], unique(preds))
  
  ROC <- list("NumObs" = length(preds),
              "Info" = ROC)
  return(structure(ROC, class = "BranchGLMROC"))
}


#' Print Method for BranchGLMROC
#' @return a BranchGLMROC object.
#' @param ROC BranchGLMROC object.
#' @export
print.BranchGLMROC <- function(ROC){
  cat(paste0("Number of observations used to make ROC curve: ", 
             ROC$NumObs, "\n\nUse plot function to make plot of ROC curve \nCan also use AUC/Cindex function to get the AUC"))
  
  invisible(ROC)
}


#' Plotting ROC Curve
#' @description This plots a ROC curve.
#' @param ROC a BranchGLMROC object.
#' @param ... arguments passed to generic plot function.
#' @examples
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' MyROC <- ROC(Fit)
#' plot(MyROC)
#' @export
plot.BranchGLMROC <- function(ROC, ...){
  plot(1 - ROC$Info$Specificity, ROC$Info$Sensitivity, xlab = "1 - Specificity", 
       ylab = "Sensitivity", type = "l",  ... )
  abline(0, 1, lty = "dotted")
}

#' Plotting Multiple ROC Curves
#' @param ... any number of BranchGLMROC objects.
#' @param legendpos where to place legend.
#' @param title title for plot.
#' @param colors vector of colors to be used on ROC curves.
#' @param names vector of names used to create legend for ROC curves.
#' @param lty vector of linetypes used to create ROC curves or a 
#' single linetype to be used for all ROC curves.
#' @param lwd vector of linewidths used to create ROC curves or a 
#' single linewidth to be used for all ROC curves.
#' #' @examples 
#' Data <- ToothGrowth
#' 
#' ### Logistic ROC
#' LogisticFit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' LogisticROC <- ROC(LogisticFit)
#' 
#' ### Probit ROC
#' ProbitFit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "probit")
#' ProbitROC <- ROC(ProbitFit)
#' 
#' ### Cloglog ROC
#' CloglogFit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "cloglog")
#' CloglogROC <- ROC(CloglogFit)
#' 
#' ### Plotting ROC curves
#' 
#' MultipleROCCurves(LogisticROC, ProbitROC, CloglogROC, 
#'                   names = c("Logistic ROC", "Probit ROC", "Cloglog ROC"))
#' 
#' @export
MultipleROCCurves <- function(..., legendpos = "bottomright", title = "ROC Curves", 
                            colors = NULL, names = NULL, lty = 1, lwd = 1){
  ROCs <- list(...)
  if(length(ROCs) == 0){
    stop("must provide at least one ROC curve")
  }
  if(!all(sapply(ROCs, is, class = "BranchGLMROC"))){
    stop("All arguments in ... must be BranchGLMROC objects")
  }
  if(is.null(colors)){
    colors <- 1:length(ROCs)
  }else if(length(ROCs) != length(colors)){
    stop("colors must have the same length as the number of ROC curves")
  }
  if(length(lty) == 1){
    lty <- rep(lty, length(colors))
  }else if(length(ROCs) != length(lty)){
    stop("lty must have the same length as the number of ROC curves or a length of 1")
  }
  if(length(lwd) == 1){
    lwd <- rep(lwd, length(colors))
  }else if(length(ROCs) != length(lwd)){
    stop("lwd must have the same length as the number of ROC curves or a length of 1")
  }
  if(length(title) > 1){
    stop("title must have a length of 1")
  }else if(!(is.character(title) || is.expression(title))){
    stop("title must be a character string or an expression")
  }
  
  plot(ROCs[[1]], col = colors[1], lty = lty[1], lwd = lwd[1], main = title)
  if(length(ROCs) > 1){
    for(i in 2:length(ROCs)){
      lines(1 - ROCs[[i]]$Info$Specificity, ROCs[[i]]$Info$Sensitivity, 
            col = colors[i], lty = lty[i], lwd = lwd[i])
    }
  }
  if(is.null(names)){
  legend(legendpos, legend = paste0("ROC ", 1:length(colors)), 
         col = colors, lty = lty, lwd = lwd)
  }else if(length(names) != length(colors)){
    stop("names must have the same length as the number of ROC curves")
  }else{
    legend(legendpos, legend = names, 
            col = colors, lty = lty, lwd = lwd)
  }
} 

