#' Confusion Matrix
#' @description Creates a confusion matrix and calculates related measures.
#' @param object a `BranchGLM` object or a numeric vector.
#' @param ... further arguments passed to other methods.
#' @param y observed values, can be a numeric vector of 0s and 1s, a two-level factor vector, or 
#' a logical vector. 
#' @param cutoff cutoff for predicted values, the default is 0.5.
#' @name Table
#' @return A `BranchGLMTable` object which is a list with the following components
#' \item{`table`}{ a matrix corresponding to the confusion matrix}
#' \item{`accuracy`}{ a number corresponding to the accuracy}
#' \item{`sensitivity`}{ a number corresponding to the sensitivity}
#' \item{`specificity`}{ a number corresponding to the specificity}
#' \item{`PPV`}{ a number corresponding to the positive predictive value}
#' \item{`levels`}{ a vector corresponding to the levels of the response variable}
#' @examples 
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' Table(Fit)
#' @export

Table <- function(object, ...) {
  UseMethod("Table")
}

#' @rdname Table
#' @export

Table.numeric <- function(object, y, cutoff = .5, ...){
  
  ## Checking y and object
  if((!is.numeric(y)) && (!is.factor(y)) && (!is.logical(y))){
    stop("y must be a numeric, two-level factor, or logical vector")
  }else if(length(y) !=  length(object)){
    stop("Length of y must be the same as the length of object")
  }else if((any(object > 1) || any(object < 0))){
    stop("object must be between 0 and 1")
  }else if(any(is.na(object)) || any(is.na(y))){
    stop("object and y must not have any missing values")
  }else if(is.factor(y) && nlevels(y) != 2){
    stop("If y is a factor vector it must have exactly two levels")
  }else if(is.numeric(y) && any((y != 1) & (y != 0))){
    stop("If y is numeric it must only contain 0s and 1s.")
  }
  if(is.numeric(y)){
    
    Table <- MakeTable(object, y, cutoff)
    
    List <- list("table" = Table, 
                 "accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
                 "sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
                 "specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
                 "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
                 "levels" = c(0, 1))
  }else if(is.factor(y)){
    
    Table <- MakeTableFactor2(object, as.character(y), levels(y), cutoff)
    
    List <- list("table" = Table, 
                 "accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
                 "sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
                 "specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
                 "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
                 "levels" = levels(y))
    
  }else{
    
    Table <- MakeTable(object, y * 1, cutoff)
    
    List <- list("table" = Table, 
                 "accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
                 "sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
                 "specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
                 "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
                 "levels" = c(FALSE, TRUE))
    
  }
  return(structure(List, class = "BranchGLMTable"))
}

#' @rdname Table
#' @export

Table.BranchGLM <- function(object, cutoff = .5, ...){
  if(is.null(object$y)){
    stop("supplied BranchGLM object must have a y component")
  }
  if(object$family != "binomial"){
    stop("This method is only valid for BranchGLM models in the binomial family")
  }
  
  preds <- predict(object, type = "response")
  
  Table <- MakeTable(preds, object$y, cutoff)
  
  List <- list("table" = Table, 
               "accuracy" = (Table[1, 1] + Table[2, 2]) / (sum(Table)),
               "sensitivity" = Table[2, 2] / (Table[2, 2] + Table[2, 1]),
               "specificity" = Table[1, 1] / (Table[1, 1] + Table[1, 2]),
               "PPV" = Table[2, 2] / (Table[2, 2] + Table[1, 2]),
               "levels" = object$ylevel)
  
  return(structure(List, class = "BranchGLMTable"))
  
}

#' Print Method for BranchGLMTable Objects
#' @description Print method for BranchGLMTable objects.
#' @param x a `BranchGLMTable` object.
#' @param digits number of digits to display.
#' @param ... further arguments passed to other methods.
#' @return The supplied `BranchGLMTable` object.
#' @export

print.BranchGLMTable <- function(x, digits = 4, ...){
  Numbers <- apply(x$table, 2, FUN = function(x){max(nchar(x))})
  
  Numbers <- pmax(Numbers, c(4, 4)) |>
             pmax(nchar(x$levels))
  
  LeftSpace <- 10 + max(nchar(x$levels))
  
  cat("Confusion matrix:\n")
  
  cat(paste0(rep("-", LeftSpace + sum(Numbers) + 2), collapse = ""))
  cat("\n")
  
  cat(paste0(paste0(rep(" ", LeftSpace + Numbers[1] - 4), collapse = ""), 
      "Predicted\n",
      paste0(rep(" ", LeftSpace + floor((Numbers[1] - nchar(x$levels[1])) / 2)), 
             collapse = ""), 
      x$levels[1],
      paste0(rep(" ", ceiling((Numbers[1] - nchar(x$levels[1])) / 2) + 1 + 
                      floor((Numbers[2] - nchar(x$levels[2])) / 2)), 
                 collapse = ""), x$levels[2], "\n\n", 
      paste0(rep(" ", 9), collapse = ""), x$levels[1], 
      paste0(rep(" ", 1 + max(nchar(x$levels)) - nchar(x$levels[1]) + 
                        floor((Numbers[1] - nchar(x$table[1, 1])) / 2)), 
               collapse = ""), 
      x$table[1, 1], 
      paste0(rep(" ", ceiling((Numbers[1] - nchar(x$table[1, 1])) / 2) + 1 + 
                      floor((Numbers[2] - nchar(x$table[1, 2])) / 2)), 
             collapse = ""),
      x$table[1, 2], 
      "\n", "Observed\n",
      paste0(rep(" ", 9), collapse = ""), x$levels[2], 
      paste0(rep(" ", 1 + max(nchar(x$levels)) - nchar(x$levels[2]) + 
                   floor((Numbers[1] - nchar(x$table[2, 1])) / 2)), 
             collapse = ""), 
      x$table[2, 1], 
      paste0(rep(" ", ceiling((Numbers[1] - nchar(x$table[2, 1])) / 2) + 1 + 
                      floor((Numbers[2] - nchar(x$table[2, 2])) / 2)), 
             collapse = ""),
      x$table[2, 2], "\n\n"))
  
  cat(paste0(rep("-", LeftSpace + sum(Numbers) + 2), collapse = ""))
  cat("\n")
  
  cat("Measures:\n")
  
  cat(paste0(rep("-", LeftSpace + sum(Numbers) + 2), collapse = ""))
  cat("\n")
  
  cat("Accuracy: ", round(x$accuracy, digits = digits), "\n")
  cat("Sensitivity: ", round(x$sensitivity, digits = digits), "\n")
  cat("Specificity: ", round(x$specificity, digits = digits), "\n")
  cat("PPV: ", round(x$PPV, digits = digits), "\n")
  
  invisible(x)
  
}

#' Cindex/AUC
#' @param object a `BranchGLM` object, a `BranchGLMROC` object, or a numeric vector.
#' @param ... further arguments passed to other methods.
#' @param y Observed values, can be a numeric vector of 0s and 1s, a two-level 
#' factor vector, or a logical vector.
#' @name Cindex
#' @return A number corresponding to the c-index/AUC.
#' @description Calculates the c-index/AUC.
#' @details Uses trapezoidal rule to calculate AUC when given a BranchGLMROC object and
#' uses Mann-Whitney U to calculate it otherwise. The trapezoidal rule method is less accurate, 
#' so the two methods may give different results.
#' @examples 
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' Cindex(Fit)
#' AUC(Fit)
#' @export

Cindex <- function(object, ...) {
  UseMethod("Cindex")
}

#' @rdname Cindex
#' @export

AUC <- Cindex

#' @rdname Cindex
#' @export

Cindex.numeric <- function(object, y, ...){
  if((!is.numeric(y)) && (!is.factor(y)) && (!is.logical(y))){
    stop("y must be a numeric, two-level factor, or logical vector")
  }else if(length(y) !=  length(object)){
    stop("Length of y must be the same as the length of object")
  }else if((any(object > 1) || any(object < 0))){
    stop("object must be between 0 and 1")
  }else if(any(is.na(object)) || any(is.na(y))){
    stop("object and y must not have any missing values")
  }else if(is.factor(y) && nlevels(y) != 2){
    stop("If y is a factor vector it must have exactly two levels")
  }else if(is.numeric(y) && any((y != 1) & (y != 0))){
    stop("If y is numeric it must only contain 0s and 1s.")
  }
  if(is.numeric(y)){
    cindex <- CindexU(object, y)
  }
  else if(is.factor(y)){
    y <- (y == levels(y)[2])
    cindex <- CindexU(object, y)
  }
  else{
    y <- y * 1
    cindex <- CindexU(object, y)
  }
  cindex
}

#' @rdname Cindex
#' @export

Cindex.BranchGLM <- function(object, ...){
  if(is.null(object$y)){
    stop("supplied BranchGLM object must have a y component")
  }
  if(object$family != "binomial"){
    stop("This method is only valid for BranchGLM models in the binomial family")
  }
  
  preds <- predict(object, type = "response")
  
  cindex <- CindexU(preds, object$y)
  
  cindex

}


#' @rdname Cindex
#' @export

Cindex.BranchGLMROC <- function(object, ...){
  cindex <- CindexTrap(object$Info$Sensitivity, 
                       object$Info$Specificity)
  
  cindex
  
}

#' Calculated AUC/cindex
#' @param preds numeric vector of predictions.
#' @param y a numeric vector of 0s and 1s.
#' @noRd

CindexU <- function(preds, y){
  y1 <- which(y == 1)
  
  Ranks <- rank(preds, ties.method = "average")
  
  U <- sum(Ranks[y1]) - (length(y1) * (length(y1) + 1))/(2)
  
  return(U / (length(y1) * as.double(length(y) - length(y1))))
}


#' ROC Curve
#' @description Creates an ROC curve.
#' @param object a `BranchGLM` object or a numeric vector.
#' @param ... further arguments passed to other methods.
#' @param y observed values, can be a numeric vector of 0s and 1s, a two-level 
#' factor vector, or a logical vector. 
#' @name ROC
#' @return A `BranchGLMROC` object which can be plotted with `plot()`. The AUC can also 
#' be calculated using `AUC()`.
#' @examples 
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' MyROC <- ROC(Fit)
#' plot(MyROC)
#' @export

ROC <- function(object, ...) {
  UseMethod("ROC")
}

#' @rdname ROC
#' @export

ROC.numeric <- function(object, y, ...){
  if((!is.numeric(y)) && (!is.factor(y)) && (!is.logical(y))){
    stop("y must be a numeric, two-level factor, or logical vector")
  }else if(length(y) !=  length(object)){
    stop("Length of y must be the same as the length of object")
  }else if((any(object > 1) || any(object < 0))){
    stop("object must be between 0 and 1")
  }else if(any(is.na(object)) || any(is.na(y))){
    stop("object and y must not have any missing values")
  }else if(is.factor(y) && nlevels(y) != 2){
    stop("If y is a factor vector it must have exactly two levels")
  }else if(is.numeric(y) && any((y != 1) & (y != 0))){
    stop("If y is numeric it must only contain 0s and 1s.")
  }
  if(is.numeric(y)){
    SortOrder <- order(object)
    object <- object[SortOrder]
    ROC <- ROCCpp(object, y[SortOrder], unique(object))
  }else if(is.factor(y)){
    y <- (y == levels(y)[2])
    SortOrder <- order(object)
    object <- object[SortOrder]
    ROC <- ROCCpp(object, y[SortOrder], unique(object))
  }else{
    y <- y * 1
    SortOrder <- order(object)
    object <- object[SortOrder]
    ROC <- ROCCpp(object, y[SortOrder], unique(object))
  }
  ROC <- list("NumObs" = length(object),
              "Info" = ROC)
  return(structure(ROC, class = "BranchGLMROC"))
}

#' @rdname ROC
#' @export

ROC.BranchGLM <- function(object, ...){
  if(is.null(object$y)){
    stop("supplied BranchGLM object must have a y component")
  }
  if(object$family != "binomial"){
    stop("This method is only valid for BranchGLM models in the binomial family")
  }
  
  preds <- predict(object, type = "response")
  SortOrder <- order(preds)
  preds <- preds[SortOrder]
  
  ROC <- ROCCpp(preds, object$y[SortOrder], unique(preds))
  
  ROC <- list("NumObs" = length(preds),
              "Info" = ROC)
  return(structure(ROC, class = "BranchGLMROC"))
}

#' Print Method for BranchGLMROC Objects
#' @description Print method for BranchGLMROC objects.
#' @param x a `BranchGLMROC` object.
#' @param ... further arguments passed to other methods.
#' @return The supplied `BranchGLMROC` object.
#' @export

print.BranchGLMROC <- function(x, ...){
  cat(paste0("Number of observations used to make ROC curve: ", 
             x$NumObs, "\n\nUse plot function to make plot of ROC curve \nCan also use AUC/Cindex function to get the AUC"))
  
  invisible(x)
}


#' Plot Method for BranchGLMROC Objects
#' @description This plots a ROC curve.
#' @param x a `BranchGLMROC` object.
#' @param xlab label for the x-axis.
#' @param ylab label for the y-axis.
#' @param type  what type of plot to draw, see more details at [plot.default].
#' @param ... further arguments passed to [plot.default].
#' @return This only produces a plot, nothing is returned.
#' @examples
#' Data <- ToothGrowth
#' Fit <- BranchGLM(supp ~ ., data = Data, family = "binomial", link = "logit")
#' MyROC <- ROC(Fit)
#' plot(MyROC)
#' @export
plot.BranchGLMROC <- function(x, xlab = "1 - Specificity", ylab = "Sensitivity", 
                              type = "l", ...){
  plot(1 - x$Info$Specificity, x$Info$Sensitivity, xlab = xlab, ylab = ylab, 
       type = type,  ... )
  abline(0, 1, lty = "dotted")
}

#' Plotting Multiple ROC Curves
#' @param ... any number of `BranchGLMROC` objects.
#' @param legendpos a keyword to describe where to place the legend, such as "bottomright".
#' The default is "bottomright"
#' @param title title for the plot.
#' @param colors vector of colors to be used on the ROC curves.
#' @param names vector of names used to create a legend for the ROC curves.
#' @param lty vector of linetypes used to create the ROC curves or a 
#' single linetype to be used for all ROC curves.
#' @param lwd vector of linewidths used to create the ROC curves or a 
#' single linewidth to be used for all ROC curves.
#' @return This only produces a plot, nothing is returned.
#' @examples 
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

