#' Variable Selection for GLMs 
#' @param formula a formula used to define upper model.
#' @param data a dataframe with the response and predictor variables.
#' @param family distribution used to model the data, one of "gaussian", "binomal", or "poisson"
#' @param link link used to link mean structure to linear predictors. One of, 
#' "identity", "logit", "probit", "cloglog", or "log".
#' @param offset offset vector, by default the zero vector is used.
#' @param fit a BranchGLM object used to define the upper model.
#' @param method one of "Fisher", "BFGS", or "LBFGS". If method is not specified, 
#' the method that was originally used to fit the BranchGLM model is used.
#' @param type one of "forward", "backward", or "branch and bound" to indicate which type of variable selection to perform.
#' @param metric metric used to choose model, the default is "AIC", but "BIC" is also available.
#' @param keep vector of names to denote variables that must be in the model.
#' @param maxsize maximum number of variables to consider in a single model, the 
#' default is the total number of variables.
#' This number adds onto any variables specified in keep. 
#' @param grads number of gradients to use to approximate inverse information with, only for LBFGS.
#' @param parallel one of TRUE or FALSE to indicate if parallelization should be used.
#' Only available for branch and bound selection.
#' @param nthreads number of threads used with OpenMP, only used if parallel is TRUE.
#' @param tol tolerance used to determine model convergence.
#' @param showprogress whether to show progress updates for branch and bound.
#' @description \code{VariableSelection} performs forward selection, backward elimination, 
#' and branch and bound selection for generalized linear models.
#' @details The model in the formula or the formula from the fitted model is 
#' treated as the upper model. The variables specified in keep along with an 
#' intercept (if included in formula) is the lower model. When an intercept is 
#' included in the model formula it is kept in each model.
#' 
#' The branch and bound method makes use of an efficient branch and bound algorithm 
#' to find the optimal model. This is will find the best model according the metric, but 
#' can be much faster than an exhaustive search.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ### Doing branch and bound selection 
#' VariableSelection(Fit, type = "branch and bound", metric = "BIC")
#' 
#' ### Now doing it in parallel (although it isn't necessary for this dataset)
#' VariableSelection(Fit, type = "branch and bound", parallel = TRUE, metric = "BIC")
#' 
#' ### Using a formula
#' VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
#' link = "identity", metric = "BIC", type = "branch and bound")
#' 
#' ### Using the keep argument
#' VariableSelection(Fit, type = "branch and bound", keep = "Petal.Width", metric = "BIC")
#' @return A \code{BranchGLMVS} object with the following components
#' \item{\code{finalmodel}}{ the final \code{BranchGLM} model selected}
#' \item{\code{variables}}{ a vector corresponding to the selected variables}
#' \item{\code{numchecked}}{ number of models fit}
#' \item{\code{order}}{ the order the variables were added to the model or removed from the model, this is not included for branch and bound selection}
#' \item{\code{type}}{ type of variable selection employed}
#' \item{\code{keep}}{ character vector of variables kept in each model, NULL if none specified}
#' \item{\code{metric}}{ metric used to select model}
#' \item{\code{bestmetric}}{ the best metric found in the search}
#' @name VariableSelection
#' @export
#' 
VariableSelection <- function(x, ...) {
  UseMethod("VariableSelection")
}

#'@rdname VariableSelection
#'@export

VariableSelection.formula <- function(formula, data, family, link, offset = NULL,
                                      method = "Fisher", type = "forward", metric = "AIC",
                                      keep = NULL, maxsize = NULL,
                                      grads = 10, parallel = FALSE, 
                                      nthreads = 8, tol = 1e-4, contrasts = NULL,
                                      showprogress = TRUE){
  
  ### Fitting upper model
  fit <- BranchGLM(formula, data = data, family = family, link = link, 
                   offset = offset, method = method, grads = grads, tol = tol, 
                   contrasts = contrasts)
  
  ### Performing variable selection
  VariableSelection(fit, type = type, metric = metric, keep = keep, 
                    maxsize = maxsize, method = method, grads = grads, 
                    parallel = parallel, 
                    nthreads = nthreads, tol = tol, showprogress = showprogress)
}

#'@rdname VariableSelection
#'@export

VariableSelection.BranchGLM <- function(fit, type = "forward", metric = "AIC",
                                        keep = NULL, maxsize = NULL, 
                                        method = NULL, grads = 10, parallel = FALSE, 
                                        nthreads = 8, tol = 1e-4, showprogress = TRUE){
  ## Performing argument checks
  if(is.null(method)){
    method <- fit$method
  }
  if(length(parallel) > 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  if((length(nthreads) > 1) || (!is.numeric(nthreads))||(nthreads <= 0)){
    warning("Please select a positive integer for nthreads, using nthreads = 8")
    nthreads <- 8
  }
  if(length(metric) > 1 || !is.character(metric)){
    stop("metric must be one of 'AIC', or 'BIC'")
  }else if(!(metric %in% c("AIC", "BIC"))){
    stop("metric must be one of 'AIC' or 'BIC'")
  }
  indices <- attributes(fit$x)$assign
  
  ## Checking for intercept
  
  if(colnames(fit$x)[1] == "(Intercept)"){
    intercept <- TRUE
  }else{
    intercept <- FALSE
    indices <- indices - 1
  }
  
  counts <- table(indices)
  if(is.null(maxsize)){
    maxsize <- length(counts)
  }else if(length(maxsize) != 1 || !is.numeric(maxsize) || maxsize < 0){
    stop("maxsize must be a positive integer specifying the max size of the models") 
  }
  
  ## Setting starting model
  keep1 <- keep
  if(is.null(keep) && type == "forward"){
    keep <- rep(0, length(counts))
    if(intercept){
      keep[1] <- -1
    }
  }else if(is.null(keep) && type == "backward"){
    keep <- rep(1, length(counts))
    keep[1] <- -1
  }else{
    CurNames <- attributes(terms(fit$formula, data = fit$data))$factors |>
      colnames()
    keep <- (CurNames %in% keep) * -1
    if(type == "backward"){
      keep[keep == 0] <- 1
    }
    if(intercept){
      keep <- c(-1, keep) 
    }
  }
  
  ## Checking for parallel
  
  if(!parallel){
    nthreads <- 1
  }
  if(type == "forward"){
    df <- ForwardCpp(fit$x, fit$y, fit$offset, indices, counts, method, grads,
                     fit$link, fit$family, nthreads, tol, keep, maxsize, 
                     metric)
    
  }else if(type == "backward"){
    df <- BackwardCpp(fit$x, fit$y, fit$offset, indices, counts, method, grads,
                      fit$link, fit$family, nthreads, tol, keep, maxsize, 
                      metric)
  }else if(type == "branch and bound"){
    if(parallel){
      df <- ParBranchAndBoundCpp(fit$x, fit$y, fit$offset, indices, counts, method, grads,
                                        fit$link, fit$family, nthreads, tol, keep, maxsize, 
                                        metric, showprogress)
    }else{
      df <- BranchAndBoundCpp(fit$x, fit$y, fit$offset, indices, counts, method, grads,
                                      fit$link, fit$family, nthreads, tol, keep, maxsize, 
                                      metric, showprogress)
    }
  }else{
    stop("type must be one of 'forward', 'backward', or 'branch and bound'")
  }
  
  df$model[df$model == -1] <- 1
  
  df$order <- df$order[df$order != -1]
  
  if(!intercept){
    df$order <- df$order + 1
  }else{
    df$model <- df$model[-1] 
  }
  
  tempnames <- paste0(fit$names[as.logical(df$model)], 
                      collapse = "+")
  
  if(nchar(tempnames) == 0 && intercept){
    tempnames <- 1
  }else if(nchar(tempnames) == 0){
    stop("Final model included no variables or intercept")
  }
  
  df$fit$formula <- as.formula(paste0(fit$yname, " ~ ", tempnames))
  
  if(!intercept){
    df$fit$formula <- deparse1(df$fit$formula) |>
      paste0(" - 1") |>
      as.formula()
  }
  
  df$fit$x <- model.matrix(df$fit$formula, fit$data, fit$contrasts)
  
  df$fit$y <- fit$y
  
  row.names(df$fit$coefficients) <- colnames(df$fit$x)
  
  df$fit$names <- attributes(terms(df$fit$formula, data = df$fit$x))$factors |>
    colnames()
  
  df$fit$yname <- fit$yname
  
  df$fit$parallel <- (parallel != FALSE)
  
  df$fit$missing <- fit$missing
  
  df$fit$link <- fit$link
  
  df$fit$contrasts <- fit$contrasts
  
  df$fit$family <- fit$family
  
  df$fit$method <- method
  
  if(fit$family == "binomial"){
    df$fit$ylevel <- fit$ylevel
  }
  if(type != "branch and bound"){
    FinalList <- list("finalmodel" = structure(df$fit, class = "BranchGLM"),
                      "variables" = df$model, 
                      "numchecked" = df$numchecked,
                      "order" = fit$names[df$order],
                      "type" = type, 
                      "keep" = keep1,
                      "metric" = metric,
                      "bestmetric" = df$bestmetric)
  }else{
    FinalList <- list("finalmodel" = structure(df$fit, class = "BranchGLM"),
                      "variables" = df$model, 
                      "numchecked" = df$numchecked,
                      "type" = type, 
                      "keep" = keep1,
                      "metric" = metric,
                      "bestmetric" = df$bestmetric)
  }
  
  structure(FinalList, class = "BranchGLMVS")
}

#' @title Print Method for BranchGLMVS
#' @param fit A BranchGLMVS model object.
#' @param coefdigits Number of digits to display for coefficients table.
#' @param digits Number of digits to display for information after table.
#' @export

print.BranchGLMVS <- function(VS, coefdigits = 4, digits = 0){
  
  fit <- VS$finalmodel
  
  coefs <- fit$coefficients
  
  spaces <- row.names(coefs) |>
    nchar() |>
    max()
  
  spaces <- spaces + 1
  
  
  if(any(coefs$p.values < 2e-16)){
    rounder <- 4
  }else if(any(coefs$p.values < 10^(-coefdigits))){
    rounder <- nchar(coefdigits) + 2
  }else{rounder <- coefdigits}
  
  coefs$p.values <- ifelse(coefs$p.values < 2e-16, "<2e-16" ,
                           ifelse(coefs$p.values < 10^(-coefdigits), 
                                  paste0("<1e-", coefdigits),
                                  format(round(coefs$p.values, digits = coefdigits), 
                                         nsmall = max(coefdigits, rounder))))
  
  Rounded <- sapply(coefs[, -4], round, digits = coefdigits, simplify = F)  |>
    sapply(format, nsmall = coefdigits, simplify = F)
  
  Rounded$p.values <- coefs$p.values 
  
  MoreSpaces <- sapply(Rounded, nchar, simplify = F) |>
    sapply(max)
  
  MoreSpaces <- pmax(MoreSpaces, c(9, 3, 7, 7))
  
  MoreSpaces[1:3] <- MoreSpaces[1:3] + 1
  
  cat("Variable Selection Info:\n")
  cat(paste0(rep("-", spaces + sum(MoreSpaces)), collapse = ""))
  cat("\n")
  if(VS$type != "backward"){
    cat(paste0("Variables were selected using ", VS$type, " selection with ", VS$metric, "\n"))
  }else{
    cat(paste0("Variables were selected using ", VS$type, " elimination with ", VS$metric, "\n"))
  }
  cat(paste0("The best value of ", VS$metric, " obtained was ", 
             round(VS$bestmetric, digits = digits), "\n"))
  cat(paste0("Number of models fit: ", VS$numchecked))
  cat("\n")
  if(!is.null(VS$keep)){
    cat("Variables that were kept in each model: ", paste0(VS$keep, collapse = ", "))
  }
  cat("\n")
  if(length(VS$order) == 0){
    if(VS$type == "forward"){
      cat("No variables were added to the model")
    }else if(VS$type == "backward"){
      cat("No variables were removed from the model")
    }
  }else if(VS$type == "forward" ){
    cat("Order the variables were added to the model:\n")
  }else if(VS$type == "backward" ){
    cat("Order the variables were removed from the model:\n")
  }
  cat("\n")
  if(length(VS$order) > 0){
    for(i in 1:length(VS$order)){
      cat(paste0(i, "). ", VS$order[i], "\n"))
    }
  }
  cat(paste0(rep("-", spaces + sum(MoreSpaces)), collapse = ""))
  cat("\n")
  cat(paste0("Final Model:\n"))
  cat(paste0(rep("-", spaces + sum(MoreSpaces)), collapse = ""))
  cat("\n")
  print(fit, coefdigits = coefdigits, digits = digits)
  
  invisible(VS)
}