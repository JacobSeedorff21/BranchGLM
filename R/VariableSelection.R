#' Variable Selection for GLMs 
#' @param object a formula or a \code{BranchGLM} object.
#' @param ... further arguments passed to other methods.
#' @param data a dataframe with the response and predictor variables.
#' @param family distribution used to model the data, one of "gaussian", "gamma", "binomial", or "poisson".
#' @param link link used to link mean structure to linear predictors. One of
#' "identity", "logit", "probit", "cloglog", "sqrt", "inverse", or "log".
#' @param offset offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS". Fisher's scoring is recommended
#' for forward selection and branch and bound selection since they will typically 
#' fit many models with a small number of covariates.
#' @param type one of "forward", "backward", or "branch and bound" to indicate 
#' which type of variable selection to perform.
#' @param metric metric used to choose model, the default is "AIC", but "BIC" is also available.
#' @param keep vector of names to denote variables that must be in the model.
#' @param maxsize maximum number of variables to consider in a single model, the 
#' default is the total number of variables.
#' This number adds onto any variables specified in keep. 
#' @param grads number of gradients used to approximate inverse information with, only for \code{method = "LBFGS"}.
#' @param parallel one of TRUE or FALSE to indicate if parallelization should be used
#' @param nthreads number of threads used with OpenMP, only used if \code{parallel = TRUE}.
#' @param tol tolerance used to determine model convergence.
#' @param maxit maximum number of iterations performed. The default for 
#' Fisher's scoring is 50 and for the other methods the default is 200.
#' @param showprogress whether to show progress updates for branch and bound.
#' @param contrasts see \code{contrasts.arg} of \code{model.matrix.default}.
#' @description Performs forward selection, backward elimination, 
#' and branch and bound selection for generalized linear models.
#' @details The model in the formula or the formula from the fitted model is 
#' treated as the upper model. The variables specified in keep along with an 
#' intercept (if included in formula) is the lower model. When an intercept is 
#' included in the model formula it is kept in each model. Interaction terms 
#' are not properly handled, i.e. an interaction term may be kept while removing 
#' the lower-order terms. Factor variables are either kept in their entirety or 
#' entirely removed.
#' 
#' The branch and bound method makes use of an efficient branch and bound algorithm 
#' to find the optimal model. This is will find the best model according to the metric, but 
#' can be much faster than an exhaustive search. The amount of speedup attained by 
#' using the branch and bound method as opposed to an exhaustive search depends on 
#' the specific problem. Sometimes it may not be able to prune much and must 
#' fit most of the models and sometimes it may be able to prune off many of the models.
#' 
#' Fisher's scoring is recommended for branch and bound selection and forward selection.
#' All observations that have any missing values in the upper model are ignored.
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
#' @return A \code{BranchGLMVS} object which is a list with the following components
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
VariableSelection <- function(object, ...) {
  UseMethod("VariableSelection")
}

#'@rdname VariableSelection
#'@export

VariableSelection.formula <- function(object, data, family, link, offset = NULL,
                                      method = "Fisher", type = "forward", metric = "AIC",
                                      keep = NULL, maxsize = NULL,
                                      grads = 10, parallel = FALSE, 
                                      nthreads = 8, tol = 1e-4, maxit = NULL,
                                      contrasts = NULL,
                                      showprogress = TRUE, ...){
  
  ### Creating pseudo BranchGLM object to use in VariableSelection.BranchGLM
  ### Validating supplied arguments
  formula <- object
  if(!is(formula, "formula")){
    stop("formula must be a valid formula")
  }
  if(!is(data, "data.frame")){
    stop("data must be a data frame")
  }
  if(length(method) != 1 || !(method %in% c("Fisher", "BFGS", "LBFGS"))){
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }
  if(!family %in% c("gaussian", "binomial", "poisson", "gamma")){
    stop("family must be one of 'gaussian', 'binomial', 'gamma', or 'poisson'")
  }
  if(!link %in% c("logit", "probit", "cloglog", "log", "identity", "inverse", "sqrt")){
    stop("link must be one of 'logit', 'probit', 'cloglog', 'log', 'inverse', 'sqrt', or 'identity'")
  }
  if(length(grads) != 1 || !is.numeric(grads) || as.integer(grads) <= 0){
    stop("grads must be a positive integer")
  }
  ### Evaluating arguments
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("data", "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf$na.action <- "na.omit"
  mf$formula <- formula
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  
  ## Getting data objects
  y <- model.response(mf, "any")
  offset <- as.vector(model.offset(mf))
  x <- model.matrix(attr(mf, "terms"), mf, contrasts)
  ### Checking offset
  if(is.null(offset)){
    offset <- rep(0, nrow(x))
  }else if(length(offset) != length(y)){
    stop("offset must have the same length as the y")
  }else if(!is.numeric(offset)){
    stop("offset must be a numeric vector")
  }
  
  ## Setting maxit
  if(is.null(maxit)){
    if(method == "Fisher"){
      maxit <- 50
    }else{
      maxit <- 200
    }
  }else if(length(maxit) != 1 || !is.numeric(maxit) || maxit != as.integer(maxit) || maxit < 0){
    stop("maxit must be a non-negative integer")
  }
  
  ## Checking y variable and link function for each family
  if(family == "binomial"){
    if(!(link %in% c("cloglog", "log", "logit", "probit"))){
      stop("valid link functions for binomial regression are 'cloglog', 'log', 'logit', and 'probit'")
    }else if(is.factor(y) && (nlevels(y) == 2)){
      ylevel <- levels(y)
      y <- as.numeric(y == ylevel[2])
    }else if(is.numeric(y) && all(y %in% c(0, 1))){
      ylevel <- c(0, 1)
    }else if(is.logical(y)){
      ylevel <- c(FALSE, TRUE)
      y <- y * 1
    }else{
      stop("response variable for binomial regression must be numeric with only 
      0s and 1s, a two-level factor, or a logical vector")
    }
  }else if(family == "poisson"){
    if(!(link %in% c("identity", "log", "sqrt"))){
      stop("valid link functions for poisson regression are 'identity', 'log', and 'sqrt'")
    }else if(!is.numeric(y) || any(y < 0)){
      stop("response variable for poisson regression must be a numeric vector of non-negative integers")
    }else if(any(as.integer(y)!= y)){
      stop("response variable for poisson regression must be a numeric vector of non-negative integers")
    }
  }else if(family == "gaussian"){
    if(!(link %in% c("inverse", "identity", "log", "sqrt"))){
      stop("valid link functions for gaussian regression are 'identity', 'inverse', 'log', and 'sqrt'")
    }else if(!is.numeric(y)){
      stop("response variable for gaussian regression must be numeric")
    }else if(link == "log" && any(y <= 0)){
      stop("gaussian regression with log link must have positive response values")
    }else if(link == "inverse" && any(y == 0)){
      stop("gaussian regression with inverse link must have non-zero response values")
    }else if(link == "sqrt" && any(y < 0)){
      stop("gaussian regression with sqrt link must have non-negative response values")
    }
  }else if(family == "gamma"){
    if(!(link %in% c("inverse", "identity", "log", "sqrt"))){
      stop("valid link functions for gamma regression are 'identity', 'inverse', 'log', and 'sqrt'")
    }else if(!is.numeric(y) || any(y <= 0)){
      stop("response variable for gamma regression must be positive")
    }
  }
  
  df <- list()
  
  df$formula <- formula
  
  df$y <- y
  
  df$x <- x
  
  df$data <- data
  
  df$names <- attributes(terms(df$formula, data = data))$factors |>
    colnames()
  
  df$yname <- attributes(terms(df$formula, data = data))$variables[-1] |>
    as.character()
  
  df$yname <- df$yname[attributes(terms(df$formula, data = data))$response]
  
  df$missing <- nrow(data) - nrow(x)
  
  df$link <- link
  
  df$contrasts <- contrasts
  
  df$offset <- offset
  
  df$terms <- attr(mf, "terms")
  
  df$family <- family
  if(family == "binomial"){
    df$ylevel <- ylevel
  }
  
  fit <- structure(df, class = "BranchGLM")
  
  ### Performing variable selection
  VariableSelection(fit, type = type, metric = metric, keep = keep, 
                    maxsize = maxsize, method = method, grads = grads, 
                    parallel = parallel, 
                    nthreads = nthreads, tol = tol, maxit = maxit,
                    showprogress = showprogress)
}

#'@rdname VariableSelection
#'@export

VariableSelection.BranchGLM <- function(object, type = "forward", metric = "AIC",
                                        keep = NULL, maxsize = NULL, 
                                        method = "Fisher", grads = 10, parallel = FALSE, 
                                        nthreads = 8, tol = 1e-4, maxit = NULL,
                                        showprogress = TRUE, ...){
  
  ## Checking if supplied BranchGLM object has x and data
  if(is.null(object$data) || is.null(object$x)){
    stop("the supplied model must have a data and an x component")
  }
    
  ## Validating supplied arguments
  if(length(parallel) > 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  if(length(method) != 1 || !(method %in% c("Fisher", "BFGS", "LBFGS"))){
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }
  if((length(nthreads) > 1) || (!is.numeric(nthreads))||(nthreads <= 0)){
    warning("Please select a positive integer for nthreads, using nthreads = 8")
    nthreads <- 8
  }
  if(length(grads) != 1 || !is.numeric(grads) || as.integer(grads) <= 0){
    stop("grads must be a positive integer")
  }
  if(length(metric) > 1 || !is.character(metric)){
    stop("metric must be one of 'AIC', or 'BIC'")
  }else if(!(metric %in% c("AIC", "BIC"))){
    stop("metric must be one of 'AIC' or 'BIC'")
  }
  indices <- attributes(object$x)$assign
  
  ## Setting maxit
  if(is.null(maxit)){
    if(method == "Fisher"){
      maxit <- 50
    }else{
      maxit <- 200
    }
  }else if(length(maxit) != 1 || !is.numeric(maxit) || maxit != as.integer(maxit) || maxit < 0){
    stop("maxit must be a non-negative integer")
  }
  
  ## Checking for intercept
  if(colnames(object$x)[1] == "(Intercept)"){
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
    CurNames <- attributes(terms(object$formula, data = object$data))$factors |>
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
  
  ## Performing variable selection
  if(type == "forward"){
    df <- ForwardCpp(object$x, object$y, object$offset, indices, counts, method, grads,
                     object$link, object$family, nthreads, tol, maxit, keep, maxsize, 
                     metric)
    
  }else if(type == "backward"){
    df <- BackwardCpp(object$x, object$y, object$offset, indices, counts, method, grads,
                      object$link, object$family, nthreads, tol, maxit, keep, maxsize, 
                      metric)
  }else if(type == "branch and bound"){
    df <- BranchAndBoundCpp(object$x, object$y, object$offset, indices, counts, method, grads,
                                      object$link, object$family, nthreads, tol, maxit, keep, maxsize, 
                                      metric, showprogress)
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
  
  tempnames <- paste0(object$names[as.logical(df$model)], 
                      collapse = "+")
  
  if(nchar(tempnames) == 0 && intercept){
    tempnames <- 1
  }else if(nchar(tempnames) == 0){
    stop("Final model included no variables or intercept")
  }
  
  df$fit$formula <- as.formula(paste0(object$yname, " ~ ", tempnames))
  
  if(!intercept){
    df$fit$formula <- deparse1(df$fit$formula) |>
      paste0(" - 1") |>
      as.formula()
  }
  
  df$fit$numobs <- nrow(object$x)
  
  df$fit$y <- object$y
  
  x <- model.matrix(df$fit$formula, object$data, object$contrasts)
  
  row.names(df$fit$coefficients) <- colnames(x)
  
  df$fit$names <- attributes(terms(df$fit$formula, data = x))$factors |>
    colnames()
  
  df$fit$yname <- object$yname
  
  df$fit$parallel <- parallel
  
  df$fit$missing <- object$missing
  
  df$fit$link <- object$link
  
  df$fit$contrasts <- object$contrasts
  
  df$fit$family <- object$family
  
  df$fit$method <- method
  
  df$fit$offset <- object$offset
  
  df$fit$terms <- terms(df$fit$formula, data = x)
  
  if(object$family == "binomial"){
    df$fit$ylevel <- object$ylevel
  }
  if(type != "branch and bound"){
    FinalList <- list("finalmodel" = structure(df$fit, class = "BranchGLM"),
                      "variables" = df$model, 
                      "numchecked" = df$numchecked,
                      "order" = object$names[df$order],
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

#' Print Method for BranchGLMVS
#' @param x a \code{BranchGLMVS} object.
#' @param coefdigits number of digits to display for coefficients table.
#' @param digits number of digits to display for information not in the table.
#' @param ... further arguments passed to other methods.
#' @return The supplied \code{BranchGLMVS} object.
#' @export

print.BranchGLMVS <- function(x, coefdigits = 4, digits = 0, ...){
  
  fit <- x$finalmodel
  
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
  if(x$type != "backward"){
    cat(paste0("Variables were selected using ", x$type, " selection with ", x$metric, "\n"))
  }else{
    cat(paste0("Variables were selected using ", x$type, " elimination with ", x$metric, "\n"))
  }
  cat(paste0("The best value of ", x$metric, " obtained was ", 
             round(x$bestmetric, digits = digits), "\n"))
  cat(paste0("Number of models fit: ", x$numchecked))
  cat("\n")
  if(!is.null(x$keep)){
    cat("Variables that were kept in each model: ", paste0(x$keep, collapse = ", "))
  }
  cat("\n")
  if(length(x$order) == 0){
    if(x$type == "forward"){
      cat("No variables were added to the model")
    }else if(x$type == "backward"){
      cat("No variables were removed from the model")
    }
  }else if(x$type == "forward" ){
    cat("Order the variables were added to the model:\n")
  }else if(x$type == "backward" ){
    cat("Order the variables were removed from the model:\n")
  }
  cat("\n")
  if(length(x$order) > 0){
    for(i in 1:length(x$order)){
      cat(paste0(i, "). ", x$order[i], "\n"))
    }
  }
  cat(paste0(rep("-", spaces + sum(MoreSpaces)), collapse = ""))
  cat("\n")
  cat(paste0("Final Model:\n"))
  cat(paste0(rep("-", spaces + sum(MoreSpaces)), collapse = ""))
  cat("\n")
  print(fit, coefdigits = coefdigits, digits = digits)
  
  invisible(x)
}