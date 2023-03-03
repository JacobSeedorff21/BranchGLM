#' Variable Selection for GLMs 
#' @param object a formula or a \code{BranchGLM} object.
#' @param ... further arguments passed to other methods.
#' @param data a dataframe with the response and predictor variables.
#' @param family distribution used to model the data, one of "gaussian", "gamma", 
#' "binomial", or "poisson".
#' @param link link used to link mean structure to linear predictors. One of
#' "identity", "logit", "probit", "cloglog", "sqrt", "inverse", or "log".
#' @param offset offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS". Fisher's scoring is recommended
#' for forward selection and branch and bound methods since they will typically 
#' fit many models with a small number of covariates.
#' @param type one of "forward", "backward", "branch and bound", "backward branch and bound", 
#' or "switch branch and bound" to indicate the type of variable selection to perform. 
#' The default value is "branch and bound". The branch and bound methods are guaranteed to 
#' find the best models according to the metric while "forward" and "backward" are 
#' heuristic approaches that may not find the optimal model.
#' @param metric metric used to choose the best models, the default is "AIC", 
#' but "BIC" and "HQIC" are also available. AIC is the Akaike information criterion, 
#' BIC is the bayesian information criterion, and HQIC is the Hannan-Quinn information 
#' criterion.
#' @param bestmodels number of the best models to find according to the chosen metric, 
#' the default is 1. This is only used for the branch and bound methods.
#' @param cutoff this is a non-negative number which indicates that the function 
#' should return all models that have a metric value within cutoff of the 
#' best metric value. The default value is 0 and only one of this or bestmodels 
#' should be specified. This is only used for the branch and bound methods.
#' @param keep vector of names to denote variables that must be in the models.
#' @param maxsize maximum number of variables to consider in a single model, the 
#' default is the total number of variables. This number adds onto any variables specified in keep. 
#' This argument only works for \code{type = "forward"} and \code{type = "branch and bound"}.
#' @param grads number of gradients used to approximate inverse information with, only for \code{method = "LBFGS"}.
#' @param parallel one of TRUE or FALSE to indicate if parallelization should be used
#' @param nthreads number of threads used with OpenMP, only used if \code{parallel = TRUE}.
#' @param tol tolerance used to determine model convergence when fitting GLMs.
#' @param maxit maximum number of iterations performed when fitting GLMs. The default for 
#' Fisher's scoring is 50 and for the other methods the default is 200.
#' @param showprogress whether to show progress updates for branch and bound methods.
#' @param contrasts see \code{contrasts.arg} of \code{model.matrix.default}.
#' @description Performs forward selection, backward elimination, 
#' and efficient best subsets variable selection with information criterion for 
#' generalized linear models. Best subsets selection is performed with branch and 
#' bound algorithms to greatly speed up the process.
#' @details The supplied formula or the formula from the fitted model is 
#' treated as the upper model. The variables specified in keep along with an 
#' intercept (if included in formula) is the lower model. When an intercept is 
#' included in the model formula it is kept in each model. Factor 
#' variables are either kept in their entirety or entirely removed.
#' 
#' 
#' The branch and bound method makes use of an efficient branch and bound algorithm 
#' to find the optimal models. This is will find the best models according to the metric and 
#' can be much faster than an exhaustive search and can be made even faster with 
#' parallel computation. The backward branch and bound method is very similar to 
#' the branch and bound method, except it tends to be faster when the best models 
#' contain most of the variables. The switch branch and bound method is a 
#' combination of the two methods and is typically the fastest of the 3 branch and 
#' bound methods. 
#' 
#' Fisher's scoring is recommended for branch and bound selection and forward selection.
#' L-BFGS may be faster for backward elimination, especially when there are many variables.
#' 
#' All observations that have any missing values in the upper model are removed.
#' 
#' @return A \code{BranchGLMVS} object which is a list with the following components
#' \item{\code{initmodel}}{ the supplied \code{BranchGLM} object or a fake \code{BranchGLM}
#' object if a formula is supplied}
#' \item{\code{numchecked}}{ number of models fit}
#' \item{\code{names}}{ character vector of the names of the predictor variables}
#' \item{\code{order}}{ the order the variables were added to the model or removed from the model, this is not included for branch and bound selection}
#' \item{\code{type}}{ type of variable selection employed}
#' \item{\code{metric}}{ metric used to select best models}
#' \item{\code{bestmodels}}{ numeric matrix used to describe the best models}
#' \item{\code{bestmetrics}}{ numeric vector with the best metrics found in the search}
#' \item{\code{cutoff}}{ the supplied cutoff}
#' @name VariableSelection
#' 
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' VS
#' 
#' ## Getting summary of the process
#' Summ <- summary(VS)
#' Summ
#' 
#' ## Plotting the BIC of the best models
#' plot(Summ, type = "b")
#' 
#' ## Getting the best model according to BIC
#' FinalModel <- fit(Summ, which = 1)
#' FinalModel
#' 
#' # Now doing it in parallel (although it isn't necessary for this dataset)
#' parVS <- VariableSelection(Fit, type = "branch and bound", parallel = TRUE, metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' 
#' ## Getting the best model according to BIC
#' FinalModel <- fit(parVS, which = 1)
#' FinalModel
#' 
#' # Using a formula
#' formVS <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
#' link = "identity", metric = "BIC", type = "branch and bound", bestmodels = 10, showprogress = FALSE)
#' 
#' ## Getting the best model according to BIC
#' FinalModel <- fit(formVS, which = 1)
#' FinalModel
#' 
#' # Using the keep argument
#' keepVS <- VariableSelection(Fit, type = "branch and bound", keep = "Petal.Width", 
#' metric = "BIC", bestmodels = 5, showprogress = FALSE)
#' keepVS
#' 
#' ## Getting the fifth best model according to BIC when keeping Petal.Width in every model
#' FinalModel <- fit(keepVS, which = 5)
#' FinalModel
#' 
#' @export
#' 
VariableSelection <- function(object, ...) {
  UseMethod("VariableSelection")
}

#'@rdname VariableSelection
#'@export

VariableSelection.formula <- function(object, data, family, link, offset = NULL,
                                      method = "Fisher", type = "branch and bound", 
                                      metric = "AIC",
                                      bestmodels = 1, cutoff = 0, 
                                      keep = NULL, maxsize = NULL,
                                      grads = 10, parallel = FALSE, 
                                      nthreads = 8, tol = 1e-6, maxit = NULL,
                                      contrasts = NULL,
                                      showprogress = TRUE, ...){
  
  ### Creating pseudo BranchGLM object to use in VariableSelection.BranchGLM
  ### Validating supplied arguments
  formula <- object
  if(!is(formula, "formula")){
    stop("formula must be a valid formula")
  }
  if(!is.data.frame(data)){
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
  
  df$names <- attr(terms(df$formula, data = data), "factors") |>
    colnames()
  
  df$yname <- attr(terms(df$formula, data = data), "variables")[-1] |>
    as.character()
  
  df$yname <- df$yname[attr(terms(df$formula, data = data), "response")]
  
  df$missing <- nrow(data) - nrow(x)
  
  df$link <- link
  
  df$contrasts <- contrasts
  
  df$offset <- offset
  
  df$terms <- terms(df$formula, data = data)
  
  df$tol <- tol
  
  df$maxit <- maxit
  
  df$method <- method
  
  df$family <- family
  
  if(family == "binomial"){
    df$ylevel <- ylevel
  }
  
  fit <- structure(df, class = "BranchGLM")
  
  ### Performing variable selection
  VariableSelection(fit, type = type, metric = metric, 
                    bestmodels = bestmodels, cutoff = cutoff,
                    keep = keep, 
                    maxsize = maxsize, method = method, grads = grads, 
                    parallel = parallel, 
                    nthreads = nthreads, tol = tol, maxit = maxit,
                    showprogress = showprogress, ...)
}

#'@rdname VariableSelection
#'@export

VariableSelection.BranchGLM <- function(object, type = "branch and bound", metric = "AIC",
                                        bestmodels = 1, cutoff = 0, 
                                        keep = NULL, maxsize = NULL, 
                                        method = "Fisher", grads = 10, parallel = FALSE, 
                                        nthreads = 8, tol = 1e-6, maxit = NULL,
                                        showprogress = TRUE, ...){
  
  ## Checking if supplied BranchGLM object has x and data
  if(is.null(object$data) || is.null(object$x)){
    stop("the supplied model must have a data and an x component")
  }
  ## Checking if supplied BranchGLM object has y
  if(is.null(object$y)){
    stop("the supplied model must have a y component")
  }
    
  ## Validating supplied arguments
  if(length(parallel) > 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  
  ## Checking method
  if(length(method) != 1 || !(method %in% c("Fisher", "BFGS", "LBFGS"))){
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }
  
  ### checking nthreads
  if((length(nthreads) > 1) || !is.numeric(nthreads) || (nthreads <= 0)){
    warning("Please select a positive integer for nthreads, using nthreads = 8")
    nthreads <- 8
  }
  
  ### Checking grads
  if(length(grads) != 1 || !is.numeric(grads) || as.integer(grads) <= 0){
    stop("grads must be a positive integer")
  }
  
  ### Checking metric
  if(length(metric) > 1 || !is.character(metric)){
    stop("metric must be one of 'AIC','BIC', or 'HQIC'")
  }else if(!(metric %in% c("AIC", "BIC", "HQIC"))){
    stop("metric must be one of 'AIC','BIC', or 'HQIC'")
  }
  
  ### Checking bestmodels
  if(length(bestmodels) != 1 || !is.numeric(bestmodels) || 
     bestmodels <= 0 || bestmodels != as.integer(bestmodels)){
    stop("bestmodels must be a positive integer")
  }
  
  ### Checking cutoff
  if(length(cutoff) != 1 || !is.numeric(cutoff) || cutoff < 0){
    stop("cutoff must be a non-negative number")
  }else if(cutoff > 0 && bestmodels > 1){
    stop("only one of bestmodels or cutoff can be specified")
  }
  
  
  indices <- attr(object$x, "assign")
  interactions <- attr(object$terms, "factors")[-1L, ]
  
  ## Removing rows with all zeros
  interactions <- interactions[apply(interactions, 1, function(x){sum(x) > 0}),]
  
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
    interactions <- rbind(0, interactions)
    interactions <- cbind(0, interactions)
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
  
  ## Setting starting model and saving keep1 for later use since keep is modified
  keep1 <- keep
  if(is.null(keep) && type != "backward"){
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
    if(bestmodels > 1 || cutoff > 0){
      warning("forward selection only finds 1 final model")
    }
    df <- ForwardCpp(object$x, object$y, object$offset, indices, counts, interactions, method, grads,
                     object$link, object$family, nthreads, tol, maxit, keep, maxsize, 
                     metric)
    
  }else if(type == "backward"){
    if(bestmodels > 1 || cutoff > 0){
      warning("backward elimination only finds 1 final model")
    }
    df <- BackwardCpp(object$x, object$y, object$offset, indices, counts, interactions, method, grads,
                      object$link, object$family, nthreads, tol, maxit, keep, maxsize, 
                      metric)
  }else if(type == "branch and bound"){
    df <- BranchAndBoundCpp(object$x, object$y, object$offset, indices, counts, interactions, method, grads,
                                       object$link, object$family, nthreads, tol, maxit, keep, maxsize,
                                       metric, showprogress, bestmodels, cutoff)
  }else if(type == "backward branch and bound"){
    df <- BackwardBranchAndBoundCpp(object$x, object$y, object$offset, indices, counts, interactions, method, grads,
                                    object$link, object$family, nthreads, tol, maxit, keep, 
                                    metric, showprogress, bestmodels, cutoff)
  }else if(type == "switch branch and bound"){
    df <- SwitchBranchAndBoundCpp(object$x, object$y, object$offset, indices, counts, interactions, method, grads,
                                       object$link, object$family, nthreads, tol, maxit, keep, 
                                       metric, showprogress, bestmodels, cutoff)
  }else{
    stop("type must be one of 'forward', 'backward', 'branch and bound', 'backward branch and bound', or 'switch branch and bound'")
  }
  
  # Creating coefficient names
  names <- object$names
  if(intercept){
    names <- c("(Intercept)", names)
  }
  
  if(type %in% c("forward", "backward")){
    # Checking for infinite best metric value
    if(is.infinite(df$bestmetric)){
      stop("no models were found that had an invertible fisher information")
    }
    df$order <- df$order[df$order > 0]
    if(!intercept){
      df$order <- df$order + 1
    }
    FinalList <- list("numchecked" = df$numchecked,
                      "order" = object$names[df$order],
                      "type" = type, 
                      "metric" = metric,
                      "bestmodels" = df$bestmodel,
                      "bestmetrics" = df$bestmetric, 
                      "names" = names, 
                      "initmodel" = object)
  }else{
    # Checking for infinite best metric values
    if(all(is.infinite(df$bestmetrics))){
      stop("no models were found that had an invertible fisher information")
    }
    
    # Only returning best models that have a finite metric value
    bestInd <- is.finite(df$bestmetrics)
    
    # Only returning best models that are not the null model
    newInd <- colSums(abs(df$bestmodels)) != 0
    bestInd <- (newInd + bestInd) == 2
    
    
    
    FinalList <- list("numchecked" = df$numchecked,
                      "type" = type, 
                      "metric" = metric,
                      "bestmodels" = df$bestmodels[, bestInd],
                      "bestmetrics" = df$bestmetrics[bestInd], 
                      "names" = names, 
                      "initmodel" = object, 
                      "cutoff" = cutoff)
  }
  structure(FinalList, class = "BranchGLMVS")
}

#' @rdname fit 
#' @export
fit.BranchGLMVS <- function(object, which = 1, keepData = TRUE, keepY = TRUE, ...){
  fit(summary(object), which = which, keepData = keepData, keepY = keepY, ...)
}

#' @rdname plot.summary.BranchGLMVS 
#' @export
plot.BranchGLMVS <- function(x, ptype = "both", marx = 7, addLines = TRUE, 
                             type = "b", ...){
  plot(summary(x), ptype = ptype, marx = marx, addLines = addLines, type = type, ...)
}

#' Extract Coefficients
#' @param object a \code{BranchGLMVS} object.
#' @param which which models to get coefficients from, the default is the best model. 
#' Can specify "all" to get coefficients from all of the best models.
#' @param ... further arguments to \link{fit.BranchGLMVS}.
#' @return A numeric matrix with the corresponding coefficient estimates.
#' @export
coef.BranchGLMVS <- function(object, which = 1, ...){
  ## Checking which
  if(!is.numeric(which) && is.character(which) && length(which) == 1){
    if(which == "all"){
      which <- 1:NCOL(object$bestmodels)
    }
    else{
      stop("which must be a sequence of positive integers or 'all'.")
    }
  }else if(any(which < 1)){
    stop("integers provided in which must be positive")
  }else if(any(which > NCOL(object$bestmodels))){
    stop("integers provided in which must be less than or equal to the number of best models")
  }
  
  ## Getting coefficients from all models in which
  allcoefs <- sapply(which, function(i){
    ## Padding coefficients with zeros
    if(is.matrix(object$bestmodels)){
      model <- object$bestmodels[, i]
    }else{
      model <- object$bestmodels
    }
    coefs <- rep(0, ncol(object$initmodel$x))
    names(coefs) <- colnames(object$initmodel$x)
    tempcoefs <- coef(fit(object, which = i, ...))
    coefs[names(tempcoefs)] <- tempcoefs
    
    return(coefs)
  })
  
  ## Adding column names to identify each model
  colnames(allcoefs) <- paste0("Model", which)
  return(allcoefs)
}

#' Predict Method for BranchGLMVS Objects
#' @param object a \code{BranchGLMVS} object.
#' @param newdata a dataframe, if not specified then the data the model was fit on is used.
#' @param type one of "linpreds" or "response", if not specified then "response" is used.
#' @param which which model to get predictions from, the default is the best model.
#' @param ... further arguments passed to \link{fit.BranchGLMVS}.
#' @details linpreds corresponds to the linear predictors and response is on the scale of the response variable.
#' Offset variables are ignored for predictions on new data.
#' @description Gets predictions from a \code{BranchGLMVS} object.
#' @return A numeric vector of predictions.
#' @export
predict.BranchGLMVS <- function(object, newdata = NULL, type = "response", which = 1, ...){
  ## Checking which
  if(!is.numeric(which) || length(which) > 1){
    stop("which must be a positive integer")
  }
  
  ### Getting BranchGLM object
  myfit <- fit(object, which = which, ...)
  
  ### Getting predictions
  predict(myfit, newdata = newdata, type = type)
}

#' Print Method for BranchGLMVS
#' @param x a \code{BranchGLMVS} object.
#' @param coefdigits number of digits to display for coefficients table.
#' @param digits number of digits to display for information not in the table.
#' @param ... further arguments passed to other methods.
#' @return The supplied \code{BranchGLMVS} object.
#' @export

print.BranchGLMVS <- function(x, coefdigits = 4, digits = 2, ...){
  
  cat("Variable Selection Info:\n")
  cat(paste0(rep("-", 24), collapse = ""))
  cat("\n")
  if(x$type != "backward"){
    cat(paste0("Variables were selected using ", x$type, " selection with ", x$metric, "\n"))
  }else{
    cat(paste0("Variables were selected using ", x$type, " elimination with ", x$metric, "\n"))
  }
  if(length(x$bestmetrics) == 1){
  cat(paste0("The best value of ", x$metric, " obtained was ", 
             round(x$bestmetrics[1], digits = digits), "\n"))
  }else{
    cat(paste0("Found the top ", length(x$bestmetrics), " models\n"))
    cat(paste0("The range of ", x$metric, " values for the top ", length(x$bestmetrics), 
               " models is (", round(x$bestmetrics[1], digits = digits), 
               ", ", round(x$bestmetrics[length(x$bestmetrics)], digits = digits), ")\n"))
    
  }
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
  
  invisible(x)
}
