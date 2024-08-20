#' Fits GLMs
#' @description Fits generalized linear models (GLMs) via RcppArmadillo with the 
#' ability to perform some computation in parallel with OpenMP.
#' @param formula a formula for the model.
#' @param data an optional data.frame, list or environment (or object coercible by 
#' [as.data.frame] to a data.frame), containing the variables in formula. 
#' Neither a matrix nor an array will be accepted. If not found in `data`, the 
#' variables are taken from `environment(formula)`, typically the environment from 
#' which `BranchGLM` is called.
#' @param family the distribution used to model the data, one of "gaussian", 
#' "gamma", "binomial", or "poisson". A `family` object may also be supplied for 
#' one of the accepted distributions.
#' @param link the link used to link the mean structure to the linear predictors. One of
#' "identity", "logit", "probit", "cloglog", "sqrt", "inverse", or "log". The accepted 
#' links depend on the specified family, see more in details. This only needs to be 
#' supplied if a string is supplied for `family`. If a `family` object is supplied 
#' for the `family` argument, then the link function is taken from that `family` 
#' object.
#' @param offset the offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS". BFGS and L-BFGS are 
#' quasi-newton methods which are typically faster than Fisher's scoring when
#' there are many covariates (at least 50).
#' @param grads a positive integer to denote the number of gradients used to 
#' approximate the inverse information with, only for `method = "LBFGS"`.
#' @param parallel a logical value to indicate if parallelization should be used.
#' @param nthreads a positive integer to denote the number of threads used with OpenMP, 
#' only used if `parallel = TRUE`.
#' @param tol a positive number to denote the tolerance used to determine model convergence.
#' @param maxit a positive integer to denote the maximum number of iterations performed. 
#' The default for Fisher's scoring is 50 and for the other methods the default is 200.
#' @param init a numeric vector of initial values for the betas, if not specified 
#' then they are automatically selected via linear regression with the transformation 
#' specified by the link function. This is ignored for linear regression models.
#' @param fit a logical value to indicate whether to fit the model or not.
#' @param keepData a logical value to indicate whether or not to store a copy of 
#' data and the design matrix, the default is TRUE. If this is FALSE, then the 
#' results from this cannot be used inside of `VariableSelection`.
#' @param keepY a logical value to indicate whether or not to store a copy of y, 
#' the default is TRUE. If this is FALSE, then the binomial GLM helper functions 
#' may not work and this cannot be used inside of `VariableSelection`.
#' @param contrasts see `contrasts.arg` of `model.matrix.default`.
#' @param x design matrix used for the fit, must be numeric.
#' @param y outcome vector, must be numeric.
#' @seealso [predict.BranchGLM], [coef.BranchGLM], [VariableSelection], [confint.BranchGLM], [logLik.BranchGLM]
#' @return `BranchGLM` returns a `BranchGLM` object which is a list with the following components
#' \item{`coefficients`}{ a data.frame with the coefficient estimates, SEs, Wald test statistics, and p-values}
#' \item{`iterations`}{ number of iterations it took the algorithm to converge, if the algorithm failed to converge then this is -1}
#' \item{`dispersion`}{ a vector of length 2 with the MLE of the dispersion parameter first and the Pearson estimator of the dispersion parameter second}
#' \item{`logLik`}{ the log-likelihood of the fitted model}
#' \item{`vcov`}{ the variance-covariance matrix of the fitted model}
#' \item{`resDev`}{ the residual deviance of the fitted model}
#' \item{`AIC`}{ the AIC of the fitted model}
#' \item{`preds`}{ predictions from the fitted model}
#' \item{`linpreds`}{ linear predictors from the fitted model}
#' \item{`residuals`}{ a numeric vector with the Pearson residuals}
#' \item{`variance`}{ a numeric vector with the variance evaluated at the final coefficient estimates}
#' \item{`tol`}{ tolerance used to fit the model}
#' \item{`maxit`}{ maximum number of iterations used to fit the model}
#' \item{`formula`}{ formula used to fit the model}
#' \item{`method`}{ iterative method used to fit the model}
#' \item{`grads`}{ number of gradients used to approximate inverse information for L-BFGS}
#' \item{`y`}{ y vector used in the model, not included if `keepY = FALSE`}
#' \item{`x`}{ design matrix used to fit the model, not included if `keepData = FALSE`}
#' \item{`rownames`}{ rownames taken from the design matrix}
#' \item{`offset`}{ offset vector in the model, not included if `keepData = FALSE`}
#' \item{`fulloffset`}{ supplied offset vector, not included if `keepData = FALSE`}
#' \item{`data`}{ original `data` argument supplied to the function, not included if `keepData = FALSE`}
#' \item{`mf`}{ the model frame, not included if `keepData = FALSE`}
#' \item{`numobs`}{ number of observations in the design matrix}
#' \item{`names`}{ names of the predictor variables}
#' \item{`yname`}{ name of y variable}
#' \item{`parallel`}{ whether parallelization was employed to speed up model fitting process}
#' \item{`missing`}{ number of missing values removed from the original dataset}
#' \item{`link`}{ link function used to model the data}
#' \item{`family`}{ family used to model the data}
#' \item{`ylevel`}{ the levels of y, only included for binomial glms}
#' \item{`xlev`}{ the levels of the factors in the dataset}
#' \item{`terms`}{the terms object used}
#' 
#' `BranchGLM.fit` returns a list with the following components
#' \item{`coefficients`}{ a data.frame with the coefficients estimates, SEs, Wald test statistics, and p-values}
#' \item{`iterations`}{ number of iterations it took the algorithm to converge, if the algorithm failed to converge then this is -1}
#' \item{`dispersion`}{ a vector of length 2 with the MLE of the dispersion parameter first and the Pearson estimator of the dispersion parameter second}
#' \item{`logLik`}{ the log-likelihood of the fitted model}
#' \item{`vcov`}{ the variance-covariance matrix of the fitted model}
#' \item{`resDev`}{ the residual deviance of the fitted model}
#' \item{`AIC`}{ the AIC of the fitted model}
#' \item{`preds`}{ predictions from the fitted model}
#' \item{`linpreds`}{ linear predictors from the fitted model}
#' \item{`residuals`}{ a numeric vector with the Pearson residuals}
#' \item{`variance`}{ a numeric vector with the variance evaluated at the final coefficient estimates}
#' \item{`tol`}{ tolerance used to fit the model}
#' \item{`maxit`}{ maximum number of iterations used to fit the model}
#' @details 
#' 
#' ## Fitting
#' Can use BFGS, L-BFGS, or Fisher's scoring to fit the GLM. BFGS and L-BFGS are 
#' typically faster than Fisher's scoring when there are at least 50 covariates 
#' and Fisher's scoring is typically best when there are fewer than 50 covariates.
#' This function does not currently support the use of weights. In the special 
#' case of gaussian regression with identity link the `method` argument is ignored
#' and the normal equations are solved directly.
#' 
#' The models are fit in C++ by using Rcpp and RcppArmadillo. In order to help 
#' convergence, each of the methods makes use of a backtracking line-search using 
#' the strong Wolfe conditions to find an adequate step size. There are 
#' three conditions used to determine convergence, the first is whether there is a 
#' sufficient decrease in the negative log-likelihood, the second is whether 
#' the l2-norm of the score is sufficiently small, and the last condition is 
#' whether the change in each of the beta coefficients is sufficiently 
#' small. The `tol` argument controls all of these criteria. If the algorithm fails to 
#' converge, then `iterations` will be -1.
#' 
#' All observations with any missing values are removed before model fitting. 
#' 
#' `BranchGLM.fit` can be faster than calling `BranchGLM` if the 
#' x matrix and y vector are already available, but doesn't return as much information.
#' The object returned by `BranchGLM.fit` is not of class `BranchGLM`, so 
#' all of the methods for `BranchGLM` objects such as `predict` or 
#' `VariableSelection` cannot be used.
#' 
#' ## Dispersion Parameter
#' The dispersion parameter for binomial and Poisson regression is always fixed to be 1.
#' For gaussian and gamma regression, the MLE of the dispersion parameter is used 
#' for the calculation of the log-likelihood and the Pearson estimator of the dispersion parameter 
#' is used for the calculation of standard errors for the coefficient estimates.
#' 
#' ## Families and Links
#' The binomial family accepts "cloglog", "log", "logit", and "probit" as possible 
#' link functions. The gamma and gaussian families accept "identity", "inverse", 
#' "log", and "sqrt" as possible link functions. The Poisson family accepts "identity", 
#' "log", and "sqrt" as possible link functions.
#' 
#' @examples
#' Data <- iris
#' 
#' # Linear regression
#' ## Using BranchGLM
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ## Using BranchGLM.fit
#' x <- model.matrix(Sepal.Length ~ ., data = Data)
#' y <- Data$Sepal.Length
#' BranchGLM.fit(x, y, family = "gaussian", link = "identity")
#' 
#' # Gamma regression
#' ## Using BranchGLM
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gamma", link = "log")
#' 
#' ### init
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gamma", link = "log", 
#' init = rep(0, 6), maxit = 50, tol = 1e-6, contrasts = NULL)
#' 
#' ### method
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gamma", link = "log", 
#' init = rep(0, 6), maxit = 50, tol = 1e-6, contrasts = NULL, method = "LBFGS")
#' 
#' ### offset
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gamma", link = "log", 
#' init = rep(0, 6), maxit = 50, tol = 1e-6, contrasts = NULL, 
#' offset = Data$Sepal.Width)
#' 
#' ## Using BranchGLM.fit
#' x <- model.matrix(Sepal.Length ~ ., data = Data)
#' y <- Data$Sepal.Length
#' BranchGLM.fit(x, y, family = "gamma", link = "log", init = rep(0, 6), 
#' maxit = 50, tol = 1e-6, offset = Data$Sepal.Width)
#' 
#' 
#' @references McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.). 
#' Chapman & Hall.
#' @export

BranchGLM <- function(formula, data = NULL, family, link, offset = NULL, 
                    method = "Fisher", grads = 10,
                    parallel = FALSE, nthreads = 8, 
                    tol = 1e-6, maxit = NULL, init = NULL, fit = TRUE, 
                    contrasts = NULL, keepData = TRUE,
                    keepY = TRUE){
  
  ### converting family, link, and method to lower
  if(is(family, "family")){
    link <- family$link
    family <- family$family
  }else if(!is.character(family)){
    stop("family must be an object of class 'family' or 'character'") 
  }
  family <- tolower(family)
  link <- tolower(link)
  method <- tolower(method)
  
  ### Validating supplied arguments
  if(!is(formula, "formula")){
    stop("formula must be a valid formula")
  }
  
  if(length(method) != 1 || !is.character(method)){
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }else if(method == "fisher"){
    method <- "Fisher"
  }else if(method == "bfgs"){
    method <- "BFGS"
  }else if(method == "lbfgs"){
    method <- "LBFGS"
  }else{
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }
  if(length(family) != 1 || !family %in% c("gaussian", "binomial", "poisson", "gamma")){
    stop("family must be one of 'gaussian', 'binomial', 'gamma', or 'poisson'")
  }
  if(length(link) != 1 ||!link %in% c("logit", "probit", "cloglog", "log", "identity", "inverse", "sqrt")){
    stop("link must be one of 'logit', 'probit', 'cloglog', 'log', 'inverse', 'sqrt', or 'identity'")
  }
  
  ### Evaluating arguments
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf$na.action <- "na.omit"
  mf[[1L]] <- quote(model.frame)
  if(is.null(data)){
    tempmf <- mf
    tempmf$na.action <- "na.pass"
    data <- eval(tempmf, parent.frame())
  }
  mf <- eval(mf, parent.frame())
  
  ## Getting data objects
  y <- model.response(mf, "any")
  fulloffset <- offset
  offset <- as.vector(model.offset(mf))
  x <- model.matrix(attr(mf, "terms"), mf, contrasts)
  
  if(is.null(offset)){
    offset <- rep(0, length(y))
  }
  
  
  ## Checking y variable for binomial family
  if(tolower(family) == "binomial"){
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
  }
  
  ## Getting maxit
  if(is.null(maxit)){
    if(method == "Fisher"){
      maxit = 50
    }else{
      maxit = 200
    }
  }
  
  ### Using BranchGLM.fit to fit GLM
  if(fit){
    df <- BranchGLM.fit(x, y, family, link, offset, method, grads, parallel, nthreads, 
                        init, maxit, tol)
    df$residuals <- as.vector(df$residuals)
    df$dispersion <- as.vector(df$dispersion)
  }else{
    df <- list("coefficients" = matrix(NA, nrow = ncol(x), ncol = 4), 
               "vcov" = matrix(NA, nrow = ncol(x), ncol = ncol(x)))
    colnames(df$coefficients) <- c("Estimate", "SE", "z", "p-values")
  }
  # Setting names for coefficients
  row.names(df$coefficients) <- colnames(x)
  
  # Setting names for vcov
  rownames(df$vcov) <- colnames(df$vcov) <- colnames(x)
  
  df$formula <- formula
  
  df$method <- method
  
  if(keepY){
    df$y <- y
  }
  
  df$numobs <- nrow(x)
  
  if(keepData){
    df$data <- data
    df$x <- x
    df$mf <- mf
    df$offset <- offset
    df$fulloffset <- fulloffset
  }
  df$names <- attributes(terms(formula, data = data))$factors |>
              colnames()
  
  df$yname <- attributes(terms(formula, data = data))$variables[-1] |>
              as.character()
  
  df$yname <- df$yname[attributes(terms(formula, data = data))$response]
  
  df$parallel <- parallel
  
  df$missing <- nrow(data) - nrow(x)
  
  df$link <- link
  
  df$contrasts <- contrasts
  
  df$family <- family
  
  df$terms <- attr(mf, "terms")
  
  df$xlev <- .getXlevels(df$terms, mf)
  
  df$grads <- grads
  
  df$tol <- tol
  
  df$maxit <- maxit
  
  if(family == "binomial"){
    df$ylevel <- ylevel
  }
  
  if(!is.null(rownames(x))){
    df$rownames <- rownames(x)
  }else{
    df$rownames <- 1:nrow(x)
  }
  
  structure(df, class = "BranchGLM")
}

#' @rdname BranchGLM
#' @export
BranchGLM.fit <- function(x, y, family, link, offset = NULL,
                          method = "Fisher", grads = 10,
                          parallel = FALSE, nthreads = 8, init = NULL,  
                          maxit = NULL, tol = 1e-6){
  ### converting family, link, and method to lower
  family <- tolower(family)
  link <- tolower(link)
  method <- tolower(method)
  
  ### Getting method
  if(length(method) != 1 || !is.character(method)){
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }else if(method == "fisher"){
    method <- "Fisher"
  }else if(method == "bfgs"){
    method <- "BFGS"
  }else if(method == "lbfgs"){
    method <- "LBFGS"
  }else{
    stop("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }
  
  ## Performing a few checks
  if(!is.matrix(x) || !is.numeric(x)){
    stop("x must be a numeric matrix")
  }else if(!is.numeric(y)){
    stop("y must be numeric")
  }else if(nrow(x) != length(y)){
    stop("the number of rows in x must be the same as the length of y")
  }else if(nrow(x) == 0){
    stop("design matrix x has no rows and y has a length of 0")
  }
  
  ## Checking grads and tol
  if(length(grads) != 1 || !is.numeric(grads) || as.integer(grads) <= 0){
    stop("grads must be a positive integer")
  }
  if(length(tol) != 1 || !is.numeric(tol) || tol <= 0){
    stop("tol must be a positive number")
  }
  
  ## Getting maxit
  if(is.null(maxit)){
    if(method == "Fisher"){
      maxit = 50
    }else{
      maxit = 200
    }
  }else if(length(maxit) != 1 || !is.numeric(maxit) || maxit <= 0){
    stop("maxit must be a positive integer")
  }
  
  ## Getting initial values
  if(is.null(init)){
    init <- rep(0, ncol(x))
    GetInit <- TRUE
  }else if(!is.numeric(init) || length(init) != ncol(x)){
    stop("init must be null or a numeric vector with length equal to the number of betas")
  }else if(any(is.infinite(init)) || any(is.na(init))){
    stop("init must not contain any infinite values, NAs, or NaNs")
  }else{
    GetInit <- FALSE
  }
  
  ## Checking y variable and link function for each family
  if(family == "binomial"){
    if(!(link %in% c("cloglog", "log", "logit", "probit"))){
      stop("valid link functions for binomial regression are 'cloglog', 'log', 'logit', and 'probit'")
    }else if(!all(y %in% c(0, 1))){
      stop("for binomial regression y must be a vector of 0s and 1s")
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
  }else{
    stop("the supplied family is not supported")
  }
  
  ## Getting offset
  if(is.null(offset)){
    offset <- rep(0, length(y))
  }else if(length(offset) != length(y)){
    stop("offset must be the same length as y")
  }else if(!is.numeric(offset)){
    stop("offset must be a numeric vector")
  }else if(any(is.infinite(offset)) || any(is.na(offset))){
    stop("offset must not contain any infinite values, NAs, or NaNs")
  }
  if(length(nthreads) != 1 || !is.numeric(nthreads) || is.na(nthreads) || nthreads <= 0){
    stop("nthreads must be a positive integer")
  }
  if(length(parallel) != 1 || !is.logical(parallel) || is.na(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }else if(parallel){
    df <- BranchGLMfit(x, y, offset, init, method, grads, link, family, nthreads, 
                       tol, maxit, GetInit) 
  }else{
    df <- BranchGLMfit(x, y, offset, init, method, grads, link, family, 1, tol, maxit, 
                       GetInit) 
  }
  
  if((family == "gaussian" || family == "gamma")){
    colnames(df$coefficients)[1:2] <- c("Estimate", "Std. Error")
    colnames(df$coefficients)[3] <- "t value"
    colnames(df$coefficients)[4] <- "Pr(>|t|)"
  }else{
    colnames(df$coefficients)[1:2] <- c("Estimate", "Std. Error")
    colnames(df$coefficients)[3] <- "z value"
    colnames(df$coefficients)[4] <- "Pr(>|z|)"
  }
  
  df$tol <- tol
  df$maxit <- maxit
  
  return(df)
}

#' Extract Model Formula from BranchGLM Objects
#' @description Extracts model formula from `BranchGLM` objects.
#' @param x a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A formula representing the model used to obtain `x`.
#' @export
formula.BranchGLM <- function(x, ...){
  return(x$formula) 
}

#' Extract Model Frame from a BranchGLM Object
#' @description Extracts model frame from `BranchGLM` objects.
#' @param formula a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A [data.frame] used for the fitted model.
#' @export
model.frame.BranchGLM <- function(formula, ...){
  return(formula$mf) 
}

#' Extract Family from BranchGLM Objects
#' @description Extracts family from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return An object of class `family`.
#' @note
#' Only the `family` and `link` components of `family` objects are used by the 
#' `BranchGLM` package. The other components of the `family` object may not 
#' be the same as what is used in the fitting process for `BranchGLM`.
#' @export
family.BranchGLM <- function(object, ...){
  myfamily <- object$family
  if(myfamily == "gamma"){
    myfamily <- "Gamma"
  }
  return(do.call(myfamily, list("link" = object$link))) 
}

#' Extract Number of Observations from BranchGLM Objects
#' @description Extracts number of observations from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A single number indicating the number of observations used to fit the model.
#' @export
nobs.BranchGLM <- function(object, ...){
  return(object$numobs) 
}

#' Extract Log-Likelihood from BranchGLM Objects
#' @description Extracts log-likelihood from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return An object of class `logLik` which is a number corresponding to 
#' the log-likelihood with the following attributes: "df" (degrees of freedom) 
#' and "nobs" (number of observations).
#' @export
logLik.BranchGLM <- function(object, ...){
  df <- length(coef(object))
  if(object$family == "gaussian" || object$family == "gamma"){
    df <- df + 1
  }
  val <- object$logLik
  attr(val, "nobs") <- nobs(object)
  attr(val, "df") <- df
  class(val) <- "logLik"
  return(val)
}

#' Extract AIC from BranchGLM Objects
#' @description Computes the (generalized) Akaike An Information Criterion for 
#' BranchGLM objects.
#' @param fit a `BranchGLM` object.
#' @param scale ignored.
#' @param k a non-negative number specifying the ‘weight’ of the equivalent degrees 
#' of freedom (= edf) part in the AIC formula.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric vector of length 2, with first and second elements giving
#' \item{`edf`}{ the ‘equivalent degrees of freedom’ for `fit`}
#' \item{`AIC`}{ the (generalized) Akaike Information Criterion for `fit`}
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' extractAIC(Fit)
#' 
#' @export
extractAIC.BranchGLM <- function(fit, scale, k = 2, ...){
  if(!is.numeric(k) || length(k) != 1 || k < 0){
    stop("k must be a non-negative number")
  }
  loglik <- logLik(fit)
  edf <- attr(loglik, "df")
  return(c(edf, -2 * loglik[[1]] + k * edf))
}

#' Extract the Deviance
#' @description Extracts the deviance from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A single number denoting the deviance.
#' @export
deviance.BranchGLM <- function(object, ...){
  return(object$resDev)
}

#' Extract covariance matrix from BranchGLM Objects
#' @description Extracts covariance matrix of beta coefficients from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric matrix which is the covariance matrix of the beta coefficients.
#' @export
vcov.BranchGLM <- function(object, ...){
  return(object$vcov)
}

#' Extract Square Root of the Dispersion Parameter Estimates
#' @description Extracts the square root of the dispersion parameter estimates from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric vector of length 2 with first and second elements giving
#' \item{`mle`}{ the MLE of the dispersion parameter}
#' \item{`pearson`}{ the Pearson estimator of the dispersion parameter}
#' @note
#' The dispersion parameter for binomial and Poisson regression is always fixed to be 1.
#' The MLE of the dispersion parameter is used in the calculation of the log-likelihood 
#' while the Pearson estimator of the dispersion parameter is used to calculate 
#' standard errors for the coefficient estimates.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' sigma(Fit)
#' 
#' @export
sigma.BranchGLM <- function(object, ...){
  temp <- sqrt(object$dispersion)
  names(temp) <- c("mle", "pearson")
  return(temp)
}

#' Extract the Pearson Residuals from BranchGLM Objects
#' @description Extracts the Pearson residuals from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A vector of the Pearson residuals from the supplied `BranchGLM` object.
#' @export
residuals.BranchGLM <- function(object, ...){
  res <- object$residuals
  names(res) <- object$rownames
  return(res)
}

#' Extract Coefficients from BranchGLM Objects
#' @description Extracts beta coefficients from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param type "estimates" to only return the coefficient estimates or "all" to 
#' return the estimates along with SEs, test statistics, and p-values.
#' @param ... further arguments passed to or from other methods.
#' @return A named vector with the corresponding coefficient estimates or a 
#' data.frame with the coefficient estimates along with SEs, test statistics, and p-values.
#' @examples
#' Data <- iris
#' 
#' # Linear regression model
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' Fit
#' 
#' # Getting only coefficient estimates
#' coef(Fit, type = "estimates")
#' 
#' # Getting coefficient estimates along with SEs, tests, and p-values
#' coef(Fit, type = "all")
#' 
#' @export

coef.BranchGLM <- function(object, type = "estimates", ...){
  type <- tolower(type)
  if(type == "estimates"){
    coefs <- object$coefficients[,1]
    names(coefs) <- row.names(object$coefficients)
  }else if(type == "all"){
    coefs <- object$coefficients
  }else{
    stop("type must be 'estimates' or 'all'")
  }
  return(coefs)
}

#' Predict Method for BranchGLM Objects
#' @description Obtains predictions from `BranchGLM` objects.
#' @param object a `BranchGLM` object.
#' @param newdata a data.frame, if not specified then the data the model was fit on is used.
#' @param offset a numeric vector containing the offset variable, this is ignored if 
#' newdata is not supplied.
#' @param type one of "linpreds" which is on the scale of the linear predictors or 
#' "response" which is on the scale of the response variable. If not specified, 
#' then "response" is used.
#' @param na.action a function which indicates what should happen when the data 
#' contains NAs. The default is `na.pass`. This is ignored if newdata is not 
#' supplied and data isn't included in the supplied `BranchGLM` object.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric vector of predictions.
#' @examples
#' Data <- airquality
#' 
#' # Example without offset
#' Fit <- BranchGLM(Temp ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ## Using default na.action
#' predict(Fit)
#' 
#' ## Using na.omit
#' predict(Fit, na.action = na.omit)
#' 
#' ## Using new data
#' predict(Fit, newdata = Data[1:20, ], na.action = na.pass)
#' 
#' # Using offset
#' FitOffset <- BranchGLM(Temp ~ . - Day, data = Data, family = "gaussian", 
#' link = "identity", offset = Data$Day * -0.1)
#' 
#' ## Getting predictions for data used to fit model
#' ### Don't need to supply offset vector
#' predict(FitOffset)
#' 
#' ## Getting predictions for new dataset
#' ### Need to include new offset vector since we are 
#' ### getting predictions for new dataset
#' predict(FitOffset, newdata = Data[1:20, ], offset = Data$Day[1:20] * -0.1)
#' 
#' @export

predict.BranchGLM <- function(object, newdata = NULL, offset = NULL, 
                              type = "response", na.action = na.pass, ...){
  
  if(!is.null(newdata) && !is(newdata, "data.frame")){
    stop("newdata argument must be a data.frame or NULL")
  }
  
  if(length(type) != 1 ){
    stop("type must have a length of 1")
  }else if(!(type %in% c("linpreds", "response"))){
    stop("type argument must be either 'linpreds' or 'response'")
  }
  
  if(is.null(newdata) && !is.null(object$data)){
    newdata <- object$data
    offset <- object$fulloffset
  }else if(is.null(newdata) && is.null(object$data)){
    if(type == "linpreds"){
      linpreds <- object$linpreds
      names(linpreds) <- object$rownames
      return(linpreds)
    }else if(type == "response"){
      preds <- object$preds
      names(preds) <- object$rownames
      return(preds)
    }
  }
  
  # Changing environment for formula and offset since we need them to be the same
  if(is.null(offset)){
    if(!is.null(newdata) && !is.null(object$fulloffset) && any(object$fulloffset != 0)){
      warning("offset should be supplied for new dataset")
    }
    offset2 <- rep(0, nrow(newdata))
  }else{
    offset2 <- offset
  }
  environment(offset2) <- environment()
  
  # Getting mf
  myterms <- delete.response(terms(object))
  environment(myterms) <- environment()
  m <- model.frame(myterms, data = newdata, na.action = na.action,
                   xlev = object$xlev, offset = offset2)
  
  # Getting offset and x
  offset <- model.offset(m)
  environment(offset) <- NULL
  x <- model.matrix(myterms, m, contrasts = object$contrasts)
  
  
  if(ncol(x) != length(coef(object))){
    stop("could not find all predictor variables in newdata")
  }else if(tolower(type) == "linpreds"){
    preds <- drop(x %*% coef(object) + offset) |> unname()
    names(preds) <- rownames(x)
    return(preds)
  }else if(tolower(type) == "response"){
    preds <- GetPreds(drop(x %*% coef(object) + offset) |> unname(), object$link)
    names(preds) <- rownames(x)
    return(preds)
  }
}

#' Get Predictions
#' @param linpreds numeric vector of linear predictors.
#' @param link the specified link.
#' @noRd

GetPreds <- function(linpreds, Link){
  if(Link == "log"){
    exp(linpreds)
  }
  else if(Link == "logit"){
    1 / (1 + exp(-linpreds))
  }
  else if(Link == "probit"){
    pnorm(linpreds)
  }
  else if(Link == "cloglog"){
    1 - exp(-exp(linpreds))
  }
  else if(Link == "inverse"){
     1 / (linpreds)
  }
  else if(Link == "identity"){
    linpreds
  }
  else{
    linpreds^2
  }
}

#' Plot Method for BranchGLM Objects
#' @description Creates a plot to help visualize fitted values from `BranchGLM` objects.
#' @param x a `BranchGLM` object.
#' @param ... further arguments passed to [plot.default]. 
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' plot(Fit)
#' 
#' @return This only produces a plot, nothing is returned.
#' @export

plot.BranchGLM <- function(x, ...){
  # Checking for y
  if(is.null(x$y)){
    stop("plot can only be used when keepY = TRUE")  
  }else if(is.null(x$preds)){
    stop("plot can only be used when fit = TRUE")
  }
  
  # Plotting fitted values
  plot(x$preds, x$y, xlab = "Fitted Values", ylab = x$yname, ...)
}

#' Print Method for BranchGLM Objects
#' @description Print method for `BranchGLM` objects.
#' @param x a `BranchGLM` object.
#' @param coefdigits number of digits to display for coefficients table.
#' @param digits number of significant digits to display for information after table.
#' @param ... further arguments passed to or from other methods.
#' @return The supplied `BranchGLM` object.
#' @export

print.BranchGLM <- function(x, coefdigits = 4, digits = 4, ...){
  if(length(coefdigits)!= 1  || !is.numeric(coefdigits) || coefdigits < 0){
    stop("coefdigits must be a non-negative number")
  }
  if(length(digits)!= 1  || !is.numeric(digits) || digits < 0){
    stop("coefdigits must be a non-negative number")
  }
  cat(paste0("Results from ", x$family, " regression with ", x$link, 
             " link function \nUsing the formula ", deparse1(x$formula), "\n\n"))
  
  printCoefmat(signif(x$coefficients, digits = coefdigits), signif.stars = TRUE, P.values = TRUE, 
               has.Pvalue = TRUE)
  
  cat(paste0("\nDispersion parameter taken to be ", 
             format(x$dispersion[1], digits = digits), " for the log-likelihood"))
  cat(paste0("\nDispersion parameter taken to be ", 
             format(x$dispersion[2], digits = digits), " for standard errors"))
  cat(paste0("\n", x$numobs, " observations used to fit the model\n(", x$missing, 
             " observations removed due to missingness)\n"))
  cat(paste0("\nResidual Deviance: ", format(x$resDev, digits = digits), " on ",
             x$numobs - nrow(x$coefficients), " degrees of freedom"))
  cat(paste0("\nAIC: ", format(x$AIC, digits = digits)))
  if(x$family != "gaussian" || x$link != "identity"){
    if(x$method == "Fisher"){
      method = "Fisher's scoring"
    }else if(x$method == "LBFGS"){
      method = "L-BFGS"
    }else{method = "BFGS"}
    if(x$iterations == 1){
      cat(paste0("\nAlgorithm converged in 1 iteration using ", method, "\n"))
    }else if(x$iterations > 1 || x$iterations == 0){
      cat(paste0("\nAlgorithm converged in ", x$iterations, " iterations using ", method, "\n"))
    }else{
      cat("\nAlgorithm failed to converge\n")
    }
  }else{
    cat("\n")
  }
  
  if(x$parallel){
    cat("Parallel computation was used to speed up model fitting process")
  }
  invisible(x)
}
