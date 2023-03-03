#' Fits GLMs
#' @param formula a formula for the model.
#' @param data a dataframe that contains the response and predictor variables.
#' @param family distribution used to model the data, one of "gaussian", "gamma", "binomial", or "poisson".
#' @param link link used to link mean structure to linear predictors. One of
#' "identity", "logit", "probit", "cloglog", "sqrt", "inverse", or "log".
#' @param offset offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS". BFGS and L-BFGS are 
#' quasi-newton methods which are typically faster than Fisher's scoring when
#' there are many covariates (at least 50).
#' @param grads number of gradients used to approximate inverse information with, only for \code{method = "LBFGS"}.
#' @param parallel whether or not to make use of parallelization via OpenMP.
#' @param nthreads number of threads used with OpenMP, only used if \code{parallel = TRUE}.
#' @param tol tolerance used to determine model convergence.
#' @param maxit maximum number of iterations performed. The default for 
#' Fisher's scoring is 50 and for the other methods the default is 200.
#' @param init initial values for the betas, if not specified then they are automatically 
#' selected.
#' @param keepData Whether or not to store a copy of data and design matrix, the default 
#' is TRUE. If this is FALSE, then the results from this cannot be used inside of \code{VariableSelection}.
#' @param keepY Whether or not to store a copy of y, the default is TRUE. If 
#' this is FALSE, then the binomial GLM helper functions may not work and this 
#' cannot be used inside of \code{VariableSelection}.
#' @param contrasts see \code{contrasts.arg} of \code{model.matrix.default}.
#' @param x design matrix used for the fit, must be numeric.
#' @param y outcome vector, must be numeric.
#' @return \code{BranchGLM} returns a \code{BranchGLM} object which is a list with the following components
#' \item{\code{coefficients}}{ a matrix with the coefficients estimates, SEs, wald test statistics, and p-values}
#' \item{\code{iterations}}{ number of iterations it took the algorithm to converge, if the algorithm failed to converge then this is -1}
#' \item{\code{dispersion}}{ the value of the dispersion parameter}
#' \item{\code{logLik}}{ the log-likelihood of the fitted model}
#' \item{\code{vcov}}{ the variance-covariance matrix of the fitted model}
#' \item{\code{resdev}}{ the residual deviance of the fitted model}
#' \item{\code{AIC}}{ the AIC of the fitted model}
#' \item{\code{preds}}{ predictions from the fitted model}
#' \item{\code{linpreds}}{ linear predictors from the fitted model}
#' \item{\code{tol}}{ tolerance used to fit the model}
#' \item{\code{maxit}}{ maximum number of iterations used to fit the model}
#' \item{\code{formula}}{ formula used to fit the model}
#' \item{\code{method}}{ iterative method used to fit the model}
#' \item{\code{grads}}{ number of gradients used to approximate inverse information for L-BFGS}
#' \item{\code{y}}{ y vector used in the model, not included if \code{keepY = FALSE}}
#' \item{\code{x}}{ design matrix used to fit the model, not included if \code{keepData = FALSE}}
#' \item{\code{offset}}{ offset vector in the model, not included if \code{keepData = FALSE}}
#' \item{\code{data}}{ original dataframe supplied to the function, not included if \code{keepData = FALSE}}
#' \item{\code{numobs}}{ number of observations in the design matrix}
#' \item{\code{names}}{ names of the variables}
#' \item{\code{yname}}{ name of y variable}
#' \item{\code{parallel}}{ whether parallelization was employed to speed up model fitting process}
#' \item{\code{missing}}{ number of missing values removed from the original dataset}
#' \item{\code{link}}{ link function used to model the data}
#' \item{\code{family}}{ family used to model the data}
#' \item{\code{ylevel}}{ the levels of y, only included for binomial glms}
#' \item{\code{xlev}}{ the levels of the factors in the dataset}
#' \item{\code{terms}}{the terms object used}
#' 
#' \code{BranchGLM.fit} returns a list with the following components
#' \item{\code{coefficients}}{ a matrix with the coefficients estimates, SEs, wald test statistics, and p-values}
#' \item{\code{iterations}}{ number of iterations it took the algorithm to converge, if the algorithm failed to converge then this is -1}
#' \item{\code{dispersion}}{ the value of the dispersion parameter}
#' \item{\code{logLik}}{ the log-likelihood of the fitted model}
#' \item{\code{resdev}}{ the residual deviance of the fitted model}
#' \item{\code{AIC}}{ the AIC of the fitted model}
#' \item{\code{preds}}{ predictions from the fitted model}
#' \item{\code{linpreds}}{ linear predictors from the fitted model}
#' \item{\code{tol}}{ tolerance used to fit the model}
#' \item{\code{maxit}}{ maximum number of iterations used to fit the model}
#' 
#' @description Fits generalized linear models via RcppArmadillo. Also has the 
#' ability to fit the models with parallelization via OpenMP.
#' @details Can use BFGS, L-BFGS, or Fisher's scoring to fit the GLM. BFGS and L-BFGS are 
#' typically faster than Fisher's scoring when there are at least 50 covariates 
#' and Fisher's scoring is typically best when there are fewer than 50 covariates.
#' This function does not currently support the use of weights. In the special 
#' case of gaussian regression with identity link the \code{method} argument is ignored
#' and the normal equations are solved directly.
#' 
#' The models are fit in C++ by using Rcpp and RcppArmadillo. In order to help 
#' convergence, each of the methods makes use of a backtracking line-search using 
#' the strong Wolfe conditions to find an adequate step size. There are also 
#' two conditions used to control convergence, the first is whether there is a 
#' sufficient decrease in the negative log-likelihood, and the other is whether 
#' the norm of the score is sufficiently small. The 
#' \code{tol} argument controls both of these criteria. If the algorithm fails to 
#' converge, then \code{iterations} will be -1.
#' 
#' All observations with any missing values are removed before model fitting. 
#' 
#' The dispersion parameter for gamma regression is estimated via maximum likelihood, 
#' very similar to the \code{gamma.dispersion} function from the MASS package.
#' 
#' \code{BranchGLM.fit} can be faster than calling \code{BranchGLM} if the 
#' x matrix and y vector are already available, but doesn't return as much information.
#' The object returned by \code{BranchGLM.fit} is not of class \code{BranchGLM}, so 
#' all of the methods for \code{BranchGLM} objects such as \code{predict} or 
#' \code{VariableSelection} cannot be used.
#' 
#' @examples
#' Data <- iris
#' ### Using BranchGLM
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ### Using BranchGLM.fit
#' x <- model.matrix(Sepal.Length ~ ., data = Data)
#' y <- Data$Sepal.Length
#' BranchGLM.fit(x, y, family = "gaussian", link = "identity")
#' @export

BranchGLM <- function(formula, data, family, link, offset = NULL, 
                    method = "Fisher", grads = 10,
                    parallel = FALSE, nthreads = 8, 
                    tol = 1e-6, maxit = NULL, init = NULL, 
                    contrasts = NULL, keepData = TRUE,
                    keepY = TRUE){
  
  ### Validating supplied arguments
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
  m <- match(c("formula", "data", "offset"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf$na.action <- "na.omit"
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  
  ## Getting data objects
  y <- model.response(mf, "any")
  offset <- as.vector(model.offset(mf))
  x <- model.matrix(attr(mf, "terms"), mf, contrasts)
  
  if(is.null(offset)){
    offset <- rep(0, length(y))
  }
  
  
  ## Checking y variable for binomial family
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
  }
  
  ### Using BranchGLM.fit to fit GLM
  df <- BranchGLM.fit(x, y, family, link, offset, method, grads, parallel, nthreads, 
                      init, maxit, tol)
  
  row.names(df$coefficients) <- colnames(x)
  
  df$formula <- formula
  
  df$method <- method
  
  if(keepY){
    df$y <- y
  }
  
  df$numobs <- nrow(x)
  
  if(keepData){
    df$data <- data
    df$x <- x
    df$offset <- offset
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
  
  if(family == "binomial"){
    df$ylevel <- ylevel
  }
  if(family == "gaussian" || family == "gamma"){
    colnames(df$coefficients)[3] <- "t"
  }
  
  # Setting names for vcov
  rownames(df$vcov) <- colnames(df$vcov) <- colnames(x)
  
  structure(df, class = "BranchGLM")
}

#' @rdname BranchGLM
#' @export
BranchGLM.fit <- function(x, y, family, link, offset = NULL,
                          method = "Fisher", grads = 10,
                          parallel = FALSE, nthreads = 8, init = NULL,  
                          maxit = NULL, tol = 1e-6){
  
  ## Performing a few checks
  if(!is.matrix(x) || !is.numeric(x)){
    stop("x must be a numeric matrix")
  }else if(!is.numeric(y)){
    stop("y must be numeric")
  }else if(nrow(x) != length(y)){
    stop("the number of rows in x must be the same as the length of y")
  }
  
  ## Getting initial values
  if(is.null(init)){
    init <- rep(0, ncol(x))
    GetInit <- TRUE
  }else if(!is.numeric(init) || length(init) != ncol(x)){
    stop("init must be null or a numeric vector with length equal to the number of betas")
  }else{
    GetInit <- FALSE
  }
  
  ## Getting maxit
  if(is.null(maxit)){
    if(method == "Fisher"){
      maxit = 50
    }else{
      maxit = 200
    }
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
  }
  
  ## Getting offset
  if(is.null(offset)){
    offset <- rep(0, length(y))
  }else if(length(offset) != length(y)){
    stop("offset must be the same length as y")
  }
  
  if(length(parallel) != 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }else if(parallel){
    df <- BranchGLMfit(x, y, offset, init, method, grads, link, family, nthreads, 
                       tol, maxit, GetInit) 
  }else{
    df <- BranchGLMfit(x, y, offset, init, method, grads, link, family, 1, tol, maxit, 
                       GetInit) 
  }
  
  df$tol <- tol
  df$maxit <- maxit
  
  return(df)
}


#' Extract Log-Likelihood
#' @param object a \code{BranchGLM} object.
#' @param ... further arguments passed to or from other methods.
#' @return An object of class \code{logLik} which is a number corresponding to 
#' the log-likelihood with the following attributes: "df" (degrees of freedom) 
#' and "nobs" (number of observations).
#' @export

logLik.BranchGLM <- function(object, ...){
  df <- length(coef(object))
  if(object$family == "gaussian" || object$family == "gamma"){
    df <- df + 1
  }
  val <- object$logLik
  attr(val, "nobs") <- length(object$y) 
  attr(val, "df") <- df
  class(val) <- "logLik"
  return(val)
}

#' Extract covariance matrix
#' @param object a \code{BranchGLM} object.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric matrix which is the covariance matrix of the beta coefficients.
#' @export
vcov.BranchGLM <- function(object, ...){
  return(object$vcov)
}

#' Extract Coefficients
#' @param object a \code{BranchGLM} object.
#' @param ... further arguments passed to or from other methods.
#' @return A named vector with the corresponding coefficient estimates.
#' @export

coef.BranchGLM <- function(object, ...){
  coefs <- object$coefficients[,1]
  names(coefs) <- row.names(object$coefficients)
  return(coefs)
}

#' Predict Method for BranchGLM Objects
#' @param object a \code{BranchGLM} object.
#' @param newdata a dataframe, if not specified then the data the model was fit on is used.
#' @param type one of "linpreds" or "response", if not specified then "response" is used.
#' @param ... further arguments passed to or from other methods.
#' @details linpreds corresponds to the linear predictors and response is on the scale of the response variable.
#' Offset variables are ignored for predictions on new data.
#' @description Gets predictions from a \code{BranchGLM} object.
#' @return A numeric vector of predictions.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' predict(Fit)
#' ### Example with new data
#' predict(Fit, newdata = iris[1:20,])
#' @export

predict.BranchGLM <- function(object, newdata = NULL, type = "response", ...){
  
  if(!is.null(newdata) && !is(newdata,"data.frame")){
    stop("newdata argument must be a dataframe or NULL")
  }
  
  if(length(type) != 1 ){
    stop("type must have a length of 1")
  }else if(!(type %in% c("linpreds", "response"))){
    stop("type argument must be either 'linpreds' or 'response'")
  }
  
  if(is.null(newdata)){
    if(type == "linpreds"){
      object$linPreds
    }else if(type == "response"){
      object$preds
    }
  }else{
    myterms <- delete.response(terms(object))
    m <- model.frame(myterms, newdata, na.action = "na.omit",
                     xlev = object$xlev)
    x <- model.matrix(myterms, m, contrasts = object$contrasts)
    
    if(ncol(x) != length(object$coefficients$Estimate)){
      stop("could not find all predictor variables in newdata")
    }else if(type == "linpreds"){
      drop(x %*% coef(object)) |> unname()
    }else if(type == "response"){
      GetPreds(drop(x %*% coef(object)) |> unname(), object$link)
    }
  }
}

GetPreds <- function(XBeta, Link){
  if(Link == "log"){
    exp(XBeta)
  }
  else if(Link == "logit"){
    1 / (1 + exp(-XBeta))
  }
  else if(Link == "probit"){
    pnorm(XBeta)
  }
  else if(Link == "cloglog"){
    exp(-exp(XBeta))
  }
  else if(Link == "inverse"){
    - 1 / (XBeta)
  }
  else if(Link == "identity"){
    XBeta
  }
  else{
    XBeta^2
  }
}

#' Print Method for BranchGLM
#' @param x a \code{BranchGLM} object.
#' @param coefdigits number of digits to display for coefficients table.
#' @param digits number of digits to display for information after table.
#' @param ... further arguments passed to or from other methods.
#' @return The supplied \code{BranchGLM} object.
#' @export

print.BranchGLM <- function(x, coefdigits = 4, digits = 2, ...){
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
  
  cat(paste0("\nDispersion parameter taken to be ", round(x$dispersion, coefdigits)))
  cat(paste0("\n", x$numobs, " observations used to fit model\n(", x$missing, 
             " observations removed due to missingness)\n"))
  cat(paste0("\nResidual Deviance: ", round(x$resDev, digits = digits), " on ",
             x$numobs - nrow(x$coefficients), " degrees of freedom"))
  cat(paste0("\nAIC: ", round(x$AIC, digits = digits)))
  if(x$family != "gaussian" || x$link != "identity"){
    if(x$method == "Fisher"){
      method = "Fisher's scoring"
    }else if(x$method == "LBFGS"){
      method = "L-BFGS"
    }else{method = "BFGS"}
    if(x$iterations == 1){
      cat(paste0("\nAlgorithm converged in 1 iteration using ", method, "\n"))
    }else if(x$iterations > 1){
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
