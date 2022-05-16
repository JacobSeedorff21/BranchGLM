#' Fits GLMs
#' @param formula a formula for the model.
#' @param data a dataframe that contains the response and predictor variables.
#' @param family distribution used to model the data, one of "gaussian", "binomial", or "poisson".
#' @param link link used to link mean structure to linear predictors. One of, 
#' "identity", "logit", "probit", "cloglog", or "log".
#' @param offset offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS".
#' @param grads number of gradients to use to approximate information with, only for LBFGS.
#' @param parallel whether or not to make use of parallelization via OpenMP.
#' @param nthreads number of threads used with OpenMP, only used if parallel = TRUE.
#' @param tol tolerance used to determine model convergence.
#' @param maxit maximum number of iterations performed. The default for 
#' Fisher scoring is 50 and for the other methods the default is 200.
#' @param contrasts see \code{contrasts.arg} of \code{model.matrix.default}.
#' @return A \code{BranchGLM} object which is a list with the following components
#' \item{\code{coefficients}}{ a matrix with the coefficients estimates, SEs, wald test statistics, and p-values}
#' \item{\code{iterations}}{ number of iterations it took the algorithm to converge, if the algorithm failed to converge then this is -1}
#' \item{\code{dispersion}}{ the value of the dispersion parameter}
#' \item{\code{logLik}}{ the log-likelihood of the fitted model}
#' \item{\code{resdev}}{ the residual deviance of the fitted model}
#' \item{\code{AIC}}{ the AIC of the fitted model}
#' \item{\code{preds}}{ predictions from the fitted model}
#' \item{\code{linpreds}}{ linear predictors from the fitted model}
#' \item{\code{formula}}{ formula used to fit the model}
#' \item{\code{method}}{ iterative method used to fit the model}
#' \item{\code{y}}{ y vector used in the model}
#' \item{\code{x}}{ design matrix used to fit the model}
#' \item{\code{data}}{ original dataframe supplied to the function}
#' \item{\code{names}}{ names of the variables}
#' \item{\code{yname}}{ name of y variable}
#' \item{\code{parallel}}{ whether parallelization was employed to speed up model fitting process}
#' \item{\code{missing}}{ number of missing values removed from the original dataset}
#' \item{\code{link}}{ link function used to model the data}
#' \item{\code{offset}}{ offset vector}
#' \item{\code{family}}{ family used to model the data}
#' \item{\code{ylevel}}{ the levels of y, only included for binomial glms}
#' @description Fits generalized linear models via RcppArmadillo. Also has the 
#' ability to fit the models with parallelization via openMP.
#' @details Can use BFGS, L-BFGS, or Fisher scoring to fit the GLM. BFGS and L-BFGS are 
#' typically faster than Fisher scoring when there are at least 50 covariates 
#' and Fisher scoring is typically best when there are fewer than 20 covariates.
#' This function does not currently support the use of weights. 
#' 
#' The models are fit in C++ by using Rcpp and RcppArmadillo. In order to help 
#' convergence, each of the methods makes use of a backtracking line-search using 
#' the armijo-goldstein condition to find an adequate step size. There are also 
#' two conditions used to control convergence, the first is whether there is a 
#' sufficient decrease in the negative log-likelihood, and the other is whether 
#' each of the elements of the beta vector changes by a sufficient amount. The 
#' \code{tol} argument controls both of these criteria. If the algorithm fails to 
#' converge, then \code{iterations} will be -1.
#' 
#' The likelihood equations are solved directly, i.e. no matrix decomposition is used.
#' @examples
#' Data <- iris
#' BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' @export

BranchGLM <- function(formula, data, family, link, offset = NULL, 
                    method = "Fisher", grads = 10, parallel = FALSE, nthreads = 8, 
                    tol = 1e-4, maxit = NULL, contrasts = NULL){
  
  if(!is(formula, "formula")){
    stop("formula must be a valid formula")
  }
  if(!is(data, "data.frame")){
    stop("data must be a data frame")
  }
  if(length(method) != 1 || !(method %in% c("Fisher", "BFGS", "LBFGS"))){
    warning("method must be exactly one of 'Fisher', 'BFGS', or 'LBFGS'")
  }
  if(!family %in% c("gaussian", "binomial", "poisson")){
    stop("family must be one of 'gaussian', 'binomial', or 'poisson'")
  }
  if(!link %in% c("logit", "probit", "cloglog", "log", "identity")){
    stop("link must be one of 'logit', 'probit', 'cloglog', 'log', or 'identity'")
  }
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  y <- model.response(mf, "any")
  
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
  
  ## Checking y variable for each family
  if(family == "binomial"){
    if(is.factor(y) && (nlevels(y) == 2)){
      ylevel <- levels(y)
      y <- as.numeric(y == ylevel[2])
    }else if(is.numeric(y) && all(y %in% c(0, 1))){
      ylevel <- c(0, 1)
    }else if(is.logical(y)){
      ylevel <- c(FALSE, TRUE)
      y <- y * 1
    }else{
      stop("response variable for binomial regression must be a numeric vector with only 
      0s and 1s, a two-level factor vector, or a logical vector")
      }
  }else if(family == "poisson"){
      if(!is.numeric(y) || any(y < 0)){
        stop("response variable for poisson regression must be a numeric vector of non-negative integers")
      }else if(any(as.integer(y)!= y)){
        stop("response variable for poisson regression must be a numeric vector of non-negative integers")
    }
  }else if(family == "gaussian"){
    if(!is.numeric(y)){
      stop("response variable for gaussian regression must be numeric")
    }
  }
  
  x <- model.matrix(formula, data, contrasts)
  
  ### Checks for offset
  if(is.null(offset)){
    offset <- rep(0, nrow(x))
  }else if(length(offset) != length(y)){
    stop("offset must have the same length as the y")
  }else if(!is.numeric(offset)){
    stop("offset must be a numeric vector")
  }
  ### Fitting GLM
  if(length(parallel) != 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }else if(parallel){
    df <- BranchGLMfit(x, y, offset, method, grads, link, family, nthreads, 
                       tol, maxit) 
  }else{
    df <- BranchGLMfit(x, y, offset, method, grads, link, family, 1, tol, maxit) 
  }
  
  row.names(df$coefficients) <- colnames(x)
  
  df$formula <- formula
  
  df$method <- method
  
  df$y <- y
  
  df$x <- x
  
  df$data <- data
  
  df$names <- attributes(terms(formula, data = data))$factors |>
              colnames()
  
  df$yname <- attributes(terms(formula, data = data))$variables[-1] |>
              as.character()
  
  df$yname <- df$yname[attributes(terms(formula, data = data))$response]
  
  df$parallel <- parallel
  
  df$missing <- nrow(data) - nrow(x)
  
  df$link <- link
  
  df$contrasts <- contrasts
  
  df$offset <- offset
  
  df$family <- family
  if(family == "binomial"){
    df$ylevel <- ylevel
  }
  structure(df, class = "BranchGLM")
}

#' Extract Log-Likelihood
#' @param object a \code{BranchGLM} model object.
#' @param ... further arguments passed to or from other methods.
#' @export

logLik.BranchGLM<- function(object, ...){
  object$logLik
}

#' Extract AIC
#' @param object a \code{BranchGLM} model object.
#' @param ... further arguments passed to or from other methods.
#' @export

AIC.BranchGLM <- function(object, ...){
  object$AIC
}

#' Extract BIC
#' @param object a \code{BranchGLM} model object.
#' @param ... further arguments passed to or from other methods.
#' @export

BIC.BranchGLM <- function(object, ...){
  k <- length(coef(object))
  if(object$family == "gaussian"){
    k <- k + 1
  }
  -2 * logLik(object) + log(length(object$y)) * k
}

#' Extract Coefficients
#' @param object a \code{BranchGLM} model object.
#' @param ... further arguments passed to or from other methods.
#' @export

coef.BranchGLM <- function(object, ...){
  coefs <- object$coefficients[,1]
  names(coefs) <- row.names(object$coefficients)
  return(coefs)
}

#' Predict Method for BranchGLM Objects
#' @param object a \code{BranchGLM} object.
#' @param newdata a dataframe, if not specified the data the model was fit on is used.
#' @param type one of "linpreds" or "response", if not specified "response" is used.
#' @param ... further arguments passed to or from other methods.
#' @details linpreds corresponds to the linear predictors and response is on the scale of the response variable.
#' @description Gets predictions from a \code{BranchGLM} object.
#' @return A numeric vector of predictions.
#' @rdname predict.BranchGLM
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
    x <- model.matrix(object$formula, newdata, object$contrasts)
    if(ncol(x) != length(object$coefficients$Estimate)){
      stop("could not find all predictor variables in newdata")
    }else if(type == "linpreds"){
      x %*% coef(object)
    }else if(type == "response"){
      GetPreds(x %*% coef(object), object$link)
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
    1 / (XBeta)
  }
  else if(Link == "identity"){
    XBeta
  }
  else{
    sqrt(XBeta)
  }
}

#' Print Method for BranchGLM
#' @param x a \code{BranchGLM} model object.
#' @param coefdigits number of digits to display for coefficients table.
#' @param digits number of digits to display for information after table.
#' @param ... further arguments passed to or from other methods.
#' @export

print.BranchGLM <- function(x, coefdigits = 4, digits = 0, ...){
  
  cat(paste0("Results from ", x$family, " regression with ", x$link, 
             " link function \nUsing the formula ", deparse1(x$formula), "\n\n"))
  
  printCoefmat(x$coefficients, signif.stars = TRUE, P.values = TRUE, 
               has.Pvalue = TRUE)
  
  cat(paste0("\nDispersion parameter taken to be ", round(x$dispersion, coefdigits)))
  cat(paste0("\n", nrow(x$x), " observations used to fit model\n(", x$missing, 
             " observations removed due to missingness)\n"))
  cat(paste0("\nResidual Deviance: ", round(x$resDev, digits = digits), " on ",
             nrow(x$x) - nrow(x$coefficients), " degrees of freedom"))
  cat(paste0("\nAIC: ", round(x$AIC, digits = digits)))
  
  if(x$method == "Fisher"){
    method = "Fisher scoring"
  }else if(x$method == "LBFGS"){
    method = "L-BFGS"
  }else{method = "BFGS"}
  if(x$iterations == 1){
    cat(paste0("\nAlgorithm converged in 1 iteration using ", method, "\n"))
  }else if(x$iterations > 1){
    cat(paste0("\nAlgorithm converged in ", x$iterations, " iterations using ", method, "\n"))
  }else{
    cat("\nAlgorithm failed to converge")
  }
  
  if(x$parallel){
    cat("Parallel computation was used to speed up model fitting process")
  }
  invisible(x)
}

### TODO: Implement LRT CI using secant method 

### Notes
#---------------#
### Reordered branch and bound works so well because it finds a good solution 
### pretty quickly, finds the model selected by forward selection first.
### Also works well because the upper models always contain the worst variables, 
### so as the better predictors get removed the likelihood on the larger models 
### get worse very quickly and thus the lower bounds get larger quickly.
### My implementation of reordered branch and bound uses wide branching.
