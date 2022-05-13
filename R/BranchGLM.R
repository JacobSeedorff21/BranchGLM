#' Fits GLMs
#' @param formula a formula for the model.
#' @param data a dataframe that contains the response and predictor variables.
#' @param family distribution used to model the data, one of "gaussian", "binomal", or "poisson"
#' @param link link used to link mean structure to linear predictors. One of, 
#' "identity", "logit", "probit", "cloglog", or "log".
#' @param offset offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS".
#' @param grads number of gradients to use to approximate information with, only for LBFGS.
#' @param parallel whether or not to make use of parallelization via OpenMP.
#' @param nthreads number of threads used with OpenMP, only used if parallel = TRUE.
#' @param tol tolerance used to determine model convergence.
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
                    tol = 1e-4, contrasts = NULL){
  
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
  
  ## Checking y variable for binomial regression
  
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
      stop("Response variable for binomial regression must be numeric with only 
      0s and 1s, a two-level factor, or a logical vector")
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
                       tol) 
  }else{
    df <- BranchGLMfit(x, y, offset, method, grads, link, family, 1, tol) 
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
#' @name logLik
#' @param fit A BranchGLM or BranchGLMboot object.
#' @description Gets log-likelihood from fitted model.
#' @return Returns log-likelihood
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' logLik(Fit)
#' @export

logLik.BranchGLM<- function(fit){
  Fit$logLik
}

#' Extract AIC
#' @name AIC
#' @param fit A BranchGLM or BranchGLMboot object.
#' @description Gets AIC from fitted model.
#' @return Returns AIC
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' AIC(Fit)
#' BIC(Fit)
#' @export

AIC.BranchGLM <- function(fit){
  Fit$AIC
}

#' @rdname AIC
#' @export

BIC.BranchGLM <- function(fit){
  k <- length(coef(fit))
  if(fit$family == "gaussian"){
    k <- k + 1
  }
  -2 * logLik(fit) + log(length(fit$y)) * k
}

#' Extract Coefficient Estimates
#' @name coef
#' @param fit A BranchGLM or BranchGLMboot object.
#' @description Extract coefficient estimates from BranchGLM or BranchGLMboot object.
#' @return  Returns a named vector of the coefficient estimates.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' coef(Fit)
#' @export

coef.BranchGLM <- function(fit){
  coefs <- fit$coefficients[,1]
  names(coefs) <- row.names(fit$coefficients)
  return(coefs)
}

#' Predict Method for BranchGLM Object
#' @param fit A BranchGLM or BranchGLMboot object.
#' @param newdata A dataframe, if not specified the data the model was fit on is used.
#' @param type One of "linpreds" or "response", if not specified "response" is used.
#' @param contrasts see contrasts.arg of model.matrix.default.
#' @details Link corresponds to the linear predictors and response corresponds to the predicted probabilities.
#' @description Gets predictions from BranchGLM model.
#' @return A numeric vector of predictions.
#' @name predict
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' predict(Fit)
#' ### Example with new data
#' predict(Fit, newdata = iris[1:20,])
#' @export

predict.BranchGLM <- function(fit, newdata = NULL, type = "response"){
  
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
      fit$linPreds
    }else if(type == "response"){
      fit$preds
    }
  }else{
    x <- model.matrix(fit$formula, newdata, fit$contrasts)
    if(ncol(x) != length(fit$coefficients$Estimate)){
      stop("could not find all predictor variables in newdata")
    }else if(type == "linpreds"){
      x %*% coef(fit)
    }else if(type == "response"){
      GetPreds(x %*% coef(fit), fit$link)
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


#' @title Print Method for BranchGLM
#' @param fit A BranchGLM model object.
#' @param coefdigits Number of digits to display for coefficients table.
#' @param digits Number of digits to display for information after table.
#' @export

print.BranchGLM <- function(fit, coefdigits = 4, digits = 0){
  
  cat(paste0("Results from ", fit$family, " regression with ", fit$link, 
             " link function \nUsing the formula ", deparse1(fit$formula), "\n\n"))
  
  printCoefmat(fit$coefficients, signif.stars = TRUE, P.values = TRUE, 
               has.Pvalue = TRUE)
  
  cat(paste0("\nDispersion parameter taken to be ", round(fit$dispersion, coefdigits)))
  cat(paste0("\n", nrow(fit$x), " observations used to fit model\n(", fit$missing, 
             " observations removed due to missingness)\n"))
  cat(paste0("\nResidual Deviance: ", round(fit$resDev, digits = digits), " on ",
             nrow(fit$x) - nrow(fit$coefficients), " degrees of freedom"))
  cat(paste0("\nAIC: ", round(fit$AIC, digits = digits)))
  
  if(fit$method == "Fisher"){
    method = "Fisher scoring"
  }else if(fit$method == "LBFGS"){
      method = "L-BFGS"
      }else{method = "BFGS"}
  if(fit$iterations == 1){
    cat(paste0("\nAlgorithm converged in 1 iteration using ", method, "\n"))
  }else if(fit$iterations > 1){
    cat(paste0("\nAlgorithm converged in ", fit$iterations, " iterations using ", method, "\n"))
  }else{
    cat("\nAlgorithm failed to converge")
  }
  
  if(fit$parallel){
    cat("Parallel computation was used to speed up model fitting process")
  }
  invisible(fit)
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
