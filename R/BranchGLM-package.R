#' @keywords internal
#' @name BranchGLM-package
#' @aliases BranchGLM-package NULL
#' @docType package
#' @examples
#' # Using iris data to demonstrate package usage
#' Data <- iris
#' 
#' # Fitting linear regression model
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' Fit
#' 
#' # Doing branch and bound best subset selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' showprogress = FALSE, bestmodels = 10)
#' VS
#' 
#' ## Plotting results
#' plot(VS, ptype = "variables")
#' 
"_PACKAGE"

## usethis namespace: start
#' @useDynLib BranchGLM, .registration = TRUE
#' @import stats
#' @import graphics
#' @importFrom methods is
#' @importFrom Rcpp evalCpp
## usethis namespace: end

NULL

#' Internal BranchGLM Functions
#' @description Internal BranchGLM Functions.
#' @details These are not intended for use by users, these are Rcpp functions 
#' that do not check the arguments, so improper usage may result in R crashing.
#' 
#' @aliases BranchGLMFit MetricIntervalCpp SwitchBranchAndBoundCpp BranchAndBoundCpp 
#' BackwardBranchAndBoundCpp ForwardCpp BackwardCpp MakeTable MakeTableFactor2 
#' CindexCpp CindexTrap ROCCpp
#' @keywords internal
