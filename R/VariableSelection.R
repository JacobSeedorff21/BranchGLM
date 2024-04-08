#' Variable Selection for GLMs 
#' @description Performs forward selection, backward elimination, and efficient 
#' best subset variable selection with information criterion for generalized linear 
#' models (GLMs). Best subset selection is performed with branch and bound algorithms 
#' to greatly speed up the process.
#' @param object a formula or a `BranchGLM` object.
#' @param ... further arguments.
#' @param data a data.frame, list or environment (or object coercible by 
#' [as.data.frame] to a data.frame), containing the variables in formula. 
#' Neither a matrix nor an array will be accepted.
#' @param family the distribution used to model the data, one of "gaussian", "gamma", 
#' "binomial", or "poisson".
#' @param link the link used to link the mean structure to the linear predictors. One of
#' "identity", "logit", "probit", "cloglog", "sqrt", "inverse", or "log".
#' @param offset the offset vector, by default the zero vector is used.
#' @param method one of "Fisher", "BFGS", or "LBFGS". Fisher's scoring is recommended
#' for forward selection and the branch and bound algorithms since they will typically 
#' fit many models with a small number of covariates.
#' @param type one of "forward", "backward", "branch and bound", "backward branch and bound", or "switch branch and bound" 
#' to indicate the type of variable selection to perform. The default value is 
#' "switch branch and bound". See more about these algorithms in details
#' @param metric the metric used to choose the best models, the default is "AIC", 
#' but "BIC" and "HQIC" are also available. AIC is the Akaike information criterion, 
#' BIC is the Bayesian information criterion, and HQIC is the Hannan-Quinn information 
#' criterion. 
#' @param bestmodels a positive integer to indicate the number of the best models to 
#' find according to the chosen metric or NULL. If this is NULL, then cutoff is 
#' used instead. This is only used for the branch and bound algorithms.
#' @param cutoff a non-negative number which indicates that the function 
#' should return all models that have a metric value within cutoff of the 
#' best metric value or NULL. Only one of this or bestmodels should be specified and 
#' when both are NULL a cutoff of 0 is used. This is only used for the branch 
#' and bound algorithms.
#' @param keep a character vector of names to denote variables that must be in the models.
#' @param keepintercept a logical value to indicate whether to keep the intercept in 
#' all models, only used if an intercept is included in the formula.
#' @param maxsize a positive integer to denote the maximum number of variables to 
#' consider in a single model, the default is the total number of variables. 
#' This number adds onto any variables specified in keep. This argument only works 
#' for `type = "forward"` and `type = "branch and bound"`. This argument is now 
#' deprecated.
#' @param grads a positive integer to denote the number of gradients used to 
#' approximate the inverse information with, only for `method = "LBFGS"`..
#' @param parallel a logical value to indicate if parallelization should be used.
#' @param nthreads a positive integer to denote the number of threads used with OpenMP, 
#' only used if `parallel = TRUE`.
#' @param tol a positive number to denote the tolerance used to determine model convergence.
#' @param maxit a positive integer to denote the maximum number of iterations performed. 
#' The default for Fisher's scoring is 50 and for the other methods the default is 200.
#' @param showprogress a logical value to indicate whether to show progress updates 
#' for branch and bound algorithms.
#' @param contrasts see `contrasts.arg` of `model.matrix.default`.
#' @seealso [plot.BranchGLMVS], [coef.BranchGLMVS], [predict.BranchGLMVS], 
#' [summary.BranchGLMVS]
#' @details 
#' 
#' ## Variable Selection Details
#' The supplied formula or the formula from the fitted model is 
#' treated as the upper model. The variables specified in keep along with an 
#' intercept (if included in formula and keepintercept = TRUE) is the lower model. 
#' Factor variables are either kept in their entirety or entirely removed and 
#' interaction terms are properly handled. All observations that have any missing 
#' values in the upper model are removed.
#' 
#' ## Branch and Bound Algorithms
#' The branch and bound algorithm is an efficient algorithm used to find the optimal 
#' models. The backward branch and bound algorithm is very similar to 
#' the branch and bound algorithm, except it tends to be faster when the best models 
#' contain most of the variables. The switch branch and bound algorithm is a 
#' combination of the two algorithms and is typically the fastest of the 3 branch and 
#' bound algorithms. All of the branch and bound algorithms are guaranteed to find 
#' the optimal models (up to numerical precision).
#' 
#' ## GLM Fitting
#' 
#' Fisher's scoring is recommended for branch and bound selection and forward selection.
#' L-BFGS may be faster for backward elimination especially when there are many variables.
#' 
#' @return A `BranchGLMVS` object which is a list with the following components
#' \item{`initmodel`}{ the `BranchGLM` object corresponding to the upper model}
#' \item{`numchecked`}{ number of models fit}
#' \item{`names`}{ character vector of the names of the predictor variables}
#' \item{`order`}{ the order the variables were added to the model or removed from the model, this is only included for the stepwise algorithms}
#' \item{`type`}{ type of variable selection employed}
#' \item{`optType`}{ whether the type specified used a heuristic or exact algorithm}
#' \item{`metric`}{ metric used to select best models}
#' \item{`bestmodels`}{ numeric matrix used to describe the best models for the branch and bound algorithms 
#' or a numeric matrix describing the models along the path taken for stepwise algorithms}
#' \item{`bestmetrics`}{ numeric vector with the best metrics found in the search for the branch and bound algorithms 
#' or a numeric vector with the metric values along the path taken for stepwise algorithms}
#' \item{`beta`}{ numeric matrix of beta coefficients for the models in bestmodels}
#' \item{`cutoff`}{ the cutoff that was used, this is set to -1 if bestmodels was used instead or if 
#' a stepwise algorithm was used}
#' \item{`keep`}{ vector of which variables were kept through the selection process}
#' @name VariableSelection
#' 
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", 
#' link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' VS
#' 
#' ## Plotting the BIC of the best models
#' plot(VS, type = "b")
#' 
#' ## Getting the coefficients of the best model according to BIC
#' FinalModel <- coef(VS, which = 1)
#' FinalModel
#' 
#' # Now doing it in parallel (although it isn't necessary for this dataset)
#' parVS <- VariableSelection(Fit, type = "branch and bound", parallel = TRUE, 
#' metric = "BIC", bestmodels = 10, showprogress = FALSE)
#' 
#' ## Getting the coefficients of the best model according to BIC
#' FinalModel <- coef(parVS, which = 1)
#' FinalModel
#' 
#' # Using a formula
#' formVS <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
#' link = "identity", metric = "BIC", type = "branch and bound", bestmodels = 10, 
#' showprogress = FALSE)
#' 
#' ## Getting the coefficients of the best model according to BIC
#' FinalModel <- coef(formVS, which = 1)
#' FinalModel
#' 
#' # Using the keep argument
#' keepVS <- VariableSelection(Fit, type = "branch and bound", 
#' keep = c("Species", "Petal.Width"), metric = "BIC", bestmodels = 4, 
#' showprogress = FALSE)
#' keepVS
#' 
#' ## Getting the coefficients from the fourth best model according to BIC when 
#' ## keeping Petal.Width and Species in every model
#' FinalModel <- coef(keepVS, which = 4)
#' FinalModel
#' 
#' # Treating categorical variable beta parameters separately
#' ## This function automatically groups together parameters from a categorical variable
#' ## to avoid this, you need to create the indicator variables yourself
#' x <- model.matrix(Sepal.Length ~ ., data = iris)
#' Sepal.Length <- iris$Sepal.Length
#' Data <- cbind.data.frame(Sepal.Length, x[, -1])
#' VSCat <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
#' link = "identity", metric = "BIC", bestmodels = 10, showprogress = FALSE)
#' VSCat
#' 
#' ## Plotting results
#' plot(VSCat, cex.names = 0.75)
#' 
#' @export
#' 
VariableSelection <- function(object, ...) {
  UseMethod("VariableSelection")
}

#'@rdname VariableSelection
#'@export

VariableSelection.formula <- function(object, data, family, link, offset = NULL,
                                      method = "Fisher", type = "switch branch and bound", 
                                      metric = "AIC",
                                      bestmodels = NULL, cutoff = NULL, 
                                      keep = NULL, keepintercept = TRUE, maxsize = NULL,
                                      grads = 10, parallel = FALSE, 
                                      nthreads = 8, tol = 1e-6, maxit = NULL,
                                      contrasts = NULL,
                                      showprogress = TRUE, ...){
  ### Performing variable selection
  ### model.frame searches for offset in the environment the formula is in, so 
  ### we need to change the environment of the formula to be the current environment
  formula <- object
  environment(formula) <- environment()
  fit <- BranchGLM(formula, data = data, family = family, link = link, 
                   offset = offset, method = method, grads = grads, 
                   tol = tol, maxit = maxit, contrasts = contrasts, 
                   fit = FALSE)
  
  VariableSelection(fit, type = type, metric = metric, 
                    bestmodels = bestmodels, cutoff = cutoff,
                    keep = keep, keepintercept = keepintercept, 
                    maxsize = maxsize, parallel = parallel, 
                    nthreads = nthreads,
                    showprogress = showprogress, ...)
}

#'@rdname VariableSelection
#'@export

VariableSelection.BranchGLM <- function(object, type = "switch branch and bound", 
                                        metric = "AIC",
                                        bestmodels = NULL, cutoff = NULL, 
                                        keep = NULL, keepintercept = TRUE, maxsize = NULL,
                                        parallel = FALSE, nthreads = 8,
                                        showprogress = TRUE, ...){
  ## converting metric to upper and type to lower
  type <- tolower(type)
  metric <- toupper(metric)
  
  ## Checking if supplied BranchGLM object has x and data
  if(is.null(object$x)){
    stop("the supplied model must have an x component")
  }else if(nrow(object$x) == 0){
    stop("the design matrix in object has 0 rows")
  }
  ## Checking if supplied BranchGLM object has y
  if(is.null(object$y)){
    stop("the supplied model must have a y component")
  }else if(length(object$y) == 0){
    stop("the y component in object has 0 rows")
  }
    
  ## Validating supplied arguments
  if(length(nthreads) != 1 || !is.numeric(nthreads) || is.na(nthreads) || nthreads <= 0){
    stop("nthreads must be a positive integer")
  }
  if(length(parallel) != 1 || !is.logical(parallel) || is.na(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  
  ### Checking showprogress
  if(length(showprogress) != 1 || !is.logical(showprogress)){
    stop("showprogress must be a logical value")
  }
  
  ### Checking metric
  if(length(metric) > 1 || !is.character(metric)){
    stop("metric must be one of 'AIC','BIC', or 'HQIC'")
  }else if(!(metric %in% c("AIC", "BIC", "HQIC"))){
    stop("metric must be one of 'AIC','BIC', or 'HQIC'")
  }
  
  ### Checking type
  if(length(type) != 1 || !is.character(type)){
    stop("type not supported, please see documentation for valid types")
  }
  
  ### Checking bestmodels
  if(is.null(bestmodels)){
    
  }else if(length(bestmodels) != 1 || !is.numeric(bestmodels) || 
     bestmodels <= 0 || bestmodels != as.integer(bestmodels)){
    stop("bestmodels must be a positive integer")
  }else if(!is.null(cutoff) && !is.null(bestmodels)){
    stop("only one of bestmodels or cutoff can be specified")
  }
  
  ### Checking cutoff
  if(is.null(cutoff)){
    
  }else if(length(cutoff) != 1 || !is.numeric(cutoff) || cutoff < 0){
    stop("cutoff must be a non-negative number")
  }
  
  if(is.null(cutoff)){
    if(is.null(bestmodels)){
      cutoff <- 0
      bestmodels <- 1
    }else{
      cutoff <- -1
    }
  }else if(is.null(bestmodels)){
    bestmodels <- 1
  }
  
  indices <- attr(object$x, "assign")  
  counts <- table(indices)
  interactions <- attr(object$terms, "factors")[-1L, ]
  
  ## Removing rows with all zeros
  if(is.matrix(interactions)){
    interactions <- interactions[apply(interactions, 1, function(x){sum(x) > 0}),]
  }else{
    ### This only happens when only 1 variable is included
    interactions <- matrix(1, nrow = 1, ncol = 1)
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

  if(is.null(maxsize)){
    maxsize <- length(counts)
  }else if(length(maxsize) != 1 || !is.numeric(maxsize) || maxsize <= 0){
    warning("maxsize argument is deprecated")
    stop("maxsize must be a positive integer specifying the max size of the models") 
  }else{
    warning("maxsize argument is deprecated")
  }
  
  ## Setting starting model and saving keep1 for later use since keep is modified
  ### Checking keep
  CurNames <- colnames(attributes(terms(object$formula, data = object$data))$factors)
  if(!is.character(keep) && !is.null(keep)){
    stop("keep must be a character vector or NULL")
  }else if(!is.null(keep) && !all(keep %in% CurNames)){
    keep <- keep[!(keep %in% CurNames)]
    stop(paste0("the following elements were found in keep, but are not variable names: ",
                paste0(keep, collapse = ", ")))
  }
  ### Checking keepintercept
  if(length(keepintercept) != 1 || !is.logical(keepintercept)){
    stop("keepintercept must be a logical value")
  }
  
  keep1 <- keep
  if(!intercept){
    # Changing keepintercept to FALSE since there is no intercept
    keepintercept <- FALSE
  }
  if(is.null(keep) && type != "backward"){
    keep <- rep(0, length(counts))
    if(intercept && keepintercept){
      keep[1] <- -1
    }
  }else if(is.null(keep) && type == "backward"){
    keep <- rep(1, length(counts))
    if(intercept && keepintercept){
      keep[1] <- -1
    }
  }else{
    keep <- (CurNames %in% keep) * -1
    if(type == "backward"){
      keep[keep == 0] <- 1
    }
    if(intercept && keepintercept){
      keep <- c(-1, keep) 
    }else if(intercept){
      keep <- c(0, keep)
    }
  }
  
  ## Checking for parallel
  if(!parallel){
    nthreads <- 1
  }
  
  ## Getting penalties
  if(metric == "AIC"){
    pen <- as.vector(counts) * 2
    penalty <- 2
  }else if(metric == "BIC"){
    pen <- as.vector(counts) * log(nrow(object$x))
    penalty <- log(nrow(object$x))
  }else if(metric == "HQIC"){
    pen <- as.vector(counts) * 2 * log(log(nrow(object$x)))
    penalty <- 2 * log(log(nrow(object$x)))
  }
  
  ## Performing variable selection
  if(type == "forward"){
    if(bestmodels > 1 || cutoff > 0){
      warning("forward selection only finds 1 final model")
    }
    df <- ForwardCpp(object$x, object$y, object$offset, indices, counts, 
                     interactions, object$method, object$grads, object$link, 
                     object$family, nthreads, object$tol, object$maxit, keep, 
                     maxsize, pen)
    optType <- "heuristic"
  }else if(type == "backward"){
    if(bestmodels > 1 || cutoff > 0){
      warning("backward elimination only finds 1 final model")
    }
    df <- BackwardCpp(object$x, object$y, object$offset, indices, counts, 
                      interactions, object$method, object$grads,
                      object$link, object$family, nthreads, object$tol, object$maxit, 
                      keep, length(counts), pen)
    optType <- "heuristic"
  }else if(type == "branch and bound"){
    df <- BranchAndBoundCpp(object$x, object$y, object$offset, indices, counts, 
                            interactions, object$method, object$grads,
                            object$link, object$family, nthreads,
                            object$tol, object$maxit, keep, maxsize,
                            pen, showprogress, bestmodels, cutoff)
    optType <- "exact"
  }else if(type == "backward branch and bound"){
    df <- BackwardBranchAndBoundCpp(object$x, object$y, object$offset, indices, 
                                    counts, interactions, object$method, object$grads,
                                    object$link, object$family, nthreads, object$tol, 
                                    object$maxit, keep, 
                                    pen, showprogress, bestmodels, cutoff)
    optType <- "exact"
  }else if(type == "switch branch and bound"){
    df <- SwitchBranchAndBoundCpp(object$x, object$y, object$offset, indices, counts, 
                                  interactions, object$method, object$grads,
                                  object$link, object$family, nthreads, 
                                  object$tol, object$maxit, keep, 
                                  pen, showprogress, bestmodels, cutoff)
    optType <- "exact"
  }else{
    stop("type not supported, please see documentation for valid types")
  }
  
  # Creating coefficient names
  names <- object$names
  if(intercept){
    names <- c("(Intercept)", names)
  }
  
  if(optType == "heuristic"){
    # Checking for infinite best metric value
    if(all(is.infinite(df$bestmetrics))){
      stop("no models were found that had an invertible fisher information")
    }
    
    # Adding penalty to gaussian and gamma families
    if(object$family %in% c("gaussian", "gamma")){
      df$bestmetrics <- df$bestmetrics + penalty
    }
    if(!intercept){
      df$order <- df$order + 1
    }
    if(is.matrix(df$order)){
      df$order <- apply(df$order, 1, function(x){
        if(x[1] > 0){
          mystr <- object$names[x[1]]
          if(x[2] > 0){
            mystr <- paste0(mystr, " and ", object$names[x[2]])
          }
          return(mystr)
        }else{
          return(NA_character_)
        }
      })
      df$order <- df$order[!is.na(df$order)]
    }else{
      df$order <- df$order[df$order > 0]
      df$order <- object$names[df$order]
    }
    
    # Making betas and bestmodels
    modelOrder <- rev(which(!is.infinite(df$bestmetrics)))
    bestmetrics <- df$bestmetrics[modelOrder]
    betas <- df$betas[, modelOrder, drop = FALSE]
    bestmodels <- df$bestmodel[, modelOrder, drop = FALSE]
    rownames(bestmodels) <- names
    rownames(betas) <- colnames(object$x)
    
    FinalList <- list("numchecked" = df$numchecked,
                      "order" = df$order,
                      "type" = type, 
                      "metric" = metric,
                      "bestmodels" = bestmodels,
                      "bestmetrics" = bestmetrics,
                      "beta" = betas,
                      "names" = names, 
                      "initmodel" = object,
                      "cutoff" = -1,
                      "keep" = keep1, 
                      "keepintercept" = keepintercept, 
                      "optType" = optType)
  }else{
    # Adding penalty to gaussian and gamma families
    if(object$family %in% c("gaussian", "gamma")){
      df$bestmetrics <- df$bestmetrics + penalty
    }
    # Checking for infinite best metric values
    if(all(is.infinite(df$bestmetrics))){
      stop("no models were found that had an invertible fisher information")
    }
    
    # Only returning best models that have a finite metric value
    newInd <- colSums(df$bestmodels != 0) != 0
    bestInd <- is.finite(df$bestmetrics)
    bestInd <- (newInd + bestInd) == 2
    bestmodels <- df$bestmodels[, bestInd, drop = FALSE]
    
    # Only returning best models that are not the null model
    bestmodels <- sapply(1:length(keep), function(i){
      ind <- which((indices + 1) == i)
      temp <- bestmodels[ind, , drop = FALSE]
      apply(temp, 2, function(x)all(x != 0) * (keep[i] + 0.5) * 2)
    })
    

    if(is.vector(bestmodels)){
      bestmodels <- matrix(bestmodels, ncol = 1)
    }else{
      bestmodels <- t(bestmodels)
    }
    beta <- df$bestmodels[, bestInd, drop = FALSE]
    rownames(bestmodels) <- names
    rownames(beta) <- colnames(object$x)
    FinalList <- list("numchecked" = df$numchecked,
                      "type" = type, 
                      "metric" = metric,
                      "bestmodels" = bestmodels,
                      "bestmetrics" = df$bestmetrics[bestInd],
                      "beta" = beta,
                      "names" = names, 
                      "initmodel" = object, 
                      "cutoff" = cutoff, 
                      "keep" = keep1,
                      "keepintercept" = keepintercept, 
                      "optType" = optType)
  }
  structure(FinalList, class = "BranchGLMVS")
}

#' @rdname plot.summary.BranchGLMVS 
#' @export
plot.BranchGLMVS <- function(x, ptype = "both", marnames = 7, addLines = TRUE, 
                             type = "b", horiz = FALSE,
                             cex.names = 1, cex.lab = 1, 
                             cex.axis = 1, cex.legend = 1,
                             cols = c("deepskyblue", "indianred", "forestgreen"), 
                             ...){
  plot(summary(x), ptype = ptype, marnames = marnames, 
       addLines = addLines, type = type, horiz = horiz, 
       cex.names = cex.names, cex.lab = cex.lab, 
       cex.axis = cex.axis, cex.legend = cex.legend,
       cols = cols, ...)
}

#' Extract Coefficients from BranchGLMVS or summary.BranchGLMVS Objects
#' @description Extracts beta coefficients from BranchGLMVS or summary.BranchGLMVS objects.
#' @param object a `BranchGLMVS` or `summary.BranchGLMVS` object.
#' @param which a numeric vector of indices or "all" to indicate which models to 
#' get coefficients from, the default is 1 which is used for the best model. For 
#' the branch and bound algorithms the number k is used for the kth best model 
#' and for the stepwise algorithms the number k is used for the model that is k - 1 steps 
#' away from the final model.
#' @param ... ignored.
#' @return A numeric matrix with the corresponding coefficient estimates.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, 
#' family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' 
#' ## Getting coefficients from best model
#' coef(VS, which = 1)
#' 
#' ## Getting coefficients from all best models
#' coef(VS, which = "all")
#' 
#' @export
coef.BranchGLMVS <- function(object, which = 1, ...){
  ## Checking which
  if(!is.numeric(which) && is.character(which) && length(which) == 1){
    if(tolower(which) == "all"){
      which <- 1:NCOL(object$beta)
    }
    else{
      stop("which must be a numeric vector or 'all'")
    }
  }else if(!is.numeric(which)){
    stop("which must be a numeric vector or 'all'")
  }else if(any(which < 1)){
    stop("integers provided in which must be positive")
  }else if(any(which > NCOL(object$bestmodels))){
    stop("integers provided in which must be less than or equal to the number of best models")
  }
  
  ## Getting coefficients from all models in which
  allcoefs <- object$beta[, which, drop = FALSE]
  rownames(allcoefs) <- colnames(object$initmodel$x)
  
  ## Adding column names to identify each model
  colnames(allcoefs) <- paste0("Model", which)
  return(allcoefs)
}

#' Predict Method for BranchGLMVS or summary.BranchGLMVS Objects
#' @description Obtains predictions from BranchGLMVS or summary.BranchGLMVS objects.
#' @param object a `BranchGLMVS` or `summary.BranchGLMVS` object.
#' @param which a positive integer to indicate which model to get predictions from, 
#' the default is the best model.
#' @param which a positive integer to indicate which model to get predictions from, 
#' the default is 1 which is used for the best model. For the branch and bound 
#' algorithms the number k is used for the kth best model and for the stepwise algorithms 
#' the number k is used for the model that is k - 1 steps away from the final model.
#' @param ... further arguments passed to [predict.BranchGLM].
#' @seealso [predict.BranchGLM]
#' @return A numeric vector of predictions.
#' @export
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, 
#' family = "gamma", link = "log")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' 
#' ## Getting predictions from best model
#' predict(VS, which = 1)
#' 
#' ## Getting linear predictors from 5th best model
#' predict(VS, which = 5, type = "linpreds")
#' 
predict.BranchGLMVS <- function(object, which = 1, ...){
  ## Checking which
  if(!is.numeric(which) || length(which) != 1){
    stop("which must be a positive integer")
  }
  
  ### Getting BranchGLM object
  myfit <- object$initmodel
  myfit$coefficients[, 1] <- coef(object, which = which)
  
  ### Getting predictions
  predict(myfit, ...)
}

#' Print Method for BranchGLMVS Objects
#' @description Print method for BranchGLMVS objects.
#' @param x a `BranchGLMVS` object.
#' @param digits number of digits to display.
#' @param ... further arguments passed to other methods.
#' @return The supplied `BranchGLMVS` object.
#' @export

print.BranchGLMVS <- function(x, digits = 2, ...){
  cat("Variable Selection Info:\n")
  cat(paste0(rep("-", 24), collapse = ""))
  cat("\n")
  if(!(x$type %in% c("backward"))){
    cat(paste0("Variables were selected using ", x$type, " selection with ", x$metric, "\n"))
  }else{
    cat(paste0("Variables were selected using ", x$type, " elimination with ", x$metric, "\n"))
  }
  
  if(x$optType == "exact"){
    if(x$cutoff >= 0){
      if(length(x$bestmetrics) == 1){
        cat(paste0("Found 1 model within ", round(x$cutoff, digits), " " , x$metric, 
                   " of the best ", x$metric, "(", round(x$bestmetrics[1], digits = digits), ")\n"))
      }else{
        cat(paste0("Found ", length(x$bestmetrics), " models within ", 
                   round(x$cutoff, digits), " " , x$metric, 
                   " of the best ", x$metric, "(", round(x$bestmetrics[1], digits = digits), ")\n"))
      }
    }else{
      if(length(x$bestmetrics) == 1){
        cat(paste0("Found the top model with ", x$metric, " = ", round(x$bestmetrics[1], digits = digits), "\n"))
      }else{
        cat(paste0("The range of ", x$metric, " values for the top ", length(x$bestmetrics), 
                   " models is (", round(x$bestmetrics[1], digits = digits), 
                   ", ", round(x$bestmetrics[length(x$bestmetrics)], digits = digits), ")\n"))
      }
    }
  }else{
    cat(paste0("The top model found had ", x$metric, " = ", round(x$bestmetrics[1], digits = digits), "\n"))
  }
  cat(paste0("Number of models fit: ", x$numchecked))
  cat("\n")
  if(!is.null(x$keep) || x$keepintercept){
    temp <- x$keep
    if(x$keepintercept){
      temp <- c("(Intercept)", temp)
    }
    cat("Variables that were kept in each model: ", paste0(temp, collapse = ", "))
  }
  cat("\n")
  if(length(x$order) == 0){
    if(x$type == "forward"){
      cat("No variables were added to the model")
    }else if(x$type %in% c("backward")){
      cat("No variables were removed from the model")
    }
  }else if(x$type == "forward" ){
    cat("Order the variables were added to the model:\n")
  }else if(x$type %in% c("backward")){
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

