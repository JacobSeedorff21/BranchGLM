#' Computes Exact or Approximate L0-penalization based Variable Importance for GLMs
#' @description Gets exact or approximate L0-penalization based variable importance values for generalized linear 
#' models (GLMs). More details about what the variable importance values are can 
#' be found in the details section. 
#' @param object an object of class `BranchGLMVS`.
#' @param VIMethod one of "separate" or "simultaneous" to denote the method used to find 
#' the variable importance values. This is ignored if the type of variable selection 
#' employed in `object` was a heuristic method.
#' @param parallel a logical value to indicate if parallelization should be used.
#' @param nthreads number of threads used with OpenMP, only used if `parallel = TRUE`.
#' @param showprogress a logical value to indicate whether or not to show progress updates.
#' @seealso [VariableImportance.boot], [barplot.BranchGLMVI]
#' @details
#' 
#' Note that variable importance values can only be found for sets of variables that are not kept 
#' through the model selection process. More details about the variable importance 
#' values will be made available in an upcoming paper.
#' 
#' When a branch and bound algorithm is used in `object`, then the exact variable 
#' importance values are computed. When a heuristic method is used, then approximate 
#' variable importance values are computed based on the specified heuristic method.
#' 
#' @return A `BranchGLMVI` object which is a list with the following components
#' \item{`results`}{ a data.frame with the variable importance values and degrees of freedom}
#' \item{`metric`}{ metric used to select the best models}
#' \item{`numchecked`}{ number of models fit}
#' \item{`VS`}{ the supplied `BranchGLMVS` object}
#' \item{`with`}{ a numeric matrix with the best models that include each set of variables}
#' \item{`withmetrics`}{ a numeric vector with the metric values for the best models with each set of variables}
#' \item{`without`}{ a numeric matrix with the best models that exclude each set of variables}
#' \item{`withoutmetrics`}{ a numeric vector with the metric values for the best models without each set of variables}
#' @name VariableImportance
#' 
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' showprogress = FALSE)
#' 
#' # Getting variable importance
#' VI <- VariableImportance(VS, showprogress = FALSE)
#' VI
#' 
#' # Plotting variable importance
#' oldmar <- par("mar")
#' par(mar = c(4, 6, 3, 1) + 0.1)
#' barplot(VI)
#' par(mar = oldmar)
#' 
#' @references Seedorff J, Cavanaugh JE. *Assessing Variable Importance for Best 
#' Subset Selection. Entropy. 2024; 26(9):801*. \doi{https://doi.org/10.3390/e26090801}
#' 
#' @export
#' 

VariableImportance <- function(object, VIMethod = "simultaneous", parallel = FALSE, nthreads = 8,
                              showprogress = TRUE){
  if(!is(object, "BranchGLMVS")){
    stop("object must be an object of class BranchGLMVS")
  }
  
  ## Checking VIMethod
  VIMethod <- tolower(VIMethod)
  if(!is.character(VIMethod) || length(VIMethod) != 1 || 
     !(VIMethod %in% c("separate", "simultaneous"))){
    stop("VIMethod must be one of 'separate' or 'simultaneous'")
  }
  
  ## Getting initial fit
  fit <- object$initmodel
  
  ## Checking if supplied BranchGLM object has x and data
  if(is.null(object$initmodel$data) || is.null(object$initmodel$x)){
    stop("the supplied model must have a data and an x component")
  }
  ## Checking if supplied BranchGLM object has y
  if(is.null(object$initmodel$y)){
    stop("the supplied model must have a y component")
  }
  
  ## Validating supplied arguments
  if(length(parallel) > 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  
  ### checking nthreads
  if((length(nthreads) > 1) || !is.numeric(nthreads) || (nthreads <= 0)){
    warning("Please select a positive integer for nthreads, using nthreads = 8")
    nthreads <- 8
  }
  
  
  indices <- attr(fit$x, "assign")
  interactions <- attr(fit$terms, "factors")[-1L, ]
  
  ## Removing rows with all zeros
  interactions <- interactions[apply(interactions, 1, function(x){sum(x) > 0}),]

  ## Checking for intercept
  if(colnames(fit$x)[1] == "(Intercept)"){
    intercept <- TRUE
    interactions <- rbind(0, interactions)
    interactions <- cbind(0, interactions)
  }else{
    intercept <- FALSE
    indices <- indices - 1
  }
  
  counts <- table(indices)
  maxsize <- length(counts)
  
  ## Setting starting model and saving keep1 for later use since keep is modified
  keep <- object$keep
  keep1 <- keep
  keepintercept <- object$keepintercept
  if(is.null(keep)){
    keep <- rep(0, length(counts))
    if(intercept && keepintercept){
      keep[1] <- -1
    }
  }else{
    CurNames <- attributes(terms(fit$formula, 
                                 data = fit$data))$factors |>
      colnames()
    keep <- (CurNames %in% keep) * -1
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
  metric <- object$metric
  if(metric == "AIC"){
    pen <- as.vector(counts) * 2
    penalty <- 2
  }else if(metric == "BIC"){
    pen <- as.vector(counts) * log(nrow(fit$x))
    penalty <- log(nrow(fit$x))
  }else if(metric == "HQIC"){
    pen <- as.vector(counts) * 2 * log(log(nrow(fit$x)))
    penalty <- 2 * log(log(nrow(fit$x)))
  }
  
  # Getting type
  type <- object$type
  if(object$optType == "exact"){
    # Getting withmodels and withoutmodels
    varInd <- which(keep != -1)
    
    # Getting best models with each variable that we already have
    withInd <- sapply(varInd, function(i){
      x <- object$bestmodels[i, ]
      temp <- which(x == 1)
      if(length(temp) == 0){
        return(NA)
      }else{
        return(min(temp))
      }
    })
    
    ## as.numeric is necessary if all values are NA then withInd is logical
    withModels <- object$bestmodels[, as.numeric(withInd)]
    withMetrics <- object$bestmetrics[withInd]
    withMetrics[is.na(withMetrics)] <- Inf 
    
    # Getting best models without each variable that we already have
    withoutInd <- sapply(varInd, function(i){
      x <- object$bestmodels[i, ]
      temp <- which(x == 0)
      if(length(temp) == 0){
        return(NA)
      }else{
        return(min(temp))
      }
    })
    
    ## as.numeric is necessary if all values are NA then withoutInd is logical
    withoutModels <- object$bestmodels[, as.numeric(withoutInd)]
    withoutMetrics <- object$bestmetrics[withoutInd]
    withoutMetrics[is.na(withoutMetrics)] <- Inf 
    
    if(VIMethod == "simultaneous"){
      withModels[is.na(withModels)] <- 0
      withoutModels[is.na(withoutModels)] <- 0
      withMetrics2 <- withMetrics
      withMetrics2[!is.infinite(withMetrics2)] <- -Inf
      withoutMetrics2 <- withoutMetrics
      withoutMetrics2[!is.infinite(withoutMetrics2)] <- -Inf
      
      ## Getting variable importance values
      df <- SwitchVariableImportanceCpp(fit$x, fit$y, fit$offset, indices, counts, 
                             interactions, 
                             withModels, withMetrics2, 
                             withoutModels, withoutMetrics2, 
                             fit$method, fit$grads,
                             fit$link, fit$family, nthreads, 
                             fit$tol, fit$maxit, keep,
                             pen, showprogress)
      
      if(fit$family %in% c("gaussian", "gamma")){
        df$withmetrics <- df$withmetrics + penalty
        df$withoutmetrics <- df$withoutmetrics + penalty
      }
      df$withmetrics[!is.infinite(withMetrics)] <- 
        withMetrics[!is.infinite(withMetrics)]
      df$withoutmetrics[!is.infinite(withoutMetrics)] <- 
        withoutMetrics[!is.infinite(withoutMetrics)]
      df$withmetrics <- as.vector(df$withmetrics)
      df$withoutmetrics <- as.vector(df$withoutmetrics)
    }else{
      numchecked <- 0
      k <- 1
      count <- 0
      for(j in 1:length(varInd)){
        i <- varInd[j]
        if(is.na(withoutInd[j])){
          Ind <- ((indices + 1) != i)
          newIndices <- indices[Ind]
          newIndices[newIndices + 1 > i] <- newIndices[newIndices + 1 > i] - 1
          newCounts <- counts[-i]
          temp <- helper(type, fit$x[, Ind], fit$y, fit$offset, newIndices, newCounts, 
                         interactions[-i, -i], fit$method, fit$grads,
                         fit$link, fit$family, nthreads, 
                         fit$tol, fit$maxit, keep[-i], pen[-i], penalty)
          tempmodel <- temp$bestmodels
          withoutModel <- rep(0, nrow(object$bestmodels))
          withoutModel[-i] <- tempmodel
          withoutModels[, k] <- withoutModel
          withoutMetrics[k] <- temp$bestmetrics
          numchecked <- numchecked + temp$numchecked
          count <- count + 1
          ### progress check
          if(showprogress && object$bestmodels[i] != -1){
            cat(paste0("Finished ", count,  " out of ", sum(is.na(c(withInd, withoutInd))), 
                       " branch and bound algorithms \n"))
          }
        }else if(is.na(withInd[j])){
          tempKeep <- keep
          tempKeep[i] <- -1
          temp <- helper(type, fit$x, fit$y, fit$offset, indices, counts, 
                         interactions, fit$method, fit$grads,
                         fit$link, fit$family, nthreads, 
                         fit$tol, fit$maxit, tempKeep, pen, penalty)
          tempmodel <- temp$bestmodels
          withModel <- tempmodel
          withModel[i] <- 1
          withModels[, k] <- withModel
          withMetrics[k] <- temp$bestmetrics
          numchecked <- numchecked + temp$numchecked
          count <- count + 1
          ### progress check
          if(showprogress){
            cat(paste0("Finished ", count,  " out of ", sum(is.na(c(withInd, withoutInd))), 
                       " branch and bound algorithms \n"))
          }
        }
        k <- k + 1
      }
      ### Creating df
      df <- list("withmodels" = withModels,
                 "withoutmodels" = withoutModels,
                 "numchecked" = numchecked,
                 "withmetrics" = withMetrics,
                 "withoutmetrics" = withoutMetrics)
    }
  }else{
    ## Creating objects to store results
    bestModel <- object$bestmodels[, 1]
    bestMetric <- object$bestmetrics[1]
    varInd <- which(bestModel != -1)
    withoutModels <- withModels <- matrix(NA, nrow = nrow(object$bestmodels), 
                                              ncol = length(varInd))
    withoutMetrics <- withMetrics <- rep(NA, length(varInd))
    if(type == "forward"){
      ## Adding models for variables that weren't in final model
      helpInd <- which(bestModel == 0)
      helpInd <- match(helpInd, varInd)
      withoutModels[, helpInd] <- rep(bestModel, length(helpInd))
      withoutMetrics[helpInd] <- bestMetric
    }else{
      ## Adding models for variables that were in final model
      helpInd <- which(bestModel == 1)
      helpInd <- match(helpInd, varInd)
      withModels[, helpInd] <- rep(bestModel, length(helpInd))
      withMetrics[helpInd] <- bestMetric
    }
    
    numchecked <- 0
    total <- sum(is.na(c(withMetrics, withoutMetrics)))
    k <- 1
    count <- 0
    for(j in 1:length(varInd)){
      i <- varInd[j]
      if(is.na(withoutMetrics[j])){
        # Getting best model without variables
        Ind <- ((indices + 1) != i)
        newIndices <- indices[Ind]
        newIndices[newIndices + 1 > i] <- newIndices[newIndices + 1 > i] - 1
        newCounts <- counts[-i]
        temp <- helper(type, fit$x[, Ind], fit$y, fit$offset, newIndices, newCounts, 
                       interactions[-i, -i], fit$method, fit$grads,
                       fit$link, fit$family, nthreads, 
                       fit$tol, fit$maxit, keep[-i], pen[-i], penalty)
        tempmodel <- temp$bestmodels
        withoutModel <- rep(0, nrow(object$bestmodels))
        withoutModel[-i] <- tempmodel
        withoutModels[, j] <- withoutModel
        withoutMetrics[j] <- temp$bestmetrics
        numchecked <- numchecked + temp$numchecked
        count <- count + 1
      }
      
      if(is.na(withMetrics[j])){
        tempKeep <- keep
        tempKeep[i] <- -1
        temp <- helper(type, fit$x, fit$y, fit$offset, indices, counts, 
                       interactions, fit$method, fit$grads,
                       fit$link, fit$family, nthreads, 
                       fit$tol, fit$maxit, tempKeep, pen, penalty)
        tempmodel <- temp$bestmodels
        withModel <- tempmodel
        withModel[i] <- 1
        withModels[, j] <- withModel
        withMetrics[j] <- temp$bestmetrics
        numchecked <- numchecked + temp$numchecked
        count <- count + 1
      }
      
      ### progress check
      if(showprogress){
        cat(paste0("Finished ", count,  " out of ", total, " selections \n"))
      }
    }
    ### Creating df
    df <- list("withmodels" = withModels,
               "withoutmodels" = withoutModels,
               "numchecked" = numchecked,
               "withmetrics" = withMetrics,
               "withoutmetrics" = withoutMetrics)
  }
  
    # Creating data frame to store results
  CurNames <- attributes(terms(fit$formula, data = fit$data))$factors |>
    colnames()
  if(intercept){
    CurNames <- c("(Intercept)", CurNames)
  }
  Names <- CurNames[!(CurNames %in% keep1)]
  if(intercept && keepintercept){
    Names <- Names[-1]
  }
  
  tests <- rep(NA, length(CurNames))
  names(tests) <- CurNames
  tests[Names] <- df$withoutmetrics - df$withmetrics
  counts <- as.vector(counts)
  tempTests <- tests
  tests <- tests + pen
  tests <- data.frame("VI" = tempTests, 
                      "mVI" = tests,
                      "df" = ifelse(is.na(tests), NA, unname(counts)))
  rownames(tests) <- CurNames
    
  # Only keeping important stuff from process
  with <- df$withmodels
  without <- df$withoutmodels
  rownames(with) <- rownames(without) <- CurNames
  colnames(with) <- colnames(without) <- Names
  res <- list("results" = tests,
              "metric" = metric, 
              "numchecked" = df$numchecked, 
              "VS" = object, 
              "with" = with,
              "withmetrics" = df$withmetrics,
              "without" = without,
              "withoutmetrics" = df$withoutmetrics)
  
  structure(res, class = "BranchGLMVI")
}

#' Performs Variable Selection with Design Matrix
#' @description Performs variable selection with design matrix. Improper 
#' use of this function can cause R to crash.
#' @param type the type of variable selection to perform.
#' @param x the design matrix.
#' @param y the response vector.
#' @param offset the offset vector.
#' @param indices a numeric vector which can be used to group together the sets of 
#' variables.
#' @param counts a numeric vector which has the number of variables in each set 
#' of variables.
#' @param interactions A numeric matrix indicating which sets of variables are 
#' interaction terms based on which other sets of variables.
#' @param method one of "Fisher", "BFGS", or "LBFGS". Fisher's scoring is recommended
#' for forward selection and branch and bound methods since they will typically 
#' fit many models with a small number of covariates.
#' @param grads number of gradients used to approximate inverse information with, only for `method = "LBFGS"`.
#' @param link the link used to link mean structure to linear predictors. One of
#' "identity", "logit", "probit", "cloglog", "sqrt", "inverse", or "log".
#' @param family the distribution used to model the data, one of "gaussian", "gamma", 
#' "binomial", or "poisson".
#' @param nthreads number of threads used with OpenMP.
#' @param tol tolerance used to determine model convergence when fitting GLMs.
#' @param maxit maximum number of iterations performed when fitting GLMs.
#' @param keep a numeric vector of 0s and -1s where -1 indicates that that 
#' set of variables should be kept throughout the search. 
#' @param pen a numeric vector with the penalty terms for including each set of 
#' variables
#' @param penalty the chosen penalty term.
#' @noRd

# Helper function used to perform variable selection with x and y
helper <- function(type, x, y, offset, indices, counts, 
                   interactions, method, grads,
                   link, family, nthreads, 
                   tol, maxit, keep, pen, penalty){
  if(type == "forward"){
    temp <- ForwardCpp(x, y, offset, indices, counts, 
                              interactions, method, grads,
                              link, family, nthreads, 
                              tol, maxit, keep, length(counts), 
                              pen)
    bestInd <- which.min(temp$bestmetrics)
    temp$beta <- temp$betas[, bestInd]
    temp$bestmodels <- temp$bestmodels[, bestInd]
    temp$bestmetrics <- temp$bestmetrics[bestInd]
  }else if(type == "backward"){
    temp <- BackwardCpp(x, y, offset, indices, counts, 
                        interactions, method, grads,
                        link, family, nthreads, 
                        tol, maxit, keep, length(counts), 
                        pen)
    bestInd <- which.min(temp$bestmetrics)
    temp$beta <- temp$betas[, bestInd]
    temp$bestmodels <- temp$bestmodels[, bestInd]
    temp$bestmetrics <- temp$bestmetrics[bestInd]
  }else if(type == "double backward"){
    temp <- DoubleBackwardCpp(x, y, offset, indices, counts, 
                        interactions, method, grads,
                        link, family, nthreads, 
                        tol, maxit, keep, length(counts), 
                        pen)
    bestInd <- which.min(temp$bestmetrics)
    temp$beta <- temp$betas[, bestInd]
    temp$bestmodels <- temp$bestmodels[, bestInd]
    temp$bestmetrics <- temp$bestmetrics[bestInd]
  }else if(type == "fast double backward"){
    temp <- FastDoubleBackwardCpp(x, y, offset, indices, counts, 
                            interactions, method, grads,
                            link, family, nthreads, 
                            tol, maxit, keep, length(counts), 
                            pen)
    bestInd <- which.min(temp$bestmetrics)
    temp$beta <- temp$betas[, bestInd]
    temp$bestmodels <- temp$bestmodels[, bestInd]
    temp$bestmetrics <- temp$bestmetrics[bestInd]
  }else if(type == "fast backward"){
    temp <- FastBackwardCpp(x, y, offset, indices, counts, 
                            interactions, method, grads,
                            link, family, nthreads, 
                            tol, maxit, keep, length(counts), 
                            pen)
    bestInd <- which.min(temp$bestmetrics)
    temp$beta <- temp$betas[, bestInd]
    temp$bestmodels <- temp$bestmodels[, bestInd]
    temp$bestmetrics <- temp$bestmetrics[bestInd]
  }else if(type == "branch and bound"){
    temp <- BranchAndBoundCpp(x, y, offset, indices, counts, 
                              interactions, method, grads,
                              link, family, nthreads, 
                              tol, maxit, keep, ncol(x), 
                              pen, FALSE, 1, -1)
    temp$beta <- temp$bestmodels
  }else if(type == "backward branch and bound"){
    temp <- BackwardBranchAndBoundCpp(x, y, offset, indices, counts, 
                                      interactions, method, grads,
                                      link, family, nthreads, 
                                      tol, maxit, keep, 
                                      pen, FALSE, 1, -1)
    temp$beta <- temp$bestmodels
  }else if(type == "switch branch and bound"){
    temp <- SwitchBranchAndBoundCpp(x, y, offset, indices, counts, 
                                    interactions, method, grads,
                                    link, family, nthreads, 
                                    tol, maxit, keep, 
                                    pen, FALSE, 1, -1)
    temp$beta <- temp$bestmodels
  }else{
    stop("supplied type is not supported")
  }
  if(family %in% c("gaussian", "gamma")){
    temp$bestmetrics <- temp$bestmetrics + penalty
  }
  temp$bestmodels <- (temp$bestmodels[!duplicated(indices)] != 0) * (keep + 0.5) * 2
  temp
}

#' Bar Plot Method for BranchGLMVI Objects
#' @description Creates a bar plot with the L0-penalization based variable importance values.
#' @param height a `BranchGLMVI` object.
#' @param modified a logical value indicating if the modified variable importance 
#' values should be plotted.
#' @param horiz a logical value to indicate whether bars should be horizontal.
#' @param decreasing a logical value to indicate whether variables should be sorted 
#' in decreasing order. Can use NA if no ordering is desired.
#' @param which which variable importance values to plot, can use a numeric vector 
#' of indices, a character vector of names, or "all" for all variables.
#' @param las the style of axis labels, see [par] for more details.
#' @param main the overall title for the plot.
#' @param lab the title for the axis corresponding to the variable importance values.
#' @param ... further arguments passed to [barplot.default].
#' @return This only produces a plot, nothing is returned.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' showprogress = FALSE)
#' 
#' # Getting variable importance
#' VI <- VariableImportance(VS, showprogress = FALSE)
#' VI
#' 
#' # Plotting variable importance
#' oldmar <- par("mar")
#' par(mar = c(4, 6, 3, 1) + 0.1)
#' barplot(VI)
#' par(mar = oldmar)
#' 
#' @export

barplot.BranchGLMVI <- function(height, modified = FALSE, horiz = TRUE, decreasing = FALSE, 
                         which = "all", las = ifelse(horiz, 1, 2), 
                         main = NULL, lab = NULL, ...){
  # Getting main
  if(is.null(main)){
    metricName <- paste0("(", height$metric, ")")
    if(modified){
      if(height$VS$type == "forward"){
        main <- paste0("Forward Selection mVI", metricName)
      }else if(height$VS$type %in% c("backward", "fast backward")){
        main <- paste0("Backward Elimination mVI", metricName)
      }else if(height$VS$type %in% c("double backward", "fast double backward")){
        main <- paste0("Double Backward Elimination mVI", metricName)
      }else{
        main <- paste0("Best Subset Selection mVI", metricName)
      }
    }else{
      if(height$VS$type == "forward"){
        main <- paste0("Forward Selection VI", metricName)
      }else if(height$VS$type %in% c("backward", "fast backward")){
        main <- paste0("Backward Elimination VI", metricName)
      }else if(height$VS$type %in% c("double backward", "fast double backward")){
        main <- paste0("Double Backward Elimination VI", metricName)
      }else{
        main <- paste0("Best Subset Selection VI", metricName)
      }
    }
  }
  
  # Getting lab 
  if(is.null(lab)){
    if(modified){
      lab <- "Modified Variable Importance"
    }else{
      lab <- "Variable Importance"
    }
  }
  
  # Using which
  if(is.character(which) && length(which) == 1 && tolower(which) == "all"){
    which <- 1:nrow(height$results)
  }else if(is.character(which)){
    which <- match(which, rownames(height$results))
  }else if(is.numeric(which) && all(which <= 0)){
    which <- (1:nrow(height$results))[which]
  }
  if(modified){
    results <- height$results$mVI
  }else{
    results <- height$results$VI
  }
  names(results) <- rownames(height$results)
  results <- results[which]
  
  # Getting importance values
  if(all(is.na(results))){
    stop("must specify at least one variable that is not kept through selection")
  }
  
  if(horiz){
    if(!is.na(decreasing)){
      myorder <- sort(results, decreasing = decreasing)
    }else{
      myorder <- results[!is.na(results)]
    }
    
    barplot(myorder, names = names(myorder), 
            main = main,
            horiz = horiz, xlim = c(min(c(0, results), na.rm = TRUE), ceiling(max(myorder))), 
            las = las,
            xlab = lab, ...)
    
  }else{
    if(!is.na(decreasing)){
      myorder <- sort(results, decreasing = decreasing)
    }else{
      myorder <- results[!is.na(results)]
    }
    
    barplot(myorder, names = names(myorder), 
            main = main,
            horiz = horiz, ylim = c(min(c(0, results), na.rm = TRUE), ceiling(max(myorder))), las = las,
            ylab = lab, ...) 
  }
}

#' Print Method for BranchGLMVI Objects
#' @description Print method for BranchGLMVI objects.
#' @param x a `BranchGLMVI` object.
#' @param digits number of significant digits to display.
#' @param ... further arguments passed to other methods.
#' @return The supplied `BranchGLMVI` object.
#' @export

print.BranchGLMVI <- function(x, digits = 3, ...){
  metricName <- paste0("(", x$metric, ")")
  if(x$VS$type == "forward"){
    temp <- paste0("Forward Selection VI", metricName, "\n")
  }else if(x$VS$type %in% c("backward", "fast backward")){
    temp <- paste0("Backward Elimination VI", metricName, "\n")
  }else if(x$VS$type %in% c("double backward", "fast double backward")){
    temp <- paste0("Double Backward Elimination VI", metricName, "\n")
  }else{
    temp <- paste0("Best Subset Selection VI", metricName, "\n")
  }
  cat(temp)
  cat(paste0(rep("-", nchar(temp)), collapse = ""))
  cat("\n")
  print(x$results, digits = digits)
  invisible(x)
}

