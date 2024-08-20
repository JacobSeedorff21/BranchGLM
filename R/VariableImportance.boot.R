#' Performs Parametric Bootstrap for Modified Variable Importance
#' @description Performs a version of the parametric bootstrap to create an 
#' approximate null distribution for the modified variable importance 
#' values in order to get approximate p-values.
#' @param object a `BranchGLMVS` or `BranchGLMVI` object.
#' @param nboot the number of bootstrap replications to perform.
#' @param parallel a logical value to indicate if parallelization should be used.
#' @param nthreads number of threads used with OpenMP, only used if `parallel = TRUE`.
#' @param showprogress a logical value to indicate if a progress bar should be displayed.
#' @param ... further arguments to [VariableImportance] when `object` is of 
#' class `BranchGLMVS`.
#' @seealso [hist.BranchGLMVI.boot], [boxplot.BranchGLMVI.boot], [VariableImportance]
#' @return a `BranchGLMVI.boot` object which is a list with the following components
#' \item{`summary`}{ a data.frame with the observed modified variable importance values and approximate p-values}
#' \item{`results`}{ a numeric matrix with the modified variable importance values for each set of bootstrap replications}
#' \item{`pvals`}{ a numeric vector with the approximate p-values based on modified variable importance}
#' \item{`nboot`}{ the number of bootstrap replications performed}
#' \item{`metric`}{ the metric used to calculate the modified variable importance values}
#' \item{`VI`}{ the supplied `BranchGLMVI` object}
#' @details This performs a version of the parametric bootstrap with the modified variable 
#' importance values to generate approximate p-values for the sets of variables. 
#' We are currently working on a paper that describes this function in further detail.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' showprogress = FALSE)
#' 
#' # Getting approximate null distributions
#' set.seed(40174)
#' myBoot <- VariableImportance.boot(VS, showprogress = FALSE)
#' myBoot
#' 
#' # Plotting histogram of results for second set of variables
#' hist(myBoot, which = 2)
#' 
#' # Plotting boxplots of results
#' oldmar <- par("mar")
#' par(mar = c(4, 6, 3, 1) + 0.1)
#' boxplot(myBoot, las = 1)
#' par(mar = oldmar)
#' 
#' @export

VariableImportance.boot <- function(object, ...){
  UseMethod("VariableImportance.boot")
}

#'@rdname VariableImportance.boot
#'@export

VariableImportance.boot.BranchGLMVS <- function(object, nboot = 100, parallel = FALSE, 
                                                nthreads = 8, showprogress = TRUE, ...){
  ### Getting observed variable importance
  VI <- VariableImportance(object, showprogress = showprogress, parallel = parallel, 
                           nthreads = nthreads, ...)
  
  ### Performing bootstrap
  return(VariableImportance.boot(VI, nboot = nboot, 
                                 parallel = parallel, nthreads = nthreads, 
                                 showprogress = showprogress))
}

#'@rdname VariableImportance.boot
#'@export

VariableImportance.boot.BranchGLMVI <- function(object, nboot = 100, 
                                                parallel = FALSE, nthreads = 8, 
                                                showprogress = TRUE, ...){
  ## Validating supplied arguments
  if(length(parallel) > 1 || !is.logical(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  
  ### checking nthreads
  if((length(nthreads) > 1) || !is.numeric(nthreads) || (nthreads <= 0)){
    warning("Please select a positive integer for nthreads, using nthreads = 8")
    nthreads <- 8
  }
  
  ## Keeping assign from x matrix since subsetting rows removes it 
  fit <- object$VS$initmodel
  VS <- object$VS
  assign <- attr(fit$x, "assign")
  indices <- attr(fit$x, "assign")
  counts <- table(indices)
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
  
  ## Getting penalties
  metric <- VS$metric
  type <- VS$type
  
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
  
  ## Setting starting model and saving keep1 for later use since keep is modified
  keep <- VS$keep
  keep1 <- keep
  keepintercept <- VS$keepintercept
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
  testInd <- which(keep != -1)
  
  
  # Getting coefs
  coefs <- list()
  disps <- rep(NA, length(testInd))
  for(k in 1:length(testInd)){
    i <- testInd[k]
    Ind <- ((indices + 1) != i)
    x <- fit$x[, Ind, drop = FALSE]
    tempFit <- BranchGLMfit(x, fit$y, fit$offset, rep(0, ncol(x)), 
                            fit$method, fit$grads, fit$link, fit$family, 
                            nthreads, fit$tol, fit$maxit, TRUE)
    coefs[[k]] <- tempFit$coefficients[, 1]
    disps[k] <-  tempFit$dispersion[[1]]
  }
  
  ## Fitting model if necessary for dispersion parameter
  if(all(is.na(fit$coefficients))){
    disp <- BranchGLMfit(fit$x, fit$y, fit$offset, rep(0, ncol(fit$x)), 
                        fit$method, fit$grads, fit$link, fit$family, 
                        nthreads, fit$tol, fit$maxit, TRUE)$dispersion[[1]]
  }else{
    disp <- fit$dispersion[[1]]
  }
  
  ## Creating bootstrapping function
  mybootstrap <- function(notused){
    ## Getting model
    tests <- rep(NA, length(testInd))
    for(k in 1:length(testInd)){
      # Performing parametric bootstrap with the current coefficient set to zero
      i <- testInd[k]
      Ind <- ((indices + 1) != i)
      newIndices <- indices[Ind]
      newIndices[newIndices + 1 > i] <- newIndices[newIndices + 1 > i] - 1
      newCounts <- counts[-i]
      x <- fit$x[, Ind, drop = FALSE]
      newY <- simulateY(x %*% coefs[[k]], fit$link, fit$family, disp)
      
      # Getting best metric without this variables
      without <- helper(type, fit$x[, Ind], newY, fit$offset, newIndices, newCounts, 
                        interactions[-i, -i], fit$method, fit$grads,
                        fit$link, fit$family, nthreads, 
                        fit$tol, fit$maxit, keep[-i], pen[-i], penalty)
      
      # Getting best metric with this variable
      tempKeep <- keep
      tempKeep[i] <- -1
      with <- helper(type, fit$x, newY, fit$offset, indices, counts, 
                     interactions, fit$method, fit$grads,
                     fit$link, fit$family, nthreads, 
                     fit$tol, fit$maxit, tempKeep, pen, penalty)
      
      # Getting test statistic
      tests[k] <- without$bestmetrics - with$bestmetrics + pen[i]
    }
    ## Updating progress bar
    if(showprogress){
      setTxtProgressBar(pb, notused) 
    }
    return(tests)
  }
  
  ## Performing bootstrapping
  if(showprogress){
    pb <- txtProgressBar(min = 0, max = nboot, style = 3)
    on.exit(close(pb))
  }
  res <- sapply(1:nboot, mybootstrap)
  
  ## Getting results in right format
  tests <- object$results$mVI
  results <- matrix(nrow = length(tests), ncol = nboot)
  rownames(results) <- object$VS$names
  results[testInd, ] <- res
  
  ## Getting p-values
  pvals <- rowMeans(tests <= results)
  summary <- data.frame("mVI" = tests, "p.values" = pvals)
  rownames(summary) <- object$VS$names
  
  ## Creating final object
  res <- list("summary" = summary, 
              "results" = results, 
              "pvals" = pvals,
              "metric" = metric,
              "nboot" = nboot,
              "VI" = object)
  return(structure(res, class = "BranchGLMVI.boot"))
}

#' Generate Simulated Response Values
#' @param linpreds numeric vector of linear predictors.
#' @param link the specified link.
#' @param family the specified family.
#' @param the estimated dispersion parameter.
#' @noRd

# Function used to generate simulated response values
simulateY <- function(linpreds, link, family, dispersion){
  ## Getting mean
  if(link == "identity"){
    mu <- linpreds
  }else if(link == "log"){
    mu <- exp(linpreds)
  }else if(link == "sqrt"){
    mu <- (linpreds)^2
  }else if(link == "logit"){
    mu <- 1 / (1 + exp(-linpreds))
  }else if(link == "probit"){
    mu <- pnorm(linpreds)
  }else if(link == "cloglog"){
    mu <- 1 - exp(-exp(linpreds))
  }else if(link == "inverse"){
    mu <- 1 / linpreds
  }else{
    stop("link not valid")
  }
  
  ## Simulating data
  if(family == "gaussian"){
    Y <- rnorm(length(mu), mu, sqrt(dispersion))
  }else if(family == "binomial"){
    Y <- rbinom(length(mu), 1, mu)
  }else if(family == "gamma"){
    Y <- rgamma(length(mu), shape = 1 / dispersion, scale = mu * dispersion)
  }else if(family == "poisson"){
    Y <- rpois(length(mu), mu)
  }else{
    stop("family not valid")
  }
  return(Y)
}

#' Histogram Method for BranchGLMVI.boot Objects
#' @description Creates histograms of approximate null distributions for 
#' the modified variable importance values.
#' @param x a `BranchGLMVI.boot` object.
#' @param which which approximate null distributions to plot, can use a numeric vector of 
#' indices, a character vector of names, or "all" for all variables. The default 
#' is to create histograms for each set of variables that are not kept in each model.
#' @param linecol the color of the line which indicates the observed modified variable 
#' importance values.
#' @param linelwd the width of the line which indicates the observed modified variable 
#' importance values.
#' @param xlim a numeric vector of length 2, giving the x coordinates range.
#' @param xlab a label for the x axis.
#' @param main a main title for the plot.
#' @param ... further arguments passed to [hist.default].
#' @seealso [boxplot.BranchGLMVI.boot]
#' @return This only produces a plot, nothing is returned.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' showprogress = FALSE)
#' 
#' # Getting approximate null distributions
#' set.seed(40174)
#' myBoot <- VariableImportance.boot(VS, showprogress = FALSE)
#' 
#' # Plotting histograms of second set of variables
#' hist(myBoot, which = 2)
#' 
#' # Plotting histograms of third set of variables
#' hist(myBoot, which = 3, linecol = "blue", linelwd = 5)
#' 
#'@export

hist.BranchGLMVI.boot <- function(x, which = "all", linecol = "red", linelwd = 2, 
                                  xlim = NULL, xlab = "Modified Variable Importance", 
                                  main = NULL, ...){
  # Getting which
  if(is.character(which) && length(which) == 1 && tolower(which) == "all"){
    which <- 1:nrow(x$results)
  }else if(is.character(which)){
    which <- match(which, rownames(x$results))
  }else if(is.numeric(which) && all(which <= 0)){
    which <- (1:nrow(x$results))[which]
  }
  
  if(is.null(main)){
    metricName <- paste0("(", x$metric, ")")
    if(x$VI$VS$type == "forward"){
      main <- paste0("Null Distributions of FS mVI", metricName)
    }else if(x$VI$VS$type %in% c("backward", "fast backward")){
      main <- paste0("Null Distributions of BE mVI", metricName)
    }else if(x$VI$VS$type %in% c("double backward", "fast double backward")){
      main <- paste0("Null Distributions of DBE mVI", metricName)
    }else{
      main <- paste0("Null Distributions of BSS mVI", metricName)
    }
  }
  
  if(all(is.na(x$results[which, ]))){
    stop("must specify at least one variable that is not kept through selection")
  }
  
  for(i in which){
    if(!all(is.na(x$results[i, ]))){
      obs <- x$VI$results$mVI[i]
      if(is.null(xlim)){
        xlim2 <- range(x$results[i, ])
        xlim2[1] <- min(xlim2[1], obs)
        xlim2[2] <- max(xlim2[2], obs)
      }
      hist(x$results[i, ], xlab = xlab, xlim = xlim2,
           main = paste0(main, " for ", rownames(x$results)[i]), 
           ...)
      abline(v = obs, col = linecol, lwd = linelwd)
    }
  }
}

#' Box Plot Method for BranchGLMVI.boot Objects
#' @description Creates box-and-whisker plots of approximate null distributions for 
#' the modified variable importance values.
#' @param x a `BranchGLMVI.boot` object.
#' @param which which approximate null distributions to plot, can use a numeric vector of 
#' indices, a character vector of names, or "all" for all variables. The default 
#' is to create box-and-whisker plots for each set of variables that are not 
#' kept in each model.
#' @param linecol the color of the line which indicates the observed modified variable 
#' importance values.
#' @param linelwd the width of the line which indicates the observed modified variable 
#' importance values.
#' @param horizontal a logical value indicating if the boxplots should be horizontal.
#' @param lim a numeric vector of length 2, giving the coordinates range.
#' @param show.names set to TRUE or FALSE to override the defaults on whether an axis label is printed for each group.
#' @param lab a label for the axis corresponding to the modified variable importance values.
#' @param main a main title for the plot.
#' @param las the style of axis labels, see more at [par].
#' @param ... further arguments passed to [boxplot.default].
#' @seealso [hist.BranchGLMVI.boot]
#' @return This only produces a plot, nothing is returned.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' showprogress = FALSE)
#' 
#' # Getting approximate null distributions
#' set.seed(40174)
#' myBoot <- VariableImportance.boot(VS, showprogress =  FALSE)
#' 
#' # Plotting boxplots of selected sets of variables
#' oldmar <- par("mar")
#' par(mar = c(4, 6, 3, 1) + 0.1)
#' boxplot(myBoot, las = 1)
#' par(mar = oldmar)
#' 
#' # Plotting boxplots of selected sets of variables
#' boxplot(myBoot, las = 1, cex.axis = 0.55)
#' 
#'@export

boxplot.BranchGLMVI.boot <- function(x, which = "all", linecol = "red", linelwd = 2, 
                                     horizontal = TRUE, lim = NULL, show.names = TRUE,
                                     lab = "Modified Variable Importance", main = NULL,
                                     las = ifelse(horizontal, 1, 2), ...){
  # Getting which
  if(is.character(which) && length(which) == 1 && tolower(which) == "all"){
    which <- 1:nrow(x$results)
  }else if(is.character(which)){
    which <- match(which, rownames(x$results))
  }else if(is.numeric(which) && all(which <= 0)){
    which <- (1:nrow(x$results))[which]
  }
  which <- which[!is.na(x$VI$results$mVI[which])]
  
  if(is.null(main)){
    metricName <- paste0("(", x$metric, ")")
    if(x$VI$VS$type == "forward"){
      main <- paste0("Null Distributions of FS mVI", metricName)
    }else if(x$VI$VS$type %in% c("backward", "fast backward")){
      main <- paste0("Null Distributions of BE mVI", metricName)
    }else if(x$VI$VS$type %in% c("double backward", "fast double backward")){
      main <- paste0("Null Distributions of DBE mVI", metricName)
    }else{
      main <- paste0("Null Distributions of BSS mVI", metricName)
    }
  }
  
  # Making boxplots
  obs <- x$VI$results$mVI[which]
  if(all(is.na(obs))){
    stop("must specify at least one variable that is not kept through selection")
  }
  if(is.null(lim)){
    lim <- range(x$results[which, ])
    lim[1] <- 0
    lim[2] <- max(lim[2], max(obs))
  }
  if(horizontal){
    boxplot(x$results[which, , drop = FALSE], xlab = lab, use.cols = FALSE, 
            ylim = lim, main = main, horizontal = TRUE, las = las, 
            show.names = show.names, ...)
    
    # Adding in observed VIs
    segments(x0 = obs, y0 = 1:length(which) - 0.5, y1 = 1:length(which) + 0.5, 
             col = linecol, lwd = linelwd)
  }else{
    boxplot(x$results[which, , drop = FALSE], ylab = lab, use.cols = FALSE,
            ylim = lim, main = main, horizontal = FALSE, show.names = show.names, 
            las = las, ...)
    
    # Adding in observed VIs
    segments(y0 = obs, x0 = 1:length(which) - 0.5, x1 = 1:length(which) + 0.5, 
             col = linecol, lwd = linelwd)
  }
}

#' Print Method for BranchGLMVI.boot Objects
#' @description Print method for BranchGLMVI.boot objects.
#' @param x a `BranchGLMVI.boot` object.
#' @param digits number of significant digits to display.
#' @param ... further arguments passed to other methods.
#' @return The supplied `BranchGLMVI.boot` object.
#' @export

print.BranchGLMVI.boot <- function(x, digits = 3, ...){
  metricName <- paste0("(", x$metric, ")\n")
  if(x$VI$VS$type == "forward"){
    temp <- paste0("Null Distributions of FS mVI", metricName)
  }else if(x$VI$VS$type %in% c("backward", "fast backward")){
    temp <- paste0("Null Distributions of BE mVI", metricName)
  }else if(x$VI$VS$type %in% c("double backward", "fast double backward")){
    temp <- paste0("Null Distributions of DBE mVI", metricName)
  }else{
    temp <- paste0("Null Distributions of BSS mVI", metricName)
  }
  cat(temp)
  cat(paste0(rep("-", nchar(temp)), collapse = ""))
  cat("\n")
  print(x$summary, digits = digits)
  invisible(x)
}
