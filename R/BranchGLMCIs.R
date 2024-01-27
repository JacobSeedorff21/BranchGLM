#' Likelihood Ratio Confidence Intervals for Beta Coefficients for BranchGLM Objects
#' @description Finds profile likelihood ratio confidence intervals for beta 
#' coefficients with the ability to calculate the intervals in parallel. 
#' @param object a `BranchGLM` object.
#' @param parm a specification of which parameters are to be given confidence intervals, 
#' either a vector of numbers or a vector of names. If missing, all parameters are considered.
#' @param level the confidence level required.
#' @param parallel a logical value to indicate if parallelization should be used.
#' @param nthreads a positive integer to denote the number of threads used with OpenMP, 
#' only used if `parallel = TRUE`.
#' @param ... further arguments passed from other methods.
#' @seealso [plot.BranchGLMCIs], [plotCI]
#' @return An object of class `BranchGLMCIs` which is a list with the following components.
#' \item{`CIs`}{ a numeric matrix with the confidence intervals}
#' \item{`level`}{ the supplied level}
#' \item{`MLE`}{ a numeric vector of the MLEs of the coefficients}
#' @details Endpoints of the confidence intervals that couldn't be found by the algorithm 
#' are filled in with NA. When there is a lot of multicollinearity in the data 
#' the algorithm may have problems finding many of the intervals.
#' @examples 
#' Data <- iris
#' ### Fitting linear regression model
#' mymodel <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ### Getting confidence intervals
#' CIs <- confint(mymodel, level = 0.95)
#' CIs
#' 
#' ### Plotting CIs
#' plot(CIs, mary = 7, cex.y = 0.9)
#' 
#' @export
confint.BranchGLM <- function(object, parm, level = 0.95, 
                              parallel = FALSE, nthreads = 8, ...){
  # Using parm
  if(missing(parm)){
    parm <- 1:ncol(object$x)
  }else if(is.character(parm)){
    parm <- match(parm, colnames(object$x), nomatch = 0L)
    if(length(parm) == 1 && parm == 0L){
      stop("no parameters specified in parm were found")
    }
  }else if(any(parm > ncol(object$x))){
    stop("numbers in parm must be less than or equal to the number of parameters")
  }
  # Checking level
  if(length(level) != 1 || !is.numeric(level) || level >= 1 || level <= 0){
    stop("level must be a number between 0 and 1")
  }
  
  # Checking nthreads and parallel
  if(length(nthreads) != 1 || !is.numeric(nthreads) || is.na(nthreads) || nthreads <= 0){
    stop("nthreads must be a positive integer")
  }
  if(length(parallel) != 1 || !is.logical(parallel) || is.na(parallel)){
    stop("parallel must be either TRUE or FALSE")
  }
  if(!parallel){
    nthreads <- 1
  }
  
  # Getting SEs for make initial values for CIs
  a <- (1 - level) / 2
  coefs <- coef(object)
  SEs <- qnorm(1 - a) * sqrt(diag(object$vcov))
  
  
  
  # Getting LR CIs
  if(object$family == "gaussian" || object$family == "gamma"){
    object$AIC <- object$AIC - 2
  }
  metrics <- rep(object$AIC, ncol(object$x))
  model <- matrix(rep(-1, ncol(object$x)), ncol = 1)
  model[parm] <- 1
  res <- MetricIntervalCpp(object$x, object$y, object$offset, 
                           1:ncol(object$x) - 1, rep(1, ncol(object$x)), model, 
                           object$method, object$grads, object$link, object$family, 
                           nthreads, object$tol, object$maxit, rep(2, ncol(object$x)), 
                           coefs, SEs,
                           metrics, qchisq(level, 1), object$AIC,"ITP")
  
  # Replacing infinities with NA
  res$LowerBounds <- ifelse(is.finite(res$LowerBounds), res$LowerBounds, NA)
  res$UpperBounds <- ifelse(is.finite(res$UpperBounds), res$UpperBounds, NA)
  
  # Getting CIs in right format
  CIs <- cbind(res$LowerBounds, res$UpperBounds)
  rownames(CIs) <- colnames(object$x)
  colnames(CIs) <- c(paste0(round(a, 3) * 100, "%"), paste0(round(1 - a, 3) * 100, "%"))
  return(structure(list("CIs" = CIs[parm, , drop = FALSE], "level" = level, "MLE" = coefs[parm]), 
                   class = "BranchGLMCIs"))
}

#' Print Method for BranchGLMCIs Objects
#' @description Print method for BranchGLMCIs objects.
#' @param x a `BranchGLMCIs` object.
#' @param digits number of significant digits to display.
#' @param ... further arguments passed from other methods.
#' @return The supplied `BranchGLMCIs` object.
#' @export
print.BranchGLMCIs <- function(x, digits = 4, ...){
  print(signif(x$CIs, digits = digits))
  invisible(x)
}


#' Plot Method for BranchGLMCIs Objects
#' @description Creates a plot to visualize confidence intervals from BranchGLMCIs objects.
#' @param x a `BranchGLMCIs` object.
#' @param which which intervals to plot, can use a numeric vector of indices, a 
#' character vector of names of desired variables, or "all" to plot all intervals.
#' @param mary a numeric value used to determine how large to make margin of y-axis. If variable 
#' names are cut-off, consider increasing this from the default value of 5. 
#' @param ... further arguments passed to [plotCI].
#' @seealso [plotCI]
#' @return This only produces a plot, nothing is returned.
#' @examples 
#' Data <- iris
#' ### Fitting linear regression model
#' mymodel <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ### Getting confidence intervals
#' CIs <- confint(mymodel, level = 0.95)
#' CIs
#' 
#' ### Plotting CIs
#' plot(CIs, mary = 7, cex.y = 0.9)
#' 
#' @export
plot.BranchGLMCIs <- function(x, which = "all", mary = 5, ...){
  # Using which
  if(is.character(which) && length(which) == 1 && tolower(which) == "all"){
    which <- 1:length(x$MLE)
  }
  x$CIs <- x$CIs[which, , drop = FALSE]
  x$MLE <- x$MLE[which]
  
  # Getting xlimits
  xlim <- c(min(min(x$CIs, na.rm = TRUE), min(x$MLE, na.rm = TRUE)), 
            max(max(x$CIs, na.rm = TRUE), max(x$MLE, na.rm = TRUE)))
  
  # Setting margins
  oldmar <- par("mar")
  on.exit(par(mar = oldmar))
  par(mar = c(5, mary, 3, 1) + 0.1)
  
  # Plotting CIs
  plotCI(x$CIs, x$MLE, 
         main = paste0(round(x$level * 100, 1), "% Likelihood Ratio CIs"), 
         xlab = "Beta Coefficients", 
         xlim = xlim, ...)
  abline(v = 0, xpd = FALSE)
}

#' Plot Confidence Intervals
#' @description Creates a plot to display confidence intervals.
#' @param CIs a numeric matrix of confidence intervals, must have exactly 2 columns.
#' The variable names displayed in the plot are taken from the column names.
#' @param points points to be plotted in the middle of the CIs, typically means or medians.
#' The default is to plot the midpoints of the intervals.
#' @param ylab a label for the y-axis.
#' @param ylas the style of the y-axis label, see more about this at `las` in [par].
#' @param cex.y font size used for variable names on y-axis.
#' @param decreasing a logical value indicating if confidence intervals should be 
#' displayed in decreasing or increasing order according to points. Can use NA 
#' if no ordering is desired.
#' @param ... further arguments passed to [plot.default].
#' @return This only produces a plot, nothing is returned.
#' @examples 
#' Data <- iris
#' ### Fitting linear regression model
#' mymodel <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' ### Getting confidence intervals
#' CIs <- confint.default(mymodel, level = 0.95)
#' xlim <- c(min(CIs), max(CIs))
#' 
#' ### Plotting CIs
#' par(mar = c(5, 7, 3, 1) + 0.1)
#' plotCI(CIs, main = "95% Confidence Intervals", xlim = xlim, cex.y = 0.9, 
#' xlab = "Beta Coefficients")
#' abline(v = 0)
#' 
#' @export
plotCI <- function(CIs, points = NULL, ylab = "", ylas = 2, cex.y = 1, 
                   decreasing = FALSE, ...){
  # Getting points
  if(is.null(points)){
    points <- apply(CIs, 1, mean)
  }
  
  # Getting CIs in right format
  if(!is.matrix(CIs) || (ncol(CIs) != 2) || !is.numeric(CIs)){
    stop("CIs must be a numeric matrix with exactly 2 columns")
  }else if(nrow(CIs) != length(points)){
    stop("the number of rows in CIs must be the same as the length of points")
  }
  CIs <- t(CIs)
    
  # Getting order of points
  if(!is.na(decreasing)){
    ind <- order(points, decreasing = decreasing)
  }else{
    ind <- 1:length(points)
  }
  
  quants <- CIs[, ind, drop = FALSE]
  points <- points[ind]
  
  # Creating plot
  ## Creating base layer of plot
  plot(points, 1:length(points), ylim = c(0, ncol(quants) + 1), ylab = ylab, 
       yaxt = "n", ...)
  
  ## Creating confidence intervals
  segments(y0 = 1:ncol(quants), x0 = quants[1, ], x1 = quants[2, ])
  segments(y0 = 1:ncol(quants) - 0.25, x0 = quants[1, ], 
           y1 = 1:ncol(quants) + 0.25)
  segments(y0 = 1:ncol(quants) - 0.25, x0 = quants[2, ], 
           y1 = 1:ncol(quants) + 0.25)
  
  ## Adding axis labels for y-axis
  axis(2, at = 1:ncol(quants), labels = colnames(quants), las = ylas, 
       cex.axis = cex.y) 
}
