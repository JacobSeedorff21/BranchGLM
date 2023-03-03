#' Likelihood Ratio Confidence Intervals for Beta Coefficients
#' @param object a \code{BranchGLM} object.
#' @param level desired confidence level.
#' @param parallel whether or not to make use of parallelization via OpenMP.
#' @param nthreads number of threads used with OpenMP, only used if \code{parallel = TRUE}.
#' @param ... further arguments passed from other methods.
#' @return An object of class \code{BranchGLMCIs} which is a list with the following components.
#' \item{\code{CIs}}{ a matrix with the confidence intervals}
#' \item{\code{level}}{ the supplied level}
#' \item{\code{MLE}}{ a numeric vector of the MLEs of the coefficients}
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
confint.BranchGLM <- function(object, level = 0.95, 
                              parallel = FALSE, nthreads = 8, ...){
  # Getting SEs for make initial values for CIs
  a <- (1 - level) / 2
  coefs <- coef(object)
  SEs <- qnorm(1 - a) * sqrt(diag(object$vcov))
  
  # Checking parallel
  if(!parallel){
    nthreads <- 1
  }
  
  # Getting LR CIs
  metrics <- rep(object$AIC, ncol(object$x))
  model <- matrix(rep(1, ncol(object$x)), ncol = 1)
  res <- MetricIntervalCpp(object$x, object$y, object$offset, 
                           1:ncol(object$x) - 1, rep(1, ncol(object$x)), model, 
                           object$method, object$grads, object$link, object$family, 
                           nthreads, object$tol, object$maxit, "AIC", coefs, SEs,
                           metrics, object$AIC + qchisq(level, 1), 
                           "ITP")
  
  # Replacing infinities with NA
  res$LowerBounds <- ifelse(is.finite(res$LowerBounds), res$LowerBounds, NA)
  res$UpperBounds <- ifelse(is.finite(res$UpperBounds), res$UpperBounds, NA)
  
  # Getting CIs in right format
  CIs <- cbind(res$LowerBounds, res$UpperBounds)
  rownames(CIs) <- colnames(object$x)
  colnames(CIs) <- c(paste0(round(a, 3) * 100, "%"), paste0(round(1 - a, 3) * 100, "%"))
  return(structure(list("CIs" = CIs, "level" = level, "MLE" = coefs), 
                   class = "BranchGLMCIs"))
}

#' Print Method for BranchGLMCIs Objects
#' @param object a \code{BranchGLMCIs} object.
#' @param digits number of significant digits to display.
#' @param ... further arguments passed from other methods.
#' @return The supplied \code{BranchGLMCIs} object.
#' @export
print.BranchGLMCIs <- function(object, digits = 4, ...){
  print(signif(object$CIs, digits = digits))
  invisible(object)
}


#' Plot Method for BranchGLMCIs Objects
#' @param x a \code{BranchGLMCIs} object.
#' @param which which intervals to plot, can use indices or names of desired variables.
#' @param mary value used to determine how large to make margin of y-axis. If variable 
#' names are cut-off, consider increasing this from the default value of 5. 
#' @param ... further arguments passed to plotCI function.
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
  if(is.character(which) && length(which) == 1 && which == "all"){
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
#' @param CIs a matrix of confidence intervals, must have either 2 rows or 2 columns.
#' The variable names displayed in the plot are taken from either the column names 
#' or row names of this.
#' @param points points to be plotted in the middle of the CIs, typically means or medians.
#' The default is to plot the midpoint of the intervals.
#' @param ylab axis label for y-axis.
#' @param las the style of the y-axis label, the default is horizontal, 
#' see more about this at \link{par}.
#' @param cex.y font size used for variable names on y-axis.
#' @param decreasing a logical value indicating if confidence intervals should be 
#' displayed in decreasing or increasing order according to points.
#' @param ... further arguments passed to default plot method.
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
plotCI <- function(CIs, points = NULL, ylab = "", las = 2, cex.y = 1, 
                   decreasing = FALSE, ...){
  # Getting CIs in right format
  if(!is.matrix(CIs) ||(ncol(CIs) > 2 && nrow(CIs) > 2)){
    stop("CIs must be a matrix with either 2 rows or 2 columns")
  }else if(ncol(CIs) == 2){
    CIs <- t(CIs)
  }

  # Getting points
  if(is.null(points)){
    points <- apply(CIs, 2, mean)
  }
    
  # Getting order of points
  ind <- order(points, decreasing = decreasing)
  if(!is.matrix(CIs)){
    CIs <- matrix(CIs, ncol = 1, dimnames = list(NULL, names(CIs)))
  }
  quants <- CIs[, ind, drop = FALSE]
  points <- points[ind]
  
  # Creating plot
  ## Creating base layer of plot
  plot(points, 1:length(points), 
       ylim = c(0, ncol(quants) + 1),
       ylab = ylab, 
       yaxt = "n", 
       ...)
  
  ## Creating confidence intervals
  segments(y0 = 1:ncol(quants), x0 = quants[1, ], x1 = points)
  segments(y0 = 1:ncol(quants), x0 = points, x1 = quants[2, ])
  segments(y0 = 1:ncol(quants) - 0.25, x0 = quants[1, ], y1 = 1:ncol(quants) + 0.25)
  segments(y0 = 1:ncol(quants) - 0.25, x0 = quants[2, ], y1 = 1:ncol(quants) + 0.25)
  
  ## Adding axis labels for y-axis
  axis(2, at = 1:ncol(quants), labels = colnames(quants), las = las, 
       cex.axis = cex.y) 
}
