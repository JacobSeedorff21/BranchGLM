#' Summary Method for BranchGLMVS Objects
#' @description Summary method for BranchGLMVS objects.
#' @param object a `BranchGLMVS` object.
#' @param ... further arguments passed to or from other methods.
#' @seealso [plot.summary.BranchGLMVS], [coef.summary.BranchGLMVS], [predict.summary.BranchGLMVS]
#' @return An object of class `summary.BranchGLMVS` which is a list with the 
#' following components
#' \item{`results`}{ a data.frame which has the metric values for the best models along 
#' with the sets of variables included in each model}
#' \item{`VS`}{ the supplied `BranchGLMVS` object}
#' \item{`formulas`}{ a list containing the formulas of the best models}
#' \item{`metric`}{ the metric used to perform variable selection}
#' @examples
#' 
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' VS
#' 
#' ## Getting summary of the process
#' Summ <- summary(VS)
#' Summ
#' 
#' ## Plotting the BIC of the best models
#' plot(Summ, type = "b")
#' 
#' ## Plotting the variables in the best models
#' plot(Summ, ptype = "variables")
#' 
#' ## Getting coefficients
#' coef(Summ)
#'  
#' @export

summary.BranchGLMVS <- function(object, ...){
  
  # Getting whether each variables is included in each model
  BestModels <- t(object$bestmodels)
  BestModels[BestModels == -1] <- "kept"
  BestModels[BestModels == 0] <- "no"
  BestModels[BestModels == 1] <- "yes"
    
  # Creating data frame with results
  df <- data.frame(BestModels, object$bestmetrics)
  colnames(df) <- c(object$names, object$metric)
  
  # Creating formulas for each model
  Models <- object$bestmodels
  if(!is.matrix(Models)){
    # if Models is a vector, then change it to a matrix
    Models <- matrix(Models, ncol = 1)
  }
  
  # Generating formulas for each of the best models
  formulas <- apply(Models, 2, FUN = function(x){
    tempnames <- object$names[x != 0]
    tempnames <- tempnames[which(tempnames != "(Intercept)")]
    if(length(tempnames) > 0){
      MyFormula <- as.formula(paste0(object$initmodel$yname, " ~ ", paste0(tempnames, collapse = "+")))
      if(!("(Intercept)" %in% object$names[x != 0])){
        MyFormula <- deparse1(MyFormula) |>
          paste0(" - 1") |>
          as.formula()
      }
    }else{
      # We can do this since we only include non-null models in bestmodels
      MyFormula <- formula(paste0(object$initmodel$yname, " ~ 1")) 
    }
    MyFormula
  }
  )
  MyList <- list("results" = df, 
                 "VS" = object,
                 "formulas" = formulas, 
                 "metric" = object$metric)
  
  return(structure(MyList, class = "summary.BranchGLMVS"))
}

#' @rdname coef.BranchGLMVS
#' @export
coef.summary.BranchGLMVS <- function(object, which = 1, ...){
  coef(object$VS, which = which)
}

#' @rdname predict.BranchGLMVS
#' @export
predict.summary.BranchGLMVS <- function(object, which = 1, ...){
  predict(object$VS, which = which, ...)
}

#' Print Method for summary.BranchGLMVS Objects
#' @description Print method for summary.BranchGLMVS objects.
#' @param x a `summary.BranchGLMVS` object.
#' @param digits number of digits to display.
#' @param ... further arguments passed to other methods.
#' @return The supplied `summary.BranchGLMVS` object.
#' @export

print.summary.BranchGLMVS <- function(x, digits = 2, ...){
  temp <- x$results
  temp[, ncol(temp)] <- round(temp[ncol(temp)], digits = digits)
  print(temp)
  return(invisible(x))
}

#' Plot Method for summary.BranchGLMVS and BranchGLMVS Objects
#' @description Creates plots to help visualize variable selection results from 
#' BranchGLMVS or summary.BranchGLMVS objects.
#' @param x a `summary.BranchGLMVS` or `BranchGLMVS` object.
#' @param ptype the type of plot to produce, look at details for more explanation.
#' @param marnames a numeric value used to determine how large to make margin of axis with variable 
#' names, this is only for the "variables" plot. If variable names are cut-off, 
#' consider increasing this from the default value of 7.
#' @param addLines a logical value to indicate whether or not to add black lines to 
#' separate the models for the "variables" plot. This is typically useful for smaller 
#' amounts of models, but can be annoying if there are many models.
#' @param type what type of plot to draw for the "metrics" plot, see more details at [plot.default]. 
#' @param horiz a logical value to indicate whether models should be displayed horizontally for the "variables" plot.
#' @param cex.names how big to make variable names in the "variables" plot.
#' @param cex.lab how big to make axis labels.
#' @param cex.axis how big to make axis annotation.
#' @param cex.legend how big to make legend labels.
#' @param cols the colors used to create the "variables" plot. Should be a character 
#' vector of length 3, the first color will be used for included variables, 
#' the second color will be used for excluded variables, and the third color will 
#' be used for kept variables.
#' @param ... further arguments passed to [plot.default] for the "metrics" plot 
#' and [image.default] for the "variables" plot.
#' @details The different values for ptype are as follows
#' \itemize{
#'  \item "metrics" for a plot that displays the metric values ordered by rank for the branch and bound algorithms 
#'  or a plot which displays the metric values in the path taken by the stepwise algorithms
#'  \item "variables" for a plot that displays which variables are in each of the top models for the branch and bound algorithms 
#'  or a plot which displays the path taken by the stepwise algorithms
#'  \item "both" for both plots
#' }
#' 
#' If there are so many models that the "variables" plot appears to be 
#' entirely black, then set addLines to FALSE.
#' 
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", bestmodels = 10, 
#' showprogress = FALSE)
#' VS
#' 
#' ## Getting summary of the process
#' Summ <- summary(VS)
#' Summ
#' 
#' ## Plotting the BIC of best models
#' plot(Summ, type = "b", ptype = "metrics")
#' 
#' ## Plotting the BIC of the best models 
#' plot(Summ, ptype = "variables")
#' 
#' ### Alternative colors
#' plot(Summ, ptype = "variables", 
#' cols = c("yellowgreen", "purple1", "grey50"))
#' 
#' ### Smaller text size for names
#' plot(Summ, ptype = "variables", cex.names = 0.75)
#' 
#' @return This only produces plots, nothing is returned.
#' @export

plot.summary.BranchGLMVS <- function(x, ptype = "both", marnames = 7, addLines = TRUE, 
                                     type = "b", horiz = FALSE,
                                     cex.names = 1, cex.lab = 1, 
                                     cex.axis = 1, cex.legend = 1,
                                     cols = c("deepskyblue", "indianred", "forestgreen"), 
                                     ...){
  # Converting VS types to title case for plotting
  s <- strsplit(x$VS$type, " ")[[1]]
  x$VS$type <- paste(toupper(substring(s, 1,1)), substring(s, 2), sep="", collapse=" ")
  
  # Converting ptype to lower
  ptype <- tolower(ptype)
  if(length(ptype) != 1 || !is.character(ptype)){
    stop("ptype must be one of 'metrics', 'variables', or 'both'")
  }else if(!ptype %in% c("metrics", "both", "variables")){
    stop("ptype must be one of 'metrics', 'variables', or 'both'")
  }
  if(ptype %in% c("metrics", "both")){
    if(x$VS$optType == "exact"){
      main <- paste0("Best Models Ranked by ", x$metric)
      xlab <- "Rank"
      x1 <- 1:nrow(x$results)
    }else{
      main <- paste0(x$VS$type, " Path with ", x$metric)
      xlab <- paste0(x$VS$type, " Step")
      x1 <- (nrow(x$results) - 1):0
    }
    plot(1:nrow(x$results), x$results[, ncol(x$results)], 
         xlab = xlab, ylab = x$metric, xaxt = "n", 
         main = main, type = type, cex.lab = cex.lab, cex.axis = cex.axis,
         ...)
    axis(1, at = 1:nrow(x$results), labels = x1, line = 1, las = 1, cex.axis = cex.names)
  }
  # Checking cols
  if(length(cols) != 3 || !is.character(cols)){
    stop("cols must be a character vector of length 3")
  }
  if(ptype %in% c("variables", "both") && !horiz){
    # This is inspired by the plot.regsubsets function
    n <- length(x$formulas)
    Names <- colnames(x$results)[-(ncol(x$results))]
    z <- x$results[, -(ncol(x$results))]
    z[z == "kept"] <- 2
    z[z == "no"] <- 1
    z[z == "yes"] <- 0
    z <- apply(z, 2, as.numeric)
    if(!is.matrix(z)){
      z <- matrix(z, ncol = length(z))
    }
    y <- 1:ncol(z)
    x1 <- 1:nrow(z)
    
    if(x$VS$optType == "exact"){
      main <- paste0("Best Models Ranked by ", x$metric)
      ylab <- paste0("Rank According to ", x$metric)
      xnum <- 1:nrow(z)
    }else{
      main <- paste0(x$VS$type, " Path with ", x$metric)
      ylab <- paste0(x$VS$type, " Step")
      xnum <- (nrow(z) - 1):0
    }
    
    # Creating image  
    oldmar <- par("mar")
    on.exit(par(mar = oldmar))
    par(mar = c(5, marnames, 3, 6) + 0.1)
    
    if(all(z != 2)){
      # Do this if there were no variable kept
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = main, 
            col = cols[-3], ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded"), 
             fill = cols[-3], 
             xpd = TRUE, cex = cex.legend)
    }else{
      # Do this if there were any kept variables
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = main, 
            col = cols, ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded", "Kept"), 
             fill = cols, 
             xpd = TRUE, cex = cex.legend)
    }
    
    # Adding lines
    if(addLines){
      abline(h = y + 0.5, v = x1 - 0.5)
    }else{
      abline(h = y + 0.5)
    }
    
    # Adding axis labels
    axis(1, at = x1, labels = xnum, line = 1, las = 1, cex.axis = cex.axis)
    axis(2, at = y, labels = Names, line = 1, las = 2, cex.axis = cex.names)
    
    # Adding y-axis title, this is used to avoid overlapping of axis title and labels
    mtext(ylab, side = 1, line = 4, cex = cex.lab)
    
  }else if(ptype %in% c("variables", "both") && horiz){
    # This is inspired by the plot.regsubsets function
    n <- length(x$formulas)
    Names <- colnames(x$results)[-(ncol(x$results))]
    z <- x$results[, -(ncol(x$results))]
    z[z == "kept"] <- 2
    z[z == "no"] <- 1
    z[z == "yes"] <- 0
    z <- apply(z, 2, as.numeric)
    if(is.matrix(z)){
      z <- t(z)
    }else{
      z <- matrix(z, nrow = length(z))
    }
    y <- 1:ncol(z)
    x1 <- 1:nrow(z)
    
    if(x$VS$optType == "exact"){
      main <- paste0("Best Models Ranked by ", x$metric)
      ylab <- paste0("Rank According to ", x$metric)
      ynum <- 1:ncol(z)
    }else{
      main <- paste0(x$VS$type, " Path with ", x$metric)
      ylab <- paste0(x$VS$type, " Step")
      ynum <- (ncol(z) - 1):0
    }
    
    # Creating image  
    oldmar <- par("mar")
    on.exit(par(mar = oldmar))
    par(mar = c(marnames, 5, 3, 6) + 0.1)
    
    if(all(z != 2)){
      # Do this if there were no variable kept
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = main, 
            col = cols[-3], ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded"), 
             fill = cols[-3], 
             xpd = TRUE, cex = cex.legend)
    }else{
      # Do this if there were any kept variables
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = main, 
            col = cols, ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded", "Kept"), 
             fill = cols,
             xpd = TRUE, cex = cex.legend)
    }
    
    # Adding lines
    if(addLines){
      abline(v = x1 - 0.5, h = y + 0.5)
    }else{
      abline(v = x1 - 0.5)
    }
    
    # Adding axis labels
    axis(1, at = x1, labels = Names, line = 1, las = 2, cex.axis = cex.names)
    axis(2, at = y, labels = ynum, line = 1, las = 2, cex.axis = cex.axis)
    
    # Adding y-axis title, this is used to avoid overlapping of axis title and labels
    mtext(ylab, side = 2, line = 4, cex = cex.lab)
    
  }
}
