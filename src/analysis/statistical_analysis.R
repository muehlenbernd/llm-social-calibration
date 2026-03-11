# Statistical Analysis for LLMandSEM Project
# This script performs statistical evaluation of the ML model results

# Load required libraries
library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(yaml)

# Load configuration
config <- yaml.load_file("../../config/config.yaml")

#' Load model results from Python
#' @param results_file Name of the results file
load_model_results <- function(results_file) {
  results_path <- file.path(config$output$results_dir, results_file)
  cat("Loading results from", results_path, "\n")
  
  # Assume results are saved as CSV or RDS
  if (grepl("\\.csv$", results_file)) {
    return(read_csv(results_path))
  } else if (grepl("\\.rds$", results_file)) {
    return(readRDS(results_path))
  } else {
    stop("Unsupported file format")
  }
}

#' Perform statistical significance tests
#' @param predictions Model predictions
#' @param true_labels True labels
#' @param baseline_accuracy Baseline accuracy to compare against
statistical_tests <- function(predictions, true_labels, baseline_accuracy = 0.5) {
  # Calculate accuracy
  accuracy <- mean(predictions == true_labels)
  n <- length(predictions)
  
  # Binomial test for significance above baseline
  binom_test <- binom.test(sum(predictions == true_labels), n, baseline_accuracy)
  
  # McNemar's test (if comparing two models)
  # mcnemar_test <- mcnemar.test(table(predictions1, predictions2))
  
  results <- list(
    accuracy = accuracy,
    n_samples = n,
    binomial_test = binom_test,
    ci_lower = binom_test$conf.int[1],
    ci_upper = binom_test$conf.int[2]
  )
  
  return(results)
}

#' Generate statistical plots
#' @param predictions Model predictions
#' @param true_labels True labels
#' @param probabilities Model probabilities (if available)
create_statistical_plots <- function(predictions, true_labels, probabilities = NULL) {
  plots_dir <- config$output$plots_dir
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Confusion matrix heatmap
  cm_df <- table(Predicted = predictions, Actual = true_labels) %>%
    as.data.frame()
  
  p1 <- ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    theme_minimal() +
    labs(title = "Confusion Matrix", x = "True Labels", y = "Predictions")
  
  ggsave(file.path(plots_dir, "confusion_matrix.png"), p1, width = 6, height = 5)
  
  # ROC curve (if probabilities available)
  if (!is.null(probabilities)) {
    roc_obj <- roc(true_labels, probabilities)
    
    png(file.path(plots_dir, "roc_curve.png"), width = 600, height = 500)
    plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc(roc_obj), 3), ")"))
    dev.off()
    
    cat("AUC:", auc(roc_obj), "\n")
  }
  
  # Accuracy confidence intervals plot
  # Add more statistical visualizations as needed
}

#' Generate statistical report
#' @param stat_results Results from statistical tests
#' @param output_file Output file name
generate_report <- function(stat_results, output_file = "statistical_report.txt") {
  report_path <- file.path(config$output$results_dir, output_file)
  
  sink(report_path)
  cat("Statistical Analysis Report\n")
  cat("==========================\n\n")
  cat("Model Accuracy:", stat_results$accuracy, "\n")
  cat("Sample Size:", stat_results$n_samples, "\n")
  cat("95% Confidence Interval: [", stat_results$ci_lower, ",", stat_results$ci_upper, "]\n")
  cat("Binomial Test p-value:", stat_results$binomial_test$p.value, "\n")
  
  if (stat_results$binomial_test$p.value < 0.05) {
    cat("Result: Model performance is significantly above baseline\n")
  } else {
    cat("Result: Model performance is not significantly above baseline\n")
  }
  
  sink()
  
  cat("Statistical report saved to", report_path, "\n")
}

# Example usage (uncomment to run)
# results <- load_model_results("model_results.csv")
# stat_tests <- statistical_tests(results$predictions, results$true_labels)
# create_statistical_plots(results$predictions, results$true_labels)
# generate_report(stat_tests)
