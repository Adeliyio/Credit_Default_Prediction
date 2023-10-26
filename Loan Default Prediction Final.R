#presents a machine learning exercise for predicting the probability that a client will 
#default on loan payments. We will use the credit card database from Brett Lantz’s book, “Machine Learning with R”


# Load libraries
library(caret) 
library(pROC)
library(tidyverse)
library(xgboost)
library(ggplot2)
library(ggcorrplot)
library(gridExtra)
library(knitr)
# Load data
data_source <- "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv"
credit_data <- read.csv(data_source, sep = ",")

# check for missing data
sum(is.na(credit_data))

# Clean data
clean_data <- credit_data %>%
  rename(target = default) %>% 
  mutate(target = factor(target))



# Exploratory Data Analysis
head(clean_data, 10)
str(clean_data)
summary(clean_data)



# Histogram for 'months_loan_duration'
eda_month_duration <- ggplot(clean_data, aes(x = months_loan_duration)) +
  geom_histogram(binwidth = 5, fill = "darkblue", color = "white") +
  labs(title = "Distribution of Loan Duration",
       x = "Loan Duration (Months)",
       y = "Frequency")

# Histogram for 'amount'
eda_amount <- ggplot(clean_data, aes(x = amount)) +
  geom_histogram(binwidth = 500, fill = "orange", color = "white") +
  labs(title = "Distribution of Loan Amount",
       x = "Loan Amount",
       y = "Frequency")

# Histogram for 'age'
eda_age <- ggplot(clean_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "purple", color = "white") +
  labs(title = "Distribution of Age",
       x = "Age",
       y = "Frequency")

# Histogram for 'existing_credits'
eda_existing_credits <- ggplot(clean_data, aes(x = existing_credits)) +
  geom_histogram(binwidth = 0.5, fill = "red", color = "white") +
  labs(title = "Distribution of Existing Credits",
       x = "Existing Credits",
       y = "Frequency")

# Arrange the plots in a grid using grid.arrange()
grid.arrange(eda_month_duration, eda_amount, eda_age, eda_existing_credits, ncol = 2)



# Categorical Variables EDA
eda_categorical1 <- ggplot(clean_data, aes(x = checking_balance)) +
  geom_bar(fill = "lightgreen") +
  labs(title = "Checking Balance Distribution",
       x = "Checking Balance",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

eda_categorical2 <- ggplot(clean_data, aes(x = credit_history)) +
  geom_bar(fill = "pink") +
  labs(title = "Credit History Distribution",
       x = "Credit History",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

eda_categorical3 <- ggplot(clean_data, aes(x = purpose)) +
  geom_bar(fill = "lightblue") +
  labs(title = "Purpose Distribution",
       x = "Purpose",
       y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Arrange the histograms in a grid using plot_grid from gridExtra
eda_categorical <- gridExtra::grid.arrange(eda_categorical1, eda_categorical2, eda_categorical3, ncol = 3)

# Display the grid of histograms
print(eda_categorical)

# Correlation Plot

correlation_matrix <- cor(clean_data[, c("months_loan_duration", "amount", "age", "existing_credits")])
ggcorrplot(correlation_matrix, hc.order = TRUE, type = "lower", lab = TRUE)



summary(clean_data)
head(clean_data)
str(clean_data)

# Split data 
set.seed(1234)
data_index <- createDataPartition(clean_data$target, times = 1, p = 0.2, list = FALSE)
train_set <- clean_data[-data_index,]
test_set <- clean_data[data_index,]

# Baseline model
set.seed(4321)
baseline_pred <- sample(c(1, 2), length(test_set$target), replace = TRUE)
mean(baseline_pred == as.numeric(test_set$target))


# GLM
set.seed(7641)
glm_fit <- train(target ~ ., train_set, method = "glm")
mean(test_set$target == predict(glm_fit, test_set))
glm_matrix <- confusionMatrix(predict(glm_fit, test_set), as.factor(test_set$target))

# KNN
trControl <- trainControl(method="cv", number=20) 
set.seed(1243)
knn_fit <- train(target ~ ., train_set, method="knn", trControl=trControl,
                 metric="Accuracy", tuneGrid = data.frame(k = seq(3, 71, 3)))
mean(test_set$target == predict(knn_fit, test_set))
knn_matrix <- confusionMatrix(predict(knn_fit, test_set), as.factor(test_set$target))

# Random forest
set.seed(2456)
rf_fit <- train(target ~ ., train_set, method = "rf",
                tuneGrid = data.frame(mtry = seq(1:7)), ntree = 100)
mean(test_set$target == predict(rf_fit, test_set)) 
rf_matrix <- confusionMatrix(predict(rf_fit, test_set), as.factor(test_set$target))

# XGBoost
train_xgb_set <- train_set %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate(target = if_else(target == 1, 0, 1))

test_xgb_set <- test_set %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  mutate(target = if_else(target == 1, 0, 1))

xgb_train_matrix <- xgb.DMatrix(as.matrix(train_xgb_set[,!names(train_xgb_set) %in% c("target")]),
                                label = train_xgb_set$target)

xgb_params <- list(objective = "binary:logistic", eval_metric = "auc",
                   max_depth = 11, eta = 0.075, subsample = 0.99, colsample_bytree = 0.90)

set.seed(1767)                   
xgb_fit <- xgb.train(params = xgb_params, data = xgb_train_matrix, verbose = 1, nrounds = 125)

xgb_test <- xgb.DMatrix(as.matrix(test_xgb_set[,!names(test_xgb_set) %in% c("target")]))
xgb_predictions <- ifelse(predict(xgb_fit, newdata = xgb_test) > 0.5,1,0)

mean(test_xgb_set$target == xgb_predictions)
xgb_matrix <- confusionMatrix(as.factor(xgb_predictions), as.factor(test_xgb_set$target))

# ROC curve analysis
glm_roc <- roc(as.numeric(test_set$target), as.numeric(predict(glm_fit, test_set)))
knn_roc <- roc(as.numeric(test_set$target), as.numeric(predict(knn_fit, test_set)))
rf_roc <- roc(as.numeric(test_set$target), as.numeric(predict(rf_fit, test_set)))
xgboost_roc <- roc(as.numeric(test_xgb_set$target), xgb_predictions)

roc_data <- rbind(
  data.frame(Model = "GLM", glm_roc |> coords()),
  data.frame(Model = "kNN", knn_roc |> coords()),
  data.frame(Model = "Random Forest", rf_roc |> coords()),
  data.frame(Model = "XGBoost", xgboost_roc |> coords())  
)

# Enhanced ggplot
ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +  
  labs(title = "ROC Curves", x = "False Positive Rate", y = "True Positive Rate") +
  facet_wrap(~ Model, scales = "free") +
  theme_bw()


# Calculate metrics from confusion matrices
get_metrics <- function(matrix) {
  acc <- sum(diag(matrix)) / sum(matrix)
  sens <- matrix[2, 2] / sum(matrix[2, ])
  spec <- matrix[1, 1] / sum(matrix[1, ])
  return(list(Accuracy = acc, Sensitivity = sens, Specificity = spec))
}

# Calculate metrics from confusion matrices
get_metrics <- function(matrix) {
  acc <- sum(diag(matrix)) / sum(matrix)
  sens <- matrix[2, 2] / sum(matrix[2, ])
  spec <- matrix[1, 1] / sum(matrix[1, ])
  return(list(Accuracy = acc, Sensitivity = sens, Specificity = spec))
}

# Get metrics for all models
baseline_metrics <- get_metrics(confusionMatrix(as.factor(baseline_pred), as.factor(test_set$target))$table)
glm_metrics <- get_metrics(glm_matrix$table)
knn_metrics <- get_metrics(knn_matrix$table)
rf_metrics <- get_metrics(rf_matrix$table)
xgb_metrics <- get_metrics(xgb_matrix$table)

# Create a data frame to store model results
model_results <- data.frame(Model = c("Baseline", "GLM", "kNN", "Random Forest", "XGBoost"),
                            Accuracy = c(baseline_metrics$Accuracy, glm_metrics$Accuracy,
                                         knn_metrics$Accuracy, rf_metrics$Accuracy,
                                         xgb_metrics$Accuracy),
                            Sensitivity = c(baseline_metrics$Sensitivity, glm_metrics$Sensitivity,
                                            knn_metrics$Sensitivity, rf_metrics$Sensitivity,
                                            xgb_metrics$Sensitivity),
                            Specificity = c(baseline_metrics$Specificity, glm_metrics$Specificity,
                                            knn_metrics$Specificity, rf_metrics$Specificity,
                                            xgb_metrics$Specificity))


# Display model results using kable
kable(model_results, caption = "Model Evaluation Results")

