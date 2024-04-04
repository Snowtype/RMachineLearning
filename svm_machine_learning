if (!requireNamespace("e1071", quietly = TRUE)) install.packages("e1071")
if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
library(e1071)
library(caret)

# DATA
file_path <- "Downloads/credit+approval/crx.data"  # Update the path as necessary
column_names <- c('Gender', 'Age', 'Debt', 'Married', 'BankCustomer', 'EducationLevel',
                  'Ethnicity', 'YearsEmployed', 'PriorDefault', 'Employed', 'CreditScore',
                  'DriversLicense', 'Citizen', 'ZipCode', 'Income', 'ApprovalStatus')

data <- read.csv(file_path, header = FALSE, na.strings = "?", col.names = column_names)

# Preprocessing Handle missing values, factors
data <- na.omit(data)
data$ApprovalStatus <- factor(data$ApprovalStatus, levels = c("+", "-"))
levels(data$ApprovalStatus) <- c("Approved", "Denied")

set.seed(1)
index <- createDataPartition(data$ApprovalStatus, p = 0.7, list = FALSE)
train_data <- data[index,]
test_data <- data[-index,]


# Hyperparameter Tuning for SVM
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                              summaryFunction = twoClassSummary)

# Tuning Grid
tuneGrid <- expand.grid(C = 10^seq(-4, 3, by = 2), sigma = 10^seq(-4, 3, by = 2))

# Train the model
svm_model <- train(ApprovalStatus~., data = train_data, method = "svmRadial",
                   metric = "ROC", trControl = train_control, tuneLength = 5,
                   preProcess = c("center", "scale"), tuneGrid = tuneGrid)
print(svm_model)


# Predictions
train_pred <- predict(svm_model, train_data)
test_pred <- predict(svm_model, test_data)

# Confusion Matrix
confusionMatrix(train_pred, train_data$ApprovalStatus)
confusionMatrix(test_pred, test_data$ApprovalStatus)

