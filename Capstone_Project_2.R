#################
#installing and loading required packages
#################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(readr)
library(randomForest)

#################
#Download Data from the GitHub Repository
#################

urlfile<-'Anonymize_Loan_Default_data.csv'
if(!file.exists(urlfile))
  download.file("https://github.com/eafinnigan/EdX_Capstone_CYO/raw/main/Anonymize_Loan_Default_data.csv", urlfile)

Anonymize_Loan_Default_data <- 
  read_csv(urlfile)

rm(urlfile)
#################
#Beginning Cleaning
#################
#rename data to a shorter name
AnonLoan <- Anonymize_Loan_Default_data

#remove unnecessary data
rm(Anonymize_Loan_Default_data)

#remove a numbering column (shown as ...1 in the data) because it isn't needed 
#and could confuse the algorithm.
#A similar column is already included in the dataset.
AnonLoan <- AnonLoan %>% mutate(...1 = NULL)

#Look at the structure of the data - this will help with our analysis
str(AnonLoan)

#Appears that the first row is junk data, (see "Dec-99" in issue_d column), so we'll remove it
#and look at the structure again
AnonLoan <- AnonLoan[-1,]
str(AnonLoan)

#find any duplicate data
sum(duplicated(AnonLoan)) #no duplicate entries

#find any NA data
sum(is.na(AnonLoan)) #a lot of NAs, will check the dataset to see if they are expected or not.
#The columns that we want to use for our analysis are loan_amnt, annual_inc, and repay_fail.
#we are most interested in the repay_fail column. 1 means that they have failed to repay, 0 means they
#have repaid it.

#check if NAs are in the data columns we want
sum(is.na(AnonLoan$loan_amnt))
sum(is.na(AnonLoan$annual_inc))
sum(is.na(AnonLoan$repay_fail))

#NAs will cause problems when doing analysis. We will remove the rows that have NAs.
AnonLoan <- AnonLoan %>% drop_na(loan_amnt) %>% drop_na(annual_inc)

#there are only 2 possible outcomes with repay_fail, so we'll turn the repay_fail column from a num data type to factor.
AnonLoan <- AnonLoan %>% mutate(repay_fail = as.factor(AnonLoan$repay_fail))

#check if the mutate function worked
str(AnonLoan)

#################
#Beginning Analysis
#################
#purpose of this analysis is to find the likelihood of someone defaulting on their loans given their annual income and loan amount.
#columns of interest are loan_amnt, annual_inc, and repay_fail.

#will use a Random Forest model and kNN model with cross-validation. We use kNN because it estimates conditional probabilities in multiple dimensions, and Random Forest because it models human decision processes while improving prediction performance and reduces instability. We will use cross-validation to find the best value of k.

#test and train sets
#Set 20% of the data to the test set. The dataset is quite large, and any higher of percentage may introduce errors into the results.
#Additionally, predictions will need to be useful on smaller datasets.
set.seed(29)
test_index <- createDataPartition(y = AnonLoan$repay_fail, times = 1, p = 0.2, list = FALSE) # create a 20% test set
test_set <- AnonLoan[test_index,]
train_set <- AnonLoan[-test_index,]

#kNN model
#train the model
train_knn <- train(repay_fail~annual_inc + loan_amnt, method = "knn", data = train_set)
#make a prediction
y_hat_knn <- predict(train_knn, test_set, type = "raw")
#see accuracy of model
confusionMatrix(y_hat_knn, test_set$repay_fail)$overall[["Accuracy"]]

#train model with cross-validation - see if we can get the accuracy up
set.seed(29)
train_knn_test <- train(repay_fail~annual_inc + loan_amnt, method = "knn",
                   data = train_set,
                   tuneGrid = data.frame(k = seq(9, 71, 2)))

#confusion matrix of best accuracy value
confusionMatrix(predict(train_knn_test, test_set, type = "raw"), test_set$repay_fail)$overall["Accuracy"]

#Random Forest model
set.seed(29)
train_rf <- randomForest(repay_fail~annual_inc + loan_amnt, data=train_set)
confusionMatrix(predict(train_rf, test_set), test_set$repay_fail)$overall["Accuracy"]

# use cross validation to find the best level of accuracy
set.seed(29)
train_rf_test <- train(repay_fail~annual_inc + loan_amnt,
                    method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                    data = train_set)
confusionMatrix(predict(train_rf_test, test_set), test_set$repay_fail)$overall["Accuracy"]

#cross-validated Random Forest model takes longer to run than kNN, and is slightly less accurate in their predictions.
