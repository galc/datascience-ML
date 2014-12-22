---
title: "Prediction Model based on Accelerometer Data"
author: "GalC"
date: "Sunday, December 21, 2014"
output: pdf_document
---

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

The goal of this assignment is to use data from accelerometers placed on the belt, forearm, arm, and dumbell of 6 participants to predict how well they were doing. 

### Libraries
These are the libraries that were used in this model:

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

## Data Loading

Load training and testing datasets:

```r
training.file <- './data/pml-training.csv'
testing.file  <- './data/pml-testing.csv'
training.url  <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testing.url   <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

if (! file.exists(training.file)) {
    download.file(training.url, destfile = training.file) }
if (! file.exists(testing.file)) {
    download.file(testing.url, destfile = testing.file)   }

raw_training <- read.csv(training.file)
raw_testing  <- read.csv(testing.file)

set.seed(1357)
```

Split the data into training and cross validation sets:

```r
trainingSet <- createDataPartition(raw_training$classe, list=FALSE, p=.9)
training = raw_training[trainingSet,]
testing = raw_training[-trainingSet,]
```

Locate and remove near zero variance:

```r
nzv <- nearZeroVar(training)
training <- training[-nzv]
testing <- testing[-nzv]
raw_testing <- raw_testing[-nzv]
```


Include only numeric features to avoid misclassifications:

```r
num_features_idx = which(lapply(training,class) %in% c('numeric')  )
```

Impute missing values in the training data:


```r
preModel  <- preProcess(training[,num_features_idx], method=c('knnImpute'))
ptraining <- cbind(training$classe, predict(preModel, training[,num_features_idx]))
ptesting  <- cbind(testing$classe, predict(preModel, testing[,num_features_idx]))
prtesting <- predict(preModel, raw_testing[,num_features_idx])

#Fix Label on classe
names(ptraining)[1] <- 'classe'
names(ptesting)[1] <- 'classe'
```

## The Model

The random forest model that is built here is based on the provided numerical variables which as can be seen provides a good prediction accuracy to the 20 test cases. Due to computation impact, the optimized values for the random forest are used and shown below:


```r
rf_model  <- randomForest(classe ~ ., ptraining, ntree=500, mtry=32)
```

## Cross Validation

We are able to measure the accuracy using our training set and our cross validation set. With the training set we can detect if our model has bias due to ridgity of our mode. With the cross validation set, we are able to determine if we have variance due to overfitting.

### In-sample accuracy

```r
training_pred <- predict(rf_model, ptraining) 
print(confusionMatrix(training_pred, ptraining$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5022    0    0    0    0
##          B    0 3418    0    0    0
##          C    0    0 3080    0    0
##          D    0    0    0 2895    0
##          E    0    0    0    0 3247
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
The in sample accuracy shows that the model is not biased.

### Out-of-sample accuracy

```r
testing_pred <- predict(rf_model, ptesting) 
```

Confusion Matrix: 

```r
print(confusionMatrix(testing_pred, ptesting$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 555   4   0   0   0
##          B   1 372   1   0   0
##          C   0   2 340   3   0
##          D   2   1   1 316   1
##          E   0   0   0   2 359
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9908          
##                  95% CI : (0.9855, 0.9945)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9884          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9946   0.9815   0.9942   0.9844   0.9972
## Specificity            0.9971   0.9987   0.9969   0.9969   0.9988
## Pos Pred Value         0.9928   0.9947   0.9855   0.9844   0.9945
## Neg Pred Value         0.9979   0.9956   0.9988   0.9969   0.9994
## Prevalence             0.2847   0.1934   0.1745   0.1638   0.1837
## Detection Rate         0.2832   0.1898   0.1735   0.1612   0.1832
## Detection Prevalence   0.2852   0.1908   0.1760   0.1638   0.1842
## Balanced Accuracy      0.9959   0.9901   0.9955   0.9907   0.9980
```

Cross validation accuracy > 99%, hence it can be used for predicting the 20 test observations. 

## Test Set Prediction Results

Applying this model to the test data provided yields 100% classification accuracy on the twenty test observations.

```r
answers <- predict(rf_model, prtesting) 
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

## Conclusion
The model provided us with a very good prediction to the weight lifting style as measured with accelerometers.

## References

[caret]: Max Kuhn. Contributions from Jed Wing, Steve Weston, Andre Williams, Chris Keefer, Allan Engelhardt, Tony Cooper, Zachary Mayer and the R Core Team. Caret package.
