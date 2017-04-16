# PREDICTION ASSIGNMENT: QUANTIFYING PERFORMANCE OF EXERCISE
LAM CHOON HO  
15 APR 2017  


## TITLE: PREDICTION ASSIGNMENT: QUANTIFYING PERFORMANCE OF EXERCISE
AUTHOR: LAM CHOON HO
Date: 15 APR 2017

#### Executive Summary
The objective of this project is to quantifying performance of exercse with using prediction model on 20 different test cases. This project is used Random Forest machine learning technique 
for prediction task.

#### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#### Citation
http://groupware.les.inf.puc-rio.br/har

#### 1. Loading necessary library
Loading necessary library for data reading and applying Decision Tree & Random Forest machine learning technique

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.3.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.3.3
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.3.3
```

#### 2. Retrieve Data
Getting the source of training data and testing data.

```r
pmlTrainingDataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
pmlTestingDataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

#### 3. Preprocessing Data
Change empty value and invalid value to NA

```r
pmlTrainingData <- read.csv(url(pmlTrainingDataURL), na.strings=c("NA","#DIV/0!",""))
pmlTestingData <- read.csv(url(pmlTestingDataURL), na.strings=c("NA","#DIV/0!",""))
```

#### 4. Data Cleaning
Remove Near-Zero-Value (NZV), NA, Empty Value and first 6 columes

```r
NearZeroVal <- nzv(pmlTrainingData, saveMetrics = T)
CleanTrainingData <- pmlTrainingData[, names(pmlTrainingData)[!(NearZeroVal[, 4])]]
NAEmptyVal <- sapply(CleanTrainingData, function(x) !(any(is.na(x) | x == "")))
CleanTrainingData <- CleanTrainingData[, names(CleanTrainingData)[NAEmptyVal]]
CleanTrainingData <- CleanTrainingData[,-(1:6)]
```

#### 5. Segmented Data into Training Data and Test Data (Cross Validation Method)
#### Cross Validation Method
Segregated to 70% training dataset and 30% testing dataset of cleaned training dataset

```r
TrainDataPart <- createDataPartition(y = CleanTrainingData$classe, p = 0.7, list = FALSE)
trainingDataSeg <- CleanTrainingData[TrainDataPart,]
testingDataSeg <- CleanTrainingData[-TrainDataPart,]
```

#### 6. Constructing Prediction Model with using Random Forest
Applying and train Random Forest machine learning technique to 70% training dataset

```r
PreModel <- randomForest(trainingDataSeg$classe ~. , data = trainingDataSeg)
```

#### 7. Prediction Model, Accuracy Rate and Out of Sample Error
According to the result, using Random Forest technique can give 99% accuracy rate. Thus,
this method is used to apply on the 20 different test cases.

***Definition of Out of Sample Error***

It is statistics speak which in most cases means "using past data to make forecasts of the future". "In sample" refers to the data that you have, and "out of sample" to the data you don't have but want to forecast or estimate.

**Decision Tree**

```r
DTTrainingMod <- rpart(classe ~ ., data=trainingDataSeg, method="class")
DTPredict <- predict(DTTrainingMod, testingDataSeg, type = "class")
rpart.plot(DTTrainingMod, main="Classification Tree", extra=102, under=TRUE, faclen=0)
```

![](PredictionAssignment_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
print(confusionMatrix(DTPredict, testingDataSeg$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1475  178   11   56   23
##          B   58  693   72   84   89
##          C   42  140  876  154  140
##          D   65   80   56  602   54
##          E   34   48   11   68  776
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7514          
##                  95% CI : (0.7402, 0.7624)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6852          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8811   0.6084   0.8538   0.6245   0.7172
## Specificity            0.9364   0.9362   0.9020   0.9482   0.9665
## Pos Pred Value         0.8462   0.6958   0.6479   0.7025   0.8282
## Neg Pred Value         0.9520   0.9088   0.9669   0.9280   0.9382
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2506   0.1178   0.1489   0.1023   0.1319
## Detection Prevalence   0.2962   0.1692   0.2297   0.1456   0.1592
## Balanced Accuracy      0.9087   0.7723   0.8779   0.7863   0.8418
```

#### Random Forest

```r
ModelPrediction <- predict(PreModel, newdata = testingDataSeg)
print(confusionMatrix(ModelPrediction, testingDataSeg$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    4    0    0    0
##          B    2 1130   12    0    0
##          C    0    5 1014   14    1
##          D    0    0    0  950    2
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9921   0.9883   0.9855   0.9972
## Specificity            0.9991   0.9971   0.9959   0.9996   1.0000
## Pos Pred Value         0.9976   0.9878   0.9807   0.9979   1.0000
## Neg Pred Value         0.9995   0.9981   0.9975   0.9972   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1920   0.1723   0.1614   0.1833
## Detection Prevalence   0.2848   0.1944   0.1757   0.1618   0.1833
## Balanced Accuracy      0.9989   0.9946   0.9921   0.9925   0.9986
```
#### Out of Sample Error
Out of Sample Error (1 - Accuracy Rate) is shown 0.007 or 0.7%

**Conclusion:**

According to the result, using Random Forest technique can give 99.3% accuracy rate which is better than Decision Tree where the accuracy rate of Decision Tree is 68.5%. However, the Out Sample of Error (1 - Accuracy Rate) is shown 0.007 or 0.7%. Thus, Random forest is chooses.

#### 8. Applying Prediction Model to 20 cases
Below show the result of 20 different test cases

```r
print(predict(PreModel, newdata = pmlTestingData))
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
