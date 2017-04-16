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
Loading necessary library for data reading and applying Random Forest machine learning technique

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

#### 7. Prediction Model Assessment
According to the result, using Random Forest technique can give 99% accuracy rate. Thus,
this method is used to apply on the 20 different test cases.

```r
ModelPrediction <- predict(PreModel, newdata = testingDataSeg)
print(confusionMatrix(ModelPrediction, testingDataSeg$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    6    0    0    0
##          B    1 1132    7    0    0
##          C    0    1 1019   12    0
##          D    0    0    0  952    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9939   0.9932   0.9876   0.9972
## Specificity            0.9986   0.9983   0.9973   0.9994   1.0000
## Pos Pred Value         0.9964   0.9930   0.9874   0.9969   1.0000
## Neg Pred Value         0.9998   0.9985   0.9986   0.9976   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1924   0.1732   0.1618   0.1833
## Detection Prevalence   0.2853   0.1937   0.1754   0.1623   0.1833
## Balanced Accuracy      0.9990   0.9961   0.9953   0.9935   0.9986
```

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
