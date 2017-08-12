library(caret)

set.seed(1)
inTrain <- createDataPartition(mutagen, p=0.75, list = FALSE)
trainDescr <- descr[inTrain,]
testDescr <- descr[-inTrain,]
trainClass <- mutagen[inTrain]
testClass <- mutagen[-inTrain]
rm(inTrain)

prop.table(table(mutagen))
prop.table(table(trainClass))
#Remove zero variance variables
nearZeroVar <- nearZeroVar(trainDescr)
trainDescr <- trainDescr[, -nearZeroVar]
testDescr  <-  testDescr[, -nearZeroVar]
#For multicollinearity
descrCorr <- cor(trainDescr)
highCorr <- findCorrelation(descrCorr, 0.90)
trainDescr <- trainDescr[, -highCorr]
testDescr  <-  testDescr[, -highCorr]
#Transform Variables- Center and Scale
xTrans <- preProcess(trainDescr)
trainDescr <- predict(xTrans, trainDescr)
testDescr  <- predict(xTrans,  testDescr)
#Model - Support Vector Machine
bootControl <- trainControl(number = 200)
set.seed(2)
svmFit <- train(trainDescr, trainClass, method = "svmRadial", tuneLength = 5, 
                trControl = bootControl, scaled = FALSE)
svmFit$finalModel
#Model - Gradient Boost Machine
gbmGrid <- expand.grid(.interaction.depth = (1:5) * 2, .n.trees = (1:10)*25, 
                       .shrinkage = .1, .n.minobsinnode = c(10))
set.seed(2)
gbmFit <- train(trainDescr, trainClass, method = "gbm", trControl = bootControl, 
                verbose = FALSE, bag.fraction = 0.5, tuneGrid = gbmGrid)
plot(gbmFit)
plot(gbmFit, metric = "Kappa")
plot(gbmFit, plotType = "level")
resampleHist(gbmFit)
#Predictions
predict(svmFit$finalModel, newdata = testDescr)[1:5]
predict(svmFit, newdata = testDescr)[1:5]

models <- list(svm = svmFit, gbm = gbmFit)
testPred <- predict(models, newdata = testDescr)
lapply(testPred, function(x) x[1:5])

predValues <- extractPrediction(models, testX = testDescr, testY = testClass)
testValues <- subset(predValues, dataType == "Test")
head(testValues)
table(testValues$model)

#probValues <- extractProb(models, testX = testDescr, testY = testClass)
#testProbs <- subset(probValues, dataType == "Test")
#plotClassProbs(testProbs)

#Performance
svmPred <- subset(testValues, model == "svmRadial")
confusionMatrix(svmPred$pred, svmPred$obs)

#Variable Importance
gbmImp <- varImp(gbmFit, scale = FALSE)
gbmImp
plot(varImp(gbmFit), top = 20)
