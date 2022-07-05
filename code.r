library(class)
library(gmodels)
library(e1071)
library(randomForest)
library(dplyr)
library(ROCR)
set.seed(123)

par(ask=TRUE)
loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
ds  = "breast-cancer-wisconsin/wdbc.data"
url = paste(loc, ds, sep="")

breast = read.table(url, sep = ",", header=FALSE, na.strings = "?")
names(breast) = c("ID", "diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")

table(breast$diagnosis)
breast=breast[-1]
breast$diagnosis = recode(breast$diagnosis, B = 0, M = 1)

barplot(table(breast$diagnosis), col=c("darkblue","red"), xlab = "0-Benign, 1-Malignant", main = "Teşhis Bar Chart", legend = rownames(table(breast$diagnosis)), beside=TRUE)

sample <- sample.int(n = nrow(breast), size = floor(.80*nrow(breast)), replace = F)
bc_train <- breast[sample, ]
bc_test  <- breast[-sample, ]

bc_train_labels = bc_train$diagnosis
bc_test_labels = bc_test$diagnosis

# KNN 
normalize = function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

bc_n = as.data.frame(lapply(breast[2:31], normalize))
summary(bc_n$area_mean)

knn_pred = knn(train = bc_train, test = bc_test, cl = bc_train_labels, k = 10)

# NAIVE BAYES

bc_classifier = naiveBayes(bc_train, bc_train_labels)
naive_pred = predict(bc_classifier,bc_test)

df.train = bc_train
df.validate = bc_test

#RANDOM FOREST

rf = randomForest(as.factor(diagnosis)~., data=df.train, proximity=TRUE, ntree = 15, mtry = 10) 
p1 <- predict(rf, df.validate)

#SUPPORT VECTOR MACHINE

svmfit = svm(as.factor(diagnosis) ~ ., data = df.train, kernel = "linear", scale = TRUE)
svm_pred = predict(svmfit, df.validate)


print("KNN Cross Table")
CrossTable(x = bc_test_labels, y = knn_pred, prob.chisq = FALSE)

print("Naive Bayes Cross Table 1")
CrossTable(naive_pred, bc_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

print("Random Forest Cross Table")
CrossTable(x = df.validate$diagnosis, y = p1, prob.chisq = FALSE,  dnn = c('actual', 'predicted'))

print("Support Vector Machine Cross Table")
CrossTable(x = df.validate$diagnosis, y = svm_pred, prob.chisq = FALSE,  dnn = c('actual', 'predicted'))

svmfit

table(breast$diagnosis)
summary(breast[-1])
head(breast[-1])

par(mfrow = c(2,5))

df = breast[-1]
for (i in 21:30) {
   boxplot(df[i], xlab = names(df)[i])
}

library(ggplot2)
library(GGally)

ggpairs(df[11:15])
summary(df[1:10])

#LOGISTIC REGRESSION

par(ask=TRUE)
loc = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
ds  = "breast-cancer-wisconsin/wdbc.data"
url = paste(loc, ds, sep="")

breast = read.table(url, sep = ",", header=FALSE, na.strings = "?")
names(breast) = c("ID", "diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
breast$diagnosis = recode(breast$diagnosis, B = 0, M = 1)
breast[,3:31] = scale(breast[,3:31])
df = breast[-1]

y = breast$diagnosis
x = df[-breast$diagnosis]

library(MASS)
model = lm(breast$diagnosis~., data = df)
step_back <- stepAIC(model, direction="backward")
sample = sample.int(n = nrow(breast), size = floor(.80*nrow(breast)), replace = F)
general_result_train = data.frame(breast$radius_mean[sample], breast$compactness_mean[sample], breast$concavity_mean[sample], breast$concave_points_mean[sample], breast$radius_se[sample], breast$smoothness_se[sample], breast$concavity_se[sample], breast$concave_points_se[sample], breast$radius_worst[sample], breast$texture_worst[sample], breast$area_worst[sample], breast$concavity_worst[sample], breast$symmetry_worst[sample], breast$fractal_dimension_worst[sample])
general_result_test = data.frame(breast$radius_mean[-sample], breast$compactness_mean[-sample], breast$concavity_mean[-sample], breast$concave_points_mean[-sample], breast$radius_se[-sample], breast$smoothness_se[-sample], breast$concavity_se[-sample], breast$concave_points_se[-sample], breast$radius_worst[-sample], breast$texture_worst[-sample], breast$area_worst[-sample], breast$concavity_worst[-sample], breast$symmetry_worst[-sample], breast$fractal_dimension_worst[-sample])
colnames(general_result_train) = c("radius_mean","compactness_mean","concavity_mean","concave_points_mean","radius_se","smoothness_mean","concavity_se","concave_points_se","radius_worst","texture_worst","area_worst","concavity_worst","symmetry_worst","fractal_dimension_worst")
colnames(general_result_test) = c("radius_mean","compactness_mean","concavity_mean","concave_points_mean","radius_se","smoothness_mean","concavity_se","concave_points_se","radius_worst","texture_worst","area_worst","concavity_worst","symmetry_worst","fractal_dimension_worst")

y_train = y[sample]
y_test = y[-sample]

#LOGISTIC REGRESSION
model1 = glm(y_train~., data = general_result_train, family = binomial())
summary(model1)
prob = predict(model1, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model1)

general_result_train = data.frame(breast$radius_mean[sample], breast$compactness_mean[sample], breast$concavity_mean[sample], breast$concave_points_mean[sample], breast$radius_se[sample], breast$smoothness_se[sample], breast$concavity_se[sample], breast$concave_points_se[sample], breast$texture_worst[sample], breast$concavity_worst[sample], breast$symmetry_worst[sample], breast$fractal_dimension_worst[sample])
general_result_test = data.frame(breast$radius_mean[-sample], breast$compactness_mean[-sample], breast$concavity_mean[-sample], breast$concave_points_mean[-sample], breast$radius_se[-sample], breast$smoothness_se[-sample], breast$concavity_se[-sample], breast$concave_points_se[-sample], breast$texture_worst[-sample], breast$concavity_worst[-sample], breast$symmetry_worst[-sample], breast$fractal_dimension_worst[-sample])
colnames(general_result_train) = c("1","2","3","4","5","6","7","8","9","10","11","12")
colnames(general_result_test) = c("1","2","3","4","5","6","7","8","9","10","11","12")
model2 = glm(y_train~., data = general_result_train, family = binomial())
summary(model2)
prob = predict(model2, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model2)

general_result_train = data.frame(breast$radius_mean[sample], breast$compactness_mean[sample], breast$concave_points_mean[sample], breast$radius_se[sample], breast$smoothness_se[sample], breast$concave_points_se[sample], breast$texture_worst[sample], breast$concavity_worst[sample], breast$symmetry_worst[sample], breast$fractal_dimension_worst[sample])
general_result_test = data.frame(breast$radius_mean[-sample], breast$compactness_mean[-sample], breast$concave_points_mean[-sample], breast$radius_se[-sample], breast$smoothness_se[-sample], breast$concave_points_se[-sample], breast$texture_worst[-sample], breast$concavity_worst[-sample], breast$symmetry_worst[-sample], breast$fractal_dimension_worst[-sample])
colnames(general_result_train) = c("1","2","3","4","5","6","7","8","9","10")
colnames(general_result_test) = c("1","2","3","4","5","6","7","8","9","10")
model3 = glm(y_train~., data = general_result_train, family = binomial())
summary(model3)
prob = predict(model3, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model3)

general_result_train = data.frame(breast$radius_mean[sample], breast$concave_points_mean[sample], breast$radius_se[sample], breast$smoothness_se[sample], breast$concave_points_se[sample], breast$texture_worst[sample], breast$concavity_worst[sample], breast$symmetry_worst[sample], breast$fractal_dimension_worst[sample])
general_result_test = data.frame(breast$radius_mean[-sample], breast$concave_points_mean[-sample], breast$radius_se[-sample], breast$smoothness_se[-sample], breast$concave_points_se[-sample], breast$texture_worst[-sample], breast$concavity_worst[-sample], breast$symmetry_worst[-sample], breast$fractal_dimension_worst[-sample])
colnames(general_result_train) = c("1","2","3","4","5","6","7","8","9")
colnames(general_result_test) = c("1","2","3","4","5","6","7","8","9")
model4 = glm(y_train~., data = general_result_train, family = binomial())
summary(model4)
prob = predict(model4, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model4)

general_result_train = data.frame(breast$concave_points_se[sample], breast$perimeter_mean[sample], breast$smoothness_worst[sample], breast$concavity_se[sample], breast$texture_worst[sample], breast$concavity_mean[sample], breast$fractal_dimension_se[sample])
general_result_test = data.frame(breast$concave_points_se[-sample], breast$perimeter_mean[-sample], breast$smoothness_worst[-sample], breast$concavity_se[-sample], breast$texture_worst[-sample], breast$concavity_mean[-sample], breast$fractal_dimension_se[-sample])
colnames(general_result_train) = c("1","2","3","4","5","6","7")
colnames(general_result_test) = c("1","2","3","4","5","6","7")
model5= glm(y_train~., data = general_result_train, family = binomial())
summary(model5)
prob = predict(model5, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model5)

general_result_train = data.frame(breast$concave_points_se[sample], breast$perimeter_mean[sample], breast$smoothness_worst[sample], breast$texture_worst[sample], breast$concavity_mean[sample], breast$fractal_dimension_se[sample])
general_result_test = data.frame(breast$concave_points_se[-sample], breast$perimeter_mean[-sample], breast$smoothness_worst[-sample], breast$texture_worst[-sample], breast$concavity_mean[-sample], breast$fractal_dimension_se[-sample])
colnames(general_result_train) = c("1","2","3","4","5","6")
colnames(general_result_test) = c("1","2","3","4","5","6")
model6= glm(y_train~., data = general_result_train, family = binomial())
summary(model6)
prob = predict(model6, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model6)

general_result_train = data.frame(breast$perimeter_mean[sample], breast$smoothness_worst[sample], breast$texture_worst[sample], breast$concavity_mean[sample], breast$fractal_dimension_se[sample])
general_result_test = data.frame(breast$perimeter_mean[-sample], breast$smoothness_worst[-sample], breast$texture_worst[-sample], breast$concavity_mean[-sample], breast$fractal_dimension_se[-sample])
colnames(general_result_train) = c("perimeter_mean","smoothness_worst","texture_worst","concavity_mean","fractal_dimension_se")
colnames(general_result_test) = c("perimeter_mean","smoothness_worst","texture_worst","concavity_mean","fractal_dimension_se")
model7= glm(y_train~., data = general_result_train, family = binomial())
summary(model7)
prob = predict(model7, general_result_test, type="response")
new_logistic_pred = ifelse(prob > 0.5, 1, 0)
CrossTable(x = y_test, y = new_logistic_pred, prob.chisq = FALSE)
car::vif(model7)

predtrain = predict(model7,general_result_train,type="response")
ROCRpred = prediction(predtrain, y_train)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
plot(ROCRperf, colorize=TRUE)

auc_ROCR = performance(ROCRpred, measure = "auc")
auc_ROCR = auc_ROCR@y.values[[1]]
auc_ROCR

exp(model7$coefficients)
güven_aralıkları<- confint(model7)
confint.default(model7)
exp(güven_aralıkları)
exp(cbind(OR = coef(model7), güven_aralıkları))

summary(model1)
car :: vif(model1)


bc_classifier = naiveBayes(general_result_train, y_train)
naive_pred = predict(bc_classifier,general_result_test)

CrossTable(naive_pred, y_test, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))


