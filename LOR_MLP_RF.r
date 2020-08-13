#读数据
w <- read.csv(
    "dataset121803.csv"
, head = TRUE)

#类别数据设置为factor属性数据
w$Aspect <- factor(w$Aspect)
w$LULC <- factor(w$LULC)
w$Classes <- factor(w$Classes)
w$Lithology <- factor(w$Lithology)
w$ST <- factor(w$ST)

#参数归一化
w <- cbind(w[,1:5], scale(w[, 6:14]))

#划分训练集与验证集
index <- sample(2, nrow(w), replace = TRUE, prob = c(0.7, 0.3))
trainset <- w[index == 1, ]
testset <- w[index == 2, ]

#模型训练：multinom
library(nnet)

#OLR
w_train_MNL <- multinom(kernel ~ ., data = trainset)
#MLP
w_train_MLP <- nnet(
    kernel~.
    , data=trainset
    , size = 15
    , decay = 0.001
    , lineout = TRUE
    , trace = TRUE
    , rang = 115
    , na.action = na.omit
    , maxit = 200)

library(randomForest)
#RF
w_train_RF <- randomForest(
    kernel ~ .
, data = testset
, ntree=300
, importance = TRUE
, proximity = TRUE
, na.action = na.omit)

#训练集计算

train_pre_prob_MLP <- predict(w_train_MLP, type = "class")

#验证集计算
test_pre_class <- predict(w_train_RF, type = "class")#计算响应类别
