load("C:/Users/anton/Documents/Carrera/Titanic/titanic.RData")
# VARIABLE DESCRIPTIONS:
#   survival        Survival
# (0 = No; 1 = Yes)
# pclass          Passenger Class
# (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
# (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# SPECIAL NOTES:
#   Pclass is a proxy for socio-economic status (SES)
# 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# Age is in Years; Fractional if Age less than One (1)
# If the Age is Estimated, it is in the form xx.5
# 
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# 
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.

library(dplyr)
library(plyr)
set.seed(123)
load("C:/Users/anton/Documents/Carrera/Redes neuronales R/titanic.RData")
setwd("C:/Users/anton/Documents/Carrera/Redes neuronales R")
datos <- na.omit(read.csv("train.csv"))
datos <- select(datos,-Cabin,-Name,-PassengerId,-Ticket,-Cabin)
##Poner como factor el dato a clasificar
datos$Survived <- as.factor(datos$Survived)
datos$Pclass <- as.factor(datos$Pclass)
datos$SibSp <- as.factor(datos$SibSp)
datos$Parch <- as.factor(datos$Parch)
datos$Age <- as.integer(datos$Age)
datos <- datos[-which(datos$Embarked==""),]
datos <- datos[-which(datos$Parch==6),]
str(datos)
train <- sample(1:nrow(datos),nrow(datos)*0.7)
datos.train <- datos[train,]
datos.test <- datos[-train,]
test <- t(select(datos.test,Survived))
errores.finales <- 1:9
names(errores.finales) <- c("Neural Network","Logistic Regression",
                            "Single Prune Tree","Random Forest","SVM Linear","SVM Polynomial",
                            "SVM Sigmoid","SVM Radial","Boosting Trees")
str(datos)
##Redes neuronales
errores.nnet <- c()
library(nnet)
set.seed(1)
for (i in 1:25){
    net.datos <- nnet(Survived ~.,
                        data = datos.train,
                        size = i,
                        linout = FALSE, #FALSE para clasificadores
                        maxit = 10000,
                        trace =FALSE)
    info.net <- as.numeric(predict(net.datos, datos.test,type = "class"))
    errores.nnet[i] <- sum(test !=info.net)/nrow(datos.test)
    print(i)
}
errores.finales[1] <- min(errores.nnet)

#Regresión lógistica
logistica.datos <- glm(Survived ~.,
                      data = datos.train,
                      family = binomial)
info.logistica <- round(predict(logistica.datos,datos.test,type = "response"))
errores.finales[2] <-
                  sum(test!=info.logistica)/nrow(datos.test) 
#Tree
library(tree)
set.seed(1)
tree.datos <- tree(Survived~.,
                    datos.train)
cv.datos <- cv.tree(tree.datos)
prune.datos <- prune.misclass(tree.datos,
                                  best =7)
plot(prune.datos)
text(prune.datos ,pretty =0)
info.tree <- predict(prune.datos,datos.test,type="class")

errores.finales[3] <- sum(info.tree!=test)/nrow(datos.test) 

#Random Forest
library(randomForest)
ntrees <- seq(from = 250, to = 2000, by = 250)
errores.random <- data.frame()
set.seed(1)
for(i in 1:(length(names(datos))))
  for(j in 1:length(ntrees)){
    {
      if(i!=length(names(datos))){
      rf.datos <- randomForest(Survived~.,
                                data = datos.train,
                                 mtry=i,
                                 ntree = ntrees[j],
                                 importance =TRUE)
      info.random <- predict(rf.datos,datos.test)
      errores.random[i,j] <-sum(info.random!=test)/nrow(datos.test) 
      }
      print(j)
    }
    print(i)
}
which(errores.random == min(errores.random),arr.ind = TRUE)
errores.finales[4] <- min(errores.random)

#Support Vector Machines
library (e1071)
#Linear
cost <- seq(from = 0.01, to = 10, by = 0.05)
svmL.erroresL <- c()
set.seed(1)
for (i in 1:200){
  svmL.datos <- svm(Survived~., 
                      data=datos.train,
                      kernel ="linear",
                      cost = cost[i])
  info.svmL <- predict(svmL.datos,datos.test)
  svmL.erroresL[i] <- sum(test !=info.svmL)/nrow(datos.test)
  print(i)
}
errores.finales[5] <- min(svmL.erroresL)

#Polinomial
cost <- seq(from = 0.01, to = 10, by = 0.05)
svm.erroresP <- c()
set.seed(1)
for (i in 1:length(cost)){
  svmP.datos <- svm(Survived~., 
                   data=datos.train,
                   kernel ="polynomial",
                   cost = cost[i])
  info.svmP <- predict(svmP.datos,datos.test)
  svm.erroresP[i] <- sum(test!=info.svmP)/nrow(datos.test)
  print(i)
}
errores.finales[6] <- min(svm.erroresP)

#Sigmoid
cost <- seq(from = 0.01, to = 10, by = 0.05)
svm.erroresS <- c()
set.seed(1)
for (i in 1:200){
  svmS.datos <- svm(Survived~., 
                   data=datos.train,
                   kernel ="sigmoid",
                   cost = cost[i])
  info.svmS <- predict(svmS.datos,datos.test)
  svm.erroresS[i] <- sum(test !=info.svmS)/nrow(datos.test)
  print(i)
}
errores.finales[7] <- min(svm.erroresS)

#Radial
svm.erroresR <- data.frame()
gama <- seq(from = 0.1,to = 5,by = 0.1)
cost <- seq(from = 0.1,to = 5,by = 0.1)
set.seed(1)
for(j in 1:length(gama)){
  for (i in 1:length(cost)){
    svmR.datos <- svm(Survived~., 
                     data=datos.train,
                     kernel ="radial",
                     gamma = gama[j],
                     cost = cost[i])
    info.svmR <- predict(svmR.datos,datos.test)
    svm.erroresR[j,i] <- sum(datos.test$Survived !=info.svmR)/nrow(datos.test)
  }
  print(j)
}
errores.finales[8] <- min(svm.erroresR)

which(svm.erroresL == min(svm.erroresL),arr.ind = TRUE)

#Boosting trees
library(gbm)
#GBM don´t accept factor variables
datos$Survived <- as.integer(datos$Survived) 
datos$Survived <- ifelse(datos$Survived == 1,0,1)
datos.train <- datos[train,]
datos.test <- datos[-train,]
ntrees <- seq(from = 250, to = 1000, by = 250)
shrink <- seq(from = 0.001, to = 0.5, by = 0.01)

gbm.list <- list()
gbm.dataframe <- data.frame()
set.seed(1)
for(k in 1:length(ntrees)){
  for(j in 1:length(datos)){
    for(i in 1:length(shrink)){
      set.seed(1)
      boost.datos <- gbm(Survived~., 
                         data = datos.train, 
                         distribution ="bernoulli",
                         n.trees = ntrees[k], 
                         interaction.depth = j, 
                         shrinkage = shrink[i],
                         verbose = FALSE)
      
      info <- predict(boost.datos,
                      datos.test,
                      n.trees = ntrees[k],
                      type = "response")
      info <- ifelse(info > 0.5,1,0)
      gbm.dataframe[j,i] <- sum(info!=test)/nrow(datos.test) 
    }
    print(j)
  }
  print(k)
  gbm.list[[k]] <- gbm.dataframe
  gbm.dataframe <- data.frame()
}
errores.finales[9] <- min(sapply(gbm.list,min))

treeOpt <- ntrees[which.min(sapply(gbm.list, min))]
treIndex <- which.min(sapply(gbm.list, min))

optimal <- which(gbm.list[[treIndex]] == min(gbm.list[[treIndex]]),
                 arr.ind = TRUE)

depOpt <- optimal[1]  #6
shrinkOpt <- shrink[optimal[2]] #0.031

set.seed(1)
boost.datos <- gbm(Survived~., 
                   data = datos.train, 
                   distribution ="bernoulli",
                   n.trees = treeOpt, 
                   interaction.depth = depOpt, 
                   shrinkage = shrinkOpt,
                   verbose = FALSE)

info <- predict(boost.datos,
                datos.test,
                n.trees = 250,
                type = "response")
info <- ifelse(info > 0.5,1,0)
print(sum(info!=test)/nrow(datos.test)) 


summary(boost.datos)

plot(boost.datos,"Pclass")
plot(boost.datos,"Sex")
plot(boost.datos,"Age")
plot(boost.datos,"SibSp")
plot(boost.datos,"Parch")
plot(boost.datos,"Fare")
plot(boost.datos,"Embarked")

#KNN
datos.cluster <- datos
for( i in 1:ncol(datos)){
  if(class(datos[,i])=="factor")
    datos.cluster[,i] <- as.integer(datos.cluster[,i])
}
datos.cluster.train <- datos.cluster[train,]
datos.cluster.test <- datos.cluster[-train,]

train_X <- select(datos.cluster.train,-Survived)
test_X <- select(datos.cluster.test,-Survived)
test_Y<- datos.cluster.test$Survived
train_Y <- datos.cluster.train$Survived
errores.knn <- c() 
library(class)
for(i in 1:30){
  knn.datos <- knn(train_X,test_X,train_Y,k=i)
  errores.knn[i] <- sum(knn.datos!=test_Y)/nrow(datos.test) #Porcentaje de error
}
errores.finales[12] <- min(errores.knn)



#Clusters
#Es necesario que las variables sean númericas
datos.cluster <- datos
for( i in 1:ncol(datos)){
  if(class(datos[,i])=="factor")
    datos.cluster[,i] <- as.integer(datos.cluster[,i])
}
clas <- kmeans(datos.cluster,2)

rbind(clas$cluster,datos.cluster$Survived)


#Exploración de datos
library(ggplot2)

datos$Survived <- as.factor(datos$Survived)
#Survived
table(datos$Survived)
survived <- split(datos$Survived,datos$Survived)
names(survived) <- c("No survived","Survived")
sapply(survived,summary)


png("survived.png")
ggplot(data = datos,aes(Survived, fill = Survived)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Survived") +
  xlab("") + 
  ylab("") +
  scale_fill_discrete(name="Survived", labels=c("No", "Yes")) +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("No","Yes"))
dev.off()

#Sex
table(datos$Sex)

sex <- split(datos$Sex,datos$Survived)
names(sex) <- c("No survived","Survived")
sapply(sex,summary)

png("sex.png")
ggplot(data = datos,aes(Sex, fill = Sex)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Sex") +
  xlab("") + 
  ylab("") +
  scale_fill_discrete(name="Survived", labels=c("No", "Yes")) +
  scale_x_discrete(breaks=c("female","male"),
                   labels=c("Female","Male"))
dev.off()

#Sex / Survived
png("sexsurvived.png")
ggplot(data = datos,aes(Sex, fill = Survived)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Sex") +
  xlab("") + 
  ylab("") +
  scale_fill_discrete(name="Survived", labels=c("No", "Yes")) +
  scale_x_discrete(breaks=c("female","male"),
                   labels=c("Female","Male"))
dev.off()

#Embarked
embarked <- split(datos$Embarked,datos$Survived)
names(embarked) <- c("No survived","Survived")
sapply(embarked,summary)


png("embarked.png")
ggplot(data = datos,aes(Embarked,fill = Embarked)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Embarked") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Embarked", labels=c("Cherbourg", "Queenstown","Southampton")) +
  scale_x_discrete(breaks=c("C","Q","S"),
                   labels=c("Cherbourg","Queenstown","Southampton"))
dev.off()

#Embarked / Survived
png("embarkedsurvived.png")
ggplot(data = datos,aes(Embarked, fill = Survived)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Embarked") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Survived", labels=c("No", "Yes")) +
  scale_x_discrete(breaks=c("C","Q","S"),
                   labels=c("Cherbourg","Queenstown","Southampton"))
dev.off()

#Passager Class
Pclass <- split(datos$Pclass,datos$Survived)
names(Pclass) <- c("No survived","Survived")
sapply(Pclass,summary)

png("pclass.png")
ggplot(data = datos,aes(Pclass,fill = Pclass)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Passager Class") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Class", labels=c("Upper", "Middle","Lower")) +
  scale_x_discrete(breaks=c("1","2","3"),
                   labels=c("Upper", "Middle","Lower"))
dev.off()

#Passager Class / Survived
png("pclasssurvived.png")
ggplot(data = datos,aes(Pclass, fill = Survived)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Passager class") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Survived", labels=c("No", "Yes")) +
  scale_x_discrete(breaks=c("1","2","3"),
                   labels=c("Upper", "Middle","Lower"))
dev.off()

#SibSp
SibSp <- split(datos$SibSp,datos$Survived)
names(SibSp) <- c("No survived","Survived")
sapply(SibSp,summary)

png("SibSp.png")
ggplot(data = datos,aes(SibSp,fill = SibSp)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Number of Siblings/Spouses Aboard") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Total", 
                      labels=c("0", "1","2","3","4","5")) +
  theme(legend.position = "none") 
dev.off()

#SibSp Survived
png("sibspsurvived.png")
ggplot(data = datos,aes(SibSp, fill = Survived)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Number of Siblings/Spouses Aboard") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Survived", labels=c("No", "Yes"))
dev.off()


#Parch

parch <- split(datos$Parch,datos$Survived)
names(parch) <- c("No survived","Survived")
sapply(parch,summary)

png("parch.png")
ggplot(data = datos,aes(Parch,fill = Parch)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Number of Parents/Children Aboard") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Total", 
                      labels=c("0", "1","2","3","4","5","6")) +
  theme(legend.position = "none") 
dev.off()

#Parch Survived
png("parchsurvived.png")
ggplot(data = datos,aes(Parch,fill = Survived)) + 
  geom_bar() + 
  theme_minimal() +
  ggtitle("Number of Parents/Children Aboard") +
  xlab("") + 
  ylab("") + 
  scale_fill_discrete(name="Survived", labels=c("No", "Yes"))
dev.off()


#Fare Histogram
png("farehist.png")
ggplot(data = datos,aes(Fare,fill= Survived)) + 
  geom_histogram() + 
  theme_minimal() +
  xlab("") +
  ylab("") +
  ggtitle("Passenger fare histogram") + 
  scale_fill_discrete(name="Survived", labels=c("No", "Yes")) +
  xlim(c(0,280))
dev.off()


#Fare
fare <- split(datos$Fare,datos$Survived)
names(fare) <- c("No survived","Survived")
sapply(fare,summary)

#Fare Boxplot
png("farebox.png")
ggplot(data = datos,aes(x = Survived,
                         y = Fare, 
                         fill = Survived)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Passenger fare boxplot") +
  xlab("") + 
  ylab("") +
  theme(legend.position = "none") +
  ylim(0,280) +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Dead","Survived"))
dev.off()

#Age
age <- split(datos$Age,datos$Survived)
names(age) <- c("No survived","Survived")
sapply(age,summary)
#Age boxplot
png("agebox")
ggplot(data = datos,aes(x = Survived,
                        y = Age, 
                        fill = Survived)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Passenger age") +
  xlab("") + 
  ylab("") +
  theme(legend.position = "none") +
  scale_x_discrete(breaks=c("0","1"),
                   labels=c("Dead","Survived")) 
dev.off()

