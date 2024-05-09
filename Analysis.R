

#-----------------------------------------------------------  

#load all required libraries
library(tidyverse)
library(ggthemes)
library(caret)
library(glmnet)
library(corrplot)
library(ggplot2)
library(tidyverse)
library(e1071)
library(class)
library(rpart)
library(randomForest)
library(rpart.plot)



data <- read.csv("UCI_Heart_Disease_Dataset_Combined.csv")
    

head(data)
names(data)
summary(data)
dim(data)
str(data)



missing_values <- sum(is.na(data))

numeric_data <- data[, sapply(data, is.numeric)]  


ggplot(data=data,aes(x=HeartDisease,fill=HeartDisease))+
  geom_bar()+
  labs(x='Heart Disease',y='Count',
       title="Overview of Data present for Heart Disease",
       subtitle = "We have fair distribution of data to study heart disease")+
  theme_bw()

data$Sex <- as.factor(data$Sex)
data$HeartDisease <- as.factor(data$HeartDisease)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)


# Define a custom palette of shades of blue
blue_palette <- c("#004c6d", "#005a84", "#00679b", "#0074b2", "#0081c9")

# distribution of gender in dataset

gender_counts <- table(data$Sex)

gender_percentages <- round(prop.table(gender_counts) * 100, 1)

pie(gender_counts, 
    labels = paste(c("Female",'Male'), ": ", gender_percentages, "%", sep = ""),
    main= "Gender Distribution",
    subtitle="Dataset is dominated by males entry representing 75% of dataset",
    col = c("#f8766d","#00bfc4"),
    cex = 0.8)+
  theme_bw()


# Age Distribution by Gender
ggplot(data, aes(x = Age, fill = Sex)) +
  geom_histogram(binwidth = 5,
                 position = "dodge",
                 ) +
  scale_fill_manual(values = c("#f8766d","#00bfc4")) +
  labs(title = "Age Distribution by Gender",
       subtitle="Age folows the normal distribution ",
       x = "Age",
       y = "Count") +
  theme_bw()

#investagating the age group ,and number of heart diseas data present in each group.

breaks <- c(20, 30, 40, 50, 60, 70, Inf)
labels <- c("20-29", "30-39", "40-49", "50-59", "60-69", "70+")
data$age_groups <- cut(data$Age, breaks = breaks, labels = labels, right = FALSE)

percentage <- data %>%
  group_by(age_groups) %>%
  summarize(total_count = n(),
            HeartDisease=sum(HeartDisease==1))

percentage <- transform(percentage, Percentage = HeartDisease / total_count * 100)
ggplot(percentage, aes(x = age_groups, y = Percentage)) +
  geom_bar(stat = "identity", fill = "#00bfc4") +
  geom_text(aes(label = paste0(round(Percentage,2), "%")), 
            vjust = -0.5, 
            color = "black", 
            size = 3) + 
  labs(title = "Percentage of Individuals with Heart Disease by Age Group",
       subtitle = "data is consitently desctribued amoung all age groups.",
       x = "Age Group",
       y = "Percentage") +
  theme_bw()




# Correlation plot

res <- cor(numeric_data)
corrplot(res, type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)



# Exercise-Induced Angina by Age Group

# Convert HeartDisease to factor
data$HeartDisease <- as.factor(data$HeartDisease)

# Distribution of Cholesterol Levels by Heart Disease Status
ggplot(data, aes(x = Cholesterol, fill = HeartDisease)) +
  geom_density(alpha=0.5) +
  labs(title = "Distribution of Cholesterol Levels by Heart Disease Status",
       subtitle="Heart rate for patient with heart disease is more dense in range of 180 - 250",
       x = "Cholesterol",
       y = "Density") +
  theme_minimal()

#-------------------------------- Predection ------------------------------------

# applying model to predict weather patience will have a heart disease or not ?
data<-subset(data,select=-(age_groups))
#splitting dataset into training and testing dataset.

set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(data), 0.8 * nrow(data))  # 80% training data
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#starting with logistic regression.
logit_model <- glm(HeartDisease ~ ., data=train_data, family=binomial)

# Evaluate the Model
predicted <- predict(logit_model, newdata=test_data, type="response")
predicted_class <- ifelse(predicted > 0.5, 1, 0) 

# evaluation metrics
confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

#naive bias classifier 

naive_bayes_model <- naiveBayes(HeartDisease ~ ., data=train_data)
predicted <- predict(naive_bayes_model, newdata=test_data)
predicted_class <- as.factor(predicted) 

confusion_matrix <- table(test_data$HeartDisease, predicted_class)
confusion_matrix
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)


#knn

k_values <- seq(1, 30)

# Vector to store accuracy for each k
accuracy_scores <- numeric(length(k_values))

# Loop through each k value
for (i in seq_along(k_values)) {
  k_value <- k_values[i]
  # Fit KNN model
  knn_model <- knn(train = train_data[, -ncol(train_data)], test = test_data[, -ncol(test_data)], cl = train_data$HeartDisease, k = k_value)
  # Evaluate accuracy
  accuracy_scores[i] <- sum(knn_model == test_data$HeartDisease) / length(test_data$HeartDisease)
}

# Print accuracy scores for each k
print(accuracy_scores)

#plotting accuracy
df <- data.frame(accuracy = accuracy_scores, index = seq_along(accuracy_scores))
ggplot(df, aes(x = index, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Accuracy Plot",
       x = "Index",
       y = "Accuracy")+
  ylim(0, 1) 

#k=3 is good fit.

# Decision tree
decision_tree_model <- rpart(HeartDisease ~ ., data = data)
predicted <- predict(decision_tree_model, newdata=test_data)

confusion_matrix <- table(test_data$HeartDisease, predicted_class)
confusion_matrix
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

rpart.plot(decision_tree_model)

#svm
y=train_data$HeartDisease
x=train_data[, !names(train_data) %in% "HeartDisease"]
y=sapply(y, as.numeric)

svmfit = svm(HeartDisease ~ .,
             data = train_data,
             type = 'C-classification',
             kernel = "linear", 
             cost = 10,
             scale = FALSE)
print(svmfit)

x_test<-test_data[, !names(train_data) %in% "HeartDisease"]
y_test<-test_data$HeartDisease
y_pred = predict(svmfit, newdata = x_test) 

cm = table(y_test, y_pred)
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Random Forest
rf_model <- randomForest(HeartDisease ~ ., 
                         data = train_data, 
                         ntree = 500)
predictions <- predict(rf_model, newdata = test_data)

confusion_matrix <- confusionMatrix(data = predictions, reference = test_data$HeartDisease)
accuracy <- confusion_matrix$overall['Accuracy']
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))
varImpPlot(rf_model)


# ------------------- improving accuracy by identifying outlier and removing it 


             
# converting oldpeak to positeve

#data$Oldpeak<-abs(data$Oldpeak)
#hist(data$Oldpeak)

#dropping row with 0 resting bp
data <- subset(data, RestingBP != 0)

#dropping rows with 0 cholostrol
data <- subset(data, Cholesterol != 0)



# Applying models  

#splitting dataset into training and testing dataset.

set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(data), 0.8 * nrow(data))  # 80% training data
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#starting with logistic regression.
logit_model <- glm(HeartDisease ~ ., data=train_data, family=binomial)

# Evaluate the Model
predicted <- predict(logit_model, newdata=test_data, type="response")
predicted_class <- ifelse(predicted > 0.5, 1, 0) 

# evaluation metrics
confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

#naive bias classifier 

naive_bayes_model <- naiveBayes(HeartDisease ~ ., data=train_data)
predicted <- predict(naive_bayes_model, newdata=test_data)
predicted_class <- as.factor(predicted) 

confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

#knn

k_values <- seq(1, 30)

# Vector to store accuracy for each k
accuracy_scores <- numeric(length(k_values))

# Loop through each k value
for (i in seq_along(k_values)) {
  k_value <- k_values[i]
  # Fit KNN model
  knn_model <- knn(train = train_data[, -ncol(train_data)], test = test_data[, -ncol(test_data)], cl = train_data$HeartDisease, k = k_value)
  # Evaluate accuracy
  accuracy_scores[i] <- sum(knn_model == test_data$HeartDisease) / length(test_data$HeartDisease)
}

# Print accuracy scores for each k
print(accuracy_scores)

#plotting accuracy
df <- data.frame(accuracy = accuracy_scores, index = seq_along(accuracy_scores))
ggplot(df, aes(x = index, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Accuracy Plot",
       x = "Index",
       y = "Accuracy")+
  ylim(0, 1) 

#k=4 is good fit.

# Decision tree
decision_tree_model <- rpart(HeartDisease ~ ., data = data)
predicted <- predict(decision_tree_model, newdata=test_data)

confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

#svm
y=train_data$HeartDisease
x=train_data[, !names(train_data) %in% "HeartDisease"]
y=sapply(y, as.numeric)

svmfit = svm(HeartDisease ~ .,
             data = train_data,
             type = 'C-classification',
             kernel = "linear", 
             cost = 10,
             scale = FALSE)
print(svmfit)

y_pred = predict(svmfit, newdata = x_test) 

cm = table(y_test, y_pred)
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Random Forest
rf_model <- randomForest(HeartDisease ~ .,
                         data = train_data,
                         ntree = 500)
predictions <- predict(rf_model, newdata = test_data)

confusion_matrix <- confusionMatrix(data = predictions, reference = test_data$HeartDisease)
accuracy <- confusion_matrix$overall['Accuracy']
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))


#-------------------------------------- improving model accuracy -------------------------

data <- read.csv("UCI_Heart_Disease_Dataset_Combined.csv")

#selecting features on based on correlation

data_subset<-subset(data,select=c("FastingBS",
                                  "ChestPainType",
                                  "RestingBP",
                                  "MaxHR",
                                  
                                  "HeartDisease",
                                  "RestingECG"))

#splitting dataset into training and testing dataset.

set.seed(123)  # For reproducibility
train_index <- sample(1:nrow(data_subset), 0.8 * nrow(data_subset))  # 80% training data
train_data <- data_subset[train_index, ]
test_data <- data_subset[-train_index, ]

#starting with logistic regression.
logit_model <- glm(HeartDisease ~ ., data=train_data, family=binomial)

# Evaluate the Model
predicted <- predict(logit_model, newdata=test_data, type="response")
predicted_class <- ifelse(predicted > 0.5, 1, 0) 

# evaluation metrics
confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

#naive bias classifier 

naive_bayes_model <- naiveBayes(HeartDisease ~ ., data=train_data)
predicted <- predict(naive_bayes_model, newdata=test_data)
predicted_class <- as.factor(predicted) 

confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

#knn 
k_values <- seq(1, 30)

# Vector to store accuracy for each k
accuracy_scores <- numeric(length(k_values))

# Loop through each k value
for (i in seq_along(k_values)) {
  k_value <- k_values[i]
  # Fit KNN model
  knn_model <- knn(train = train_data[, -ncol(train_data)], test = test_data[, -ncol(test_data)], cl = train_data$HeartDisease, k = k_value)
  # Evaluate accuracy
  accuracy_scores[i] <- sum(knn_model == test_data$HeartDisease) / length(test_data$HeartDisease)
}

# Print accuracy scores for each k
print(accuracy_scores)

#plotting accuracy
df <- data.frame(accuracy = accuracy_scores, index = seq_along(accuracy_scores))
ggplot(df, aes(x = index, y = accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Accuracy Plot",
       x = "Index",
       y = "Accuracy")+
  ylim(0, 1) 

#Decision Tree
decision_tree_model <- rpart(HeartDisease ~ ., data = data_subset)
predicted <- predict(decision_tree_model, newdata=test_data)

confusion_matrix <- table(test_data$HeartDisease, predicted_class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", accuracy, "\n")

#svm
y=train_data$HeartDisease
x=train_data[, !names(train_data) %in% "HeartDisease"]
y=sapply(y, as.numeric)

svmfit = svm(HeartDisease ~ .,
             data = train_data,
             type = 'C-classification',
             kernel = "linear", 
             cost = 10,
             scale = FALSE)
print(svmfit)

y_pred = predict(svmfit, newdata = x_test) 

cm = table(y_test, y_pred)
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# Random Forest
rf_model <- randomForest(HeartDisease ~ ., data = train_data, ntree = 500)
predictions <- predict(rf_model, newdata = test_data)

confusion_matrix <- confusionMatrix(data = predictions,
                                    reference = test_data$HeartDisease)
accuracy <- confusion_matrix$overall['Accuracy']
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

##################################################################
##################################################################
##################################################################

#unsupervised learning 
heart_data<-read.csv("UCI_Heart_Disease_Dataset_Combined.csv")

#dropping row with 0 resting bp
heart_data <- subset(heart_data, RestingBP != 0)

#dropping rows with 0 cholostrol
heart_data <- subset(heart_data, Cholesterol != 0)

heart_data_heartDisease<-heart_data$HeartDisease

heart_data <- na.omit(heart_data)

wss <- numeric(20)
for (i in 1:20) {
  kmeans_fit <- kmeans(heart_data, centers = i, nstart = 10)
  wss[i] <- kmeans_fit$tot.withinss
}

plot(1:20, wss,
     type = "b",
     xlab = "Number of Clusters (K)",
     ylab = "Within-Cluster Sum of Squares (WSS)",
     main = "K-Means Clustring for Heart Disease")


#Selecting k

k<-3
kmeans_fit <- kmeans(heart_data, centers = k, nstart = 10)
cluster_assignments <- kmeans_fit$cluster

heart_data_clustered <- cbind(heart_data, Cluster = cluster_assignments)

aggregate(heart_data, by = list(heart_data_clustered$Cluster), FUN = mean)


#binding heart disease column to it.
heart_data_final<-cbind(heart_data_clustered,heart_data_heartDisease)
  
summary<- heart_data_final %>%
  group_by(Cluster) %>%
  summarise(Probability_of_HeartDisease = mean(HeartDisease),
            Choloestrol=mean(Cholesterol),
            Resting_BP=mean(RestingBP),
            Maximum_HR=mean(MaxHR),
            Fasting_BS=mean(FastingBS),
            RestingECG=mean(RestingECG),
            Oldpeak=mean(Oldpeak)
            )                                                  

#plotting findings
library(gridExtra)

# Create individual bar charts for each column
bar_chart1 <- ggplot(data = summary, 
                     aes(x = Cluster, 
                         y = Probability_of_HeartDisease,
                         fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Probability of Heart Disease",y="Probabilit of HeartDisease") +
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()


bar_chart2 <- ggplot(data = summary, 
                     aes(x = Cluster, y = Choloestrol,fill = factor(Cluster)),
                     ) +
  geom_bar(stat = "identity") +
  labs(title = "Cholesterol Level",y="Choloestrol Level")+
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()

bar_chart3 <- ggplot(data = summary, aes(x = Cluster, y = Resting_BP,fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Resting Blood Pressure",y="Resting Blood Pressure")+
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()

bar_chart4 <- ggplot(data = summary, aes(x = Cluster, y = Maximum_HR,fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Maximum Heart Rate",y="Maximum Heart Rate")+
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()

bar_chart5 <- ggplot(data = summary, aes(x = Cluster, y = Fasting_BS,fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Fasting Blood Sugar",y="Fasting Blood Sugar")+
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()

bar_chart6 <- ggplot(data = summary, aes(x = Cluster, y = RestingECG,fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Resting Electrocardiographic Results",y="Resting ECG")+
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()

bar_chart7 <- ggplot(data = summary, aes(x = Cluster, y = Oldpeak,fill = factor(Cluster))) +
  geom_bar(stat = "identity") +
  labs(title = "Oldpeak",y="OldPeak")+
  scale_fill_manual(values = c("#FF0000", "#33FF33", "#33FF33"))+
  theme_bw()

# Combine the individual bar charts into a single image
combined_plot <- grid.arrange(bar_chart1,
                              bar_chart2,
                              bar_chart3,
                              bar_chart4,
                              bar_chart5,
                              bar_chart6,
                              bar_chart7,
                              nrow = 4)+title('Clusters of Data')


title_grob <- textGrob("Clusters of Data",
                       gp = gpar(fontsize = 16,
                                 fontface = "bold")
                       )
arranged_plots <- arrangeGrob(combined_plot, top = title_grob)
# Display the combined plot
print(combined_plot)

