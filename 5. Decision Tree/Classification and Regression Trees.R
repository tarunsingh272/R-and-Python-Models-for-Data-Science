#https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
#

- Supervised Learning Methods
- Tree based methods empower predictive models with high accuracy, stability and ease of interpretation.
- Unlike linear models, they map non-linear relationships quite well. They are adaptable at solving any kind of problem at hand (classification or regression).

#Decision Tree
- supervised learning algorithm (having a pre-defined target variable)
- that is mostly used in classification problems.
- In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) 
- based on most significant splitter / differentiator in input variables.
- Root Node,
- Splitting
- Decision Node : When a sub-node splits into further sub-nodes, then it is called decision node
- Leaf/ Terminal Node: Nodes do not split is called Leaf or Terminal node.
- Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.

#Advnatages
- Easy to Understand:
- Useful in Data exploration: Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. 
- With the help of decision trees, we can create new variables / features that has better power to predict target variable.
- Less data cleaning required It is not influenced by outliers and missing values to a fair degree.
- Non Parametric Method: Decision tree is considered to be a non-parametric method. 
- This means that decision trees have no assumptions about the space distribution and the classifier structure.

#Disadvantages 
- Over fitting
- Not fit for continuous variables: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.

#######Regression Trees vs Classification Trees#########
- Regression trees are used when dependent variable is continuous. Classification trees are used when dependent variable is categorical.
- In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we'll make its prediction with mean value.
- In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling in that region. Thus, if an unseen data observation falls in that region, we'll make its prediction with mode value

#How does a tree decide where to split?
Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes. The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable.
The algorithm selection is also based on type of target variables. Let's look at the four most commonly used algorithms in decision tree:
- Gini Index : 
- Chi Square : It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node. We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable.
- Entropy is one way
- Reduction In variance

#preventing overfitting is pivotal while modeling a decision tree and it can be done in 2 ways:
  
Setting constraints on tree size
Tree pruning

Setting Constraints on Tree Size
- Minimum samples for a node split
- Minimum samples for a terminal node (leaf)
- Maximum depth of tree (vertical depth)
- Maximum number of terminal nodes
- Maximum features to consider for split


#If I can use logistic regression for classification problems and linear regression for regression problems, 
#why is there a need to use trees"? Many of us have this question. And, this is a valid one too.

#Actually, you can use any algorithm. It is dependent on the type of problem you are solving. Let's look at some key factors which will help you to decide which algorithm to use:
  
#If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
#If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
#If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. 
#Decision tree models are even simpler to interpret than linear regression!

#For R users, there are multiple packages available to implement decision tree such as ctree, rpart, tree etc.

#########Classification ###############
Classification is used to identify a category of new observations (testing datasets) based on
a classification model built from the training dataset, of which the categories are already
known. Similar to regression, classification is categorized as a supervised learning method
as it employs known answers (label) of a training dataset to predict the answer (label) of the
testing dataset. The main difference between regression and classification is that regression
is used to predict continuous values.

For example, one may use regression to predict the future price of a given stock based on
historical prices. However, one should use the classification method to predict whether the
stock price will rise or fall.

In this chapter, we will illustrate how to use R to perform classification. We first build a training
dataset and a testing dataset from the churn dataset, and then apply different classification
methods to classify the churn dataset. In the following recipes, we will introduce the treebased
classification method using a traditional classification tree and a conditional inference
tree, lazy-based algorithm, and a probabilistic-based method using the training dataset to
build up a classification model, and then use the model to predict the category (class label) of
the testing dataset. We will also use a confusion matrix to measure the performance.

#Perform the following steps to split the churn dataset into training and testing datasets:
#1. You can retrieve the churn dataset from the C50 package:
install.packages("C50")
library(C50)
data(churn)

str(churnTrain);

#We can remove the state, area_code, and account_length attributes, which are not appropriate for classification features:
  
churnTrain = churnTrain[,! names(churnTrain) %in% c("state","area_code", "account_length") ]

#Then, split 70 percent of the data into the training dataset and 30 percent of the data into the testing dataset:
set.seed(2)
ind = sample(2, nrow(churnTrain), replace = TRUE, prob=c(0.7,0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2,]

dim(trainset)
dim(testset)

The dataset contains 20 variables with 3,333 observations. We would like to build a classification model
to predict whether a customer will churn, which is very important to the telecom company as the cost 
of acquiring a new customer is significantly more than retaining one.

You can combine the split process of the training and testing datasets into the split.data
function. Therefore, you can easily split the data into the two datasets by calling this function
and specifying the proportion and seed in the parameters:

split.data = function(data, p = 0.7, s = 666){
    set.seed(s)
    index = sample(1:dim(data)[1])
    train = data[index[1:floor(dim(data)[1] * p)], ]
    test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
return(list(train = train, test = test))
}

#Building a classification model with recursive partitioning trees
A classification tree uses a split condition to predict class labels based on one or multiple
input variables. The classification process starts from the root node of the tree; at each node,
the process will check whether the input value should recursively continue to the right or left
sub-branch according to the split condition, and stops when meeting any leaf (terminal) nodes
of the decision tree.

#Perform the following steps to split the churn dataset into training and testing datasets:
1. Load the rpart package:
    library(rpart)
2. Use the rpart function to build a classification tree model:
   churn.rp = rpart(churn ~ ., data=trainset)
3. Type churn.rp to retrieve the node detail of the classification tree:
  churn.rp
4. Next, use the printcp function to examine the complexity parameter:
   printcp(churn.rp)
5. plotcp(churn.rp)
6. summary(churn.rp)

After the model is built, you can type the variable name of the built model, churn.rp, to
display the tree node details. In the printed node detail, n indicates the sample size, loss
indicates the misclassification cost, yval stands for the classified membership (no or yes, in
this case), and yprob stands for the probabilities of two classes (the left value refers to the
probability reaching label no, and the right value refers to the probability reaching label, yes).

The advantage of using the decision tree is that it is very flexible and easy to interpret. It works
on both classification and regression problems, and more; it is nonparametric. Therefore, one
does not have to worry about whether the data is linear separable. As for the disadvantage
of using the decision tree, it is that it tends to be biased and over-fitted. However, you can
conquer the bias problem through the use of a conditional inference tree, and solve the
problem of over-fitting through a random forest method or tree pruning.

#Visualizing a recursive partitioning tree
From the last recipe, we learned how to print the classification tree in a text format. To make
the tree more readable, we can use the plot function to obtain the graphical display of a
built classification tree.

Perform the following steps to visualize the classification tree:
1. Use the plot function and the text function to plot the classification tree:
plot(churn.rp, margin= 0.1)
text(churn.rp, all=TRUE, use.n = TRUE)


#You can also specify the uniform, branch, and margin parameter to adjust the layout:
plot(churn.rp, uniform=TRUE, branch=0.6, margin=0.1)
text(churn.rp, all=TRUE, use.n = TRUE)


#Measuring the prediction performance of a recursive partitioning tree
Since we have built a classification tree in the previous recipes, we can use it to predict the
category (class label) of new observations. Before making a prediction, we first validate the
prediction power of the classification tree, which can be done by generating a classification
table on the testing dataset. In this recipe, we will introduce how to generate a predicted label
versus a real label table with the predict function and the table function, and explain how
to generate a confusion matrix to measure the performance.

#You can use the predict function to generate a predicted label of testing the dataset:
 predictions = predict(churn.rp, testset, type="class")
 
#Use the table function to generate a classification table for the testing dataset:
 table(testset$churn, predictions)
 
#One can further generate a confusion matrix using the confusionMatrix function  provided in the caret package:
library(caret)
confusionMatrix(table(predictions, testset$churn))

In this recipe, we use a predict function and built up classification model, churn.rp, to
predict the possible class labels of the testing dataset, testset. The predicted categories
(class labels) are coded as either no or yes. Then, we use the table function to generate
a classification table on the testing dataset. From the table, we discover that there are
859 correctly predicted as no, while 18 are misclassified as yes. 100 of the yes predictions
are correctly predicted, but 41 observations are misclassified into no. Further, we use the
confusionMatrix function from the caret package to produce a measurement of the
classification model.

#Pruning a recursive partitioning tree
In previous recipes, we have built a complex decision tree for the churn dataset. However,
sometimes we have to remove sections that are not powerful in classifying instances to avoid
over-fitting, and to improve the prediction accuracy. Therefore, in this recipe, we introduce the
cost complexity pruning method to prune the classification tree.

#Perform the following steps to prune the classification tree:
1. Find the minimum cross-validation error of the classification tree model:
  min(churn.rp$cptable[,"xerror"])
2. Locate the record with the minimum cross-validation errors:
  which.min(churn.rp$cptable[,"xerror"])
3. Get the cost complexity parameter of the record with the minimum cross-validation errors:
  churn.cp = churn.rp$cptable[7,"CP"]
  churn.cp
4. Prune the tree by setting the cp parameter to the CP value of the record with minimum cross-validation errors:
   prune.tree = prune(churn.rp, cp= churn.cp)
5. Visualize the classification tree by using the plot and text function:
  plot(prune.tree, uniform=TRUE, branch=0.6, margin= 0.1)
  text(prune.tree, all=TRUE , use.n=TRUE)
  printcp(prune.tree)
  
6. Next, you can generate a classification table based on the pruned classification tree model:
  predictions = predict(prune.tree, testset, type="class")
  table(testset$churn, predictions)
  
  we discussed pruning a classification tree to avoid over-fitting and producing
  a more robust classification model. We first located the record with the minimum crossvalidation
  errors within the cptable, and we then extracted the CP of the record and
  assigned the value to churn.cp. Next, we used the prune function to prune the
  classification tree with churn.cp as the parameter. Then, by using the plot function,
  we graphically displayed the pruned classification tree. From Figure 5, it is clear that the
  split of the tree is less than the original classification tree (Figure 3). Lastly, we produced a
  classification table and used the confusion matrix to validate the performance of the pruned
  tree. The result shows that the accuracy (0.9411) is slightly lower than the original model
  (0.942), and also suggests that the pruned tree may not perform better than the original
  classification tree as we have pruned some split conditions (Still, one should examine the
                                                               change in sensitivity and specificity). However, the pruned tree model is more robust as it
  removes some split conditions that may lead to over-fitting.
  

#Building a classification model with a conditional inference tree
  In addition to traditional decision trees (rpart), conditional inference trees (ctree)
  are another popular tree-based classification method. Similar to traditional decision trees,
  conditional inference trees also recursively partition the data by performing a univariate
  split on the dependent variable. However, what makes conditional inference trees different
  from traditional decision trees is that conditional inference trees adapt the significance test
  procedures to select variables rather than selecting variables by maximizing information
  measures (rpart employs a Gini coefficient). In this recipe, we will introduce how to adapt
  a conditional inference tree to build a classification model.
  
#Perform the following steps to build the conditional inference tree:
1. First, we use ctree from the party package to build the classification model:
 library(party)
 ctree.model = ctree(churn ~ . , data = trainset)
2. Then, we examine the built tree model:
  ctree.model

Similar to rpart, the party package also provides a visualization method for users to plot
conditional inference trees. In the following recipe, we will introduce how to use the plot
function to visualize conditional inference trees

plot(ctree.model,margin=0.1);


#To obtain a simple conditional inference tree, one can reduce the built model with less input features, and redraw the classification tree:
daycharge.model = ctree(churn ~ total_day_charge, data =trainset)
plot(daycharge.model)

#Classifying data with the k-nearest neighbor classifier
############################################################
K-nearest neighbor (knn) is a nonparametric lazy learning method. From a nonparametric
view, it does not make any assumptions about data distribution. In terms of lazy learning,
it does not require an explicit learning phase for generalization. The following recipe will
introduce how to apply the k-nearest neighbor algorithm on the churn dataset.

#Perform the following steps to classify the churn data with the k-nearest neighbor algorithm:
1. First, one has to install the class package and have it loaded in an R session:
install.packages("class")
library(class)

2. Replace yes and no of the voice_mail_plan and international_plan
attributes in both the training dataset and testing dataset to 1 and 0:
  
  levels(trainset$international_plan) = list("0"="no", "1"="yes")
  levels(trainset$voice_mail_plan) = list("0"="no", "1"="yes")
  levels(testset$international_plan) = list("0"="no", "1"="yes")
  levels(testset$voice_mail_plan) = list("0"="no", "1"="yes")

3. Use the knn classification method on the training dataset and the testing dataset:
  churn.knn = knn(trainset[,! names(trainset) %in% c("churn")],testset[,! names(testset) %in% c("churn")], trainset$churn, k=3)
  summary(churn.knn)

#Classifying data with logistic regression
#################################################
  Logistic regression is a form of probabilistic statistical classification model, which can be
  used to predict class labels based on one or more features. The classification is done by
  using the logit function to estimate the outcome probability. One can use logistic regression
  by specifying the family as a binomial while using the glm function.

#Perform the following steps to classify the churn data with logistic regression:
1. With the specification of family as a binomial, we apply the glm function on the   dataset, trainset, by using churn as a class label and the rest of the variables as
  input features:
  fit = glm(churn ~ ., data = trainset, family=binomial)

2. Use the summary function to obtain summary information of the built logistic regression model:
  summary(fit)

#Then, we find that the built model contains insignificant variables, which would lead to misclassification. Therefore, we use significant variables only to train the
classification model:
 fit = glm(churn ~ international_plan + voice_mail_plan+total_intl_calls+number_customer_service_calls, data = trainset,
              family=binomial)

summary(fit)

3. Then, you can then use a fitted model, fit, to predict the outcome of testset. You can also determine the class by judging whether the probability is above 0.5:
    pred = predict(fit,testset, type="response")
    Class = pred >.5
\
Next, the use of the summary function will show you the binary outcome count, and reveal whether the probability is above 0.5:
summary(Class)
    
4. You can generate the counting statistics based on the testing dataset label and predicted result:
tb = table(testset$churn,Class)
tb

7. You can turn the statistics of the previous step into a classification table, and then
generate the confusion matrix:
churn.mod = ifelse(testset$churn == "yes", 1, 0)
pred_class = churn.mod
pred_class[pred<=.5] = 1- pred_class[pred<=.5]
ctb = table(churn.mod, pred_class)
ctb
confusionMatrix(ctb)


Logistic regression is very similar to linear regression; the main difference is that the
dependent variable in linear regression is continuous, but the dependent variable in logistic
regression is dichotomous (or nominal). The primary goal of logistic regression is to use logit
to yield the probability of a nominal variable is related to the measurement variable. We can
formulate logit in following equation: ln(P/(1-P)), where P is the probability that certain event
occurs.
The advantage of logistic regression is that it is easy to interpret, it directs model logistic
probability, and provides a confidence interval for the result. Unlike the decision tree, which
is hard to update the model, you can quickly update the classification model to incorporate
new data in logistic regression. The main drawback of the algorithm is that it suffers from
multicollinearity and, therefore, the explanatory variables must be linear independent. glm
provides a generalized linear regression model, which enables specifying the model in the
option family. If the family is specified to a binomial logistic, you can set the family as a
binomial to classify the dependent variable of the category.
The classification process begins by generating a logistic regression model with the use of
the training dataset by specifying Churn as the class label, the other variables as training
features, and family set as binomial. We then use the summary function to generate the
models summary information. From the summary information, we may find some insignificant
variables (p-values > 0.05), which may lead to misclassification. Therefore, we should
consider only significant variables for the model.
Next, we use the fit function to predict the categorical dependent variable of the testing
dataset, testset. The fit function outputs the probability of a class label, with a result
equal to 0.5 and below, suggesting that the predicted label does not match the label of
the testing dataset, and a probability above 0.5 indicates that the predicted label matches
the label of the testing dataset. Further, we can use the summary function to obtain the
statistics of whether the predicted label matches the label of the testing dataset. Lastly, in
order to generate a confusion matrix, we first generate a classification table, and then use
confusionMatrix to generate the performance measurement.



#Classification and Regression Trees (CART)
#Non linear models
#Recursive Parttioning Techniques

#We develop regression tree for the regression problem when output is continuous variable
#If output binary then we develop a classification tree

#Regression tree using rpart function from rpart package
#Recursive partioning when data is partitioned on basis of one variable then partioned again on the basis of other criteria or variable

library(RSADBE)
data(CART_Dummy);
str(CART_Dummy);

#Convert Y from int to factor variable
CART_Dummy$Y<-as.factor(CART_Dummy$Y);
attach(CART_Dummy);

#Initialise the graphics windows for the three samples by using 
par(mfrow=c(1,2));

#Create blank scatter plot
plot(c(0,12),c(0,10),type="n",xlab="X1",ylab="X2");
points(X1[Y==0],X2[Y==0],pch=15,col="red");
points(X1[Y==1],X2[Y==1],pch=19,col="green");
title(main="A Difficult Classification Problem");

#Second plot
plot(c(0,12),c(0,10),type="n",xlab="X1",ylab="X2");
points(X1[Y==0],X2[Y==0],pch=15,col="red");
points(X1[Y==1],X2[Y==1],pch=19,col="green");
#Partition X1 using abline 
abline(v=6,lwd=2);

#Add segments on the graph with the "segment" function
segments(x0=c(0,0,6,6),y0=c(3.75,6.25,2.25,5),x1=c(6,6,12,12),y1=c(3.75,6.25,2.25,5),lwd=2)
title(main="Looks a Solvable problem under partitions");

#So in above example a linear model is not appropriate and if we use non linear generalised models or polynomial 
#piecewise and spline regression models then it gets too complex so use classification techniques

#Explanatory variables/Independent variables can be discrete or continuous
#Discrete variables can be categorical and ordinal
#Ordinal order is imp, (Low, Medium, High)
#For continuou variable, we identify unique distinct values and then partition
#In case of ordinal variable : If there are m distinct orders, we consider m-1 distinct splitsof overall data
#In case of categorical variable with m categories, the no of possible splits becomes 2^(m-1)-1.

install.packages("rpart");
library("rpart");
CART_Dummy_rpart<-rpart(Y~X1+X2,data=CART_Dummy);
plot(CART_Dummy_rpart); 
text(CART_Dummy_rpart);
summary(CART_Dummy_rpart);
abline()

#Construction of a regression tree : output variable is continuous
#Classification Tree: If output variable categorical or ordinal random variable
#Main aim to minimise the residual sum of squares
#We need to split data at the points that keep the residual sum of squares a minimum
#That is for each unique value of predictor, which is candidate for node value
#We find sum of squares of y's within each partition of the data, then add them up
#Leading to leAST sum of squares error
#The residual sum of squares at each child node will be lesser than that in parent node
#We will first define a function, which will extract the best split given by the covariate and dependent variable.
#This action will be repeated for all available covariates and then we find best overall split
#The data will be partitioned by using best overall split and then best split will be identified for each of the partitioned data
# Now, the data is partitioned into two parts according to the best split. The process of finding the best split within each partition is repeated in the same spirit as for the first split

##############
#We will begin with a simple example of a regression tree, and use the rpart function to plot the regression function.
#Then, we will first define a function, which will extract the best split given by the covariate and dependent variable. This action will be repeated for all the available covariates, and then we find the best overall split. This will be verified with the regression tree. 
#The data will then be partitioned by using the best overall split, and then the best split will be identified for each of the partitioned data.


#REGRESSION TREEE
################################################
library("MASS")
data(cpus);
str(cpus);
plot(cpus$perf)


#Create regression tree for logarithm of perf as function of covariates and display regression tree
cpus.ltrpart<-rpart(log10(perf)~syct+mmin+mmax+cach+chmin+chmax,data=cpus)
plot(cpus.ltrpart);
text(cpus.ltrpart);

#Define getNode function. We need to find best split, the evaluation needs to be done for every distinct value of the covariate. 
#If there are m distinct points, we need m-1 evaluations. At each distinct point, the regressand needs to be partitioned accordingly, and the sum of squares should be obtained for each partition. The two sums of squares (in each part) are then added to obtain the reduced sum of squares. 
getNode <- function(x,y)  {
  xu <- sort(unique(x),decreasing=TRUE) 
  ss <- numeric(length(xu)-1)
  for(i in 1:length(ss))	{ partR <- y[x>xu[i]]
                           partL <- y[x<=xu[i]]
                           partRSS <- sum((partR-mean(partR))^2) 
                           partLSS <- sum((partL-mean(partL))^2) 
                           ss[i]<-partRSS + partLSS
  }
  return(list(xnode=xu[which(ss==min(ss,na.rm=TRUE))], 
  minss = min(ss,na.rm=TRUE),ss,xu))
}

The getNode function gives the best split for a given covariate. It returns a list consisting of four objects:
 xnode, which is a datum of the covariate x that gives the minimum residual sum of squares for the regressand y
	The value of the minimum residual sum of squares
	The vector of the residual sum of squares for the distinct points of the vector x
	The vector of the distinct x values

#We will run this function for each of the six covariates, and find the best overall split. 
#The argument na.rm=TRUE is required, as at the maximum value of x we won't get a numeric value.

#Execute the getNode function on syct covariate and look at the output
getNode(cpus$syct,log10(cpus$perf))$xnode;
getNode(cpus$syct,log10(cpus$perf))$minss
getNode(cpus$syct,log10(cpus$perf))[[3]]

#The least sum of squares at a split for the best split value of the syct variable is 24.72, and it occurs at a value of syct greater than 48
#The third and fourth list objects given by getNode, respectively, contain the details of the sum of squares for the potential candidates and the unique values of syct.

#Now, run the getNode function for the remaining five covariates: 
getNode(cpus$syct,log10(cpus$perf))[[2]] 
getNode(cpus$mmin,log10(cpus$perf))[[2]] 
getNode(cpus$mmax,log10(cpus$perf))[[2]] 
getNode(cpus$cach,log10(cpus$perf))[[2]] 
getNode(cpus$chmin,log10(cpus$perf))[[2]] 
getNode(cpus$chmax,log10(cpus$perf))[[2]] 
getNode(cpus$cach,log10(cpus$perf))[[1]] 
sort(getNode(cpus$cach,log10(cpus$perf))[[4]],decreasing=FALSE)

#The sum of squares for "cach" is the lowest, and hence we need to find the best split associated with it, which is 24. 
#However, the regression tree shows that the best split is for the cach value of 27. The getNode function says that the best split
#occurs at a point greater than 24, and hence we take the average of 24 and the next unique point at 30. 
#Having obtained the best overall split, we next obtain the first partition of the dataset.

#Partition the data by using the best overall split point:
cpus_FS_R <- cpus[cpus$cach>=27,] 
cpus_FS_L <- cpus[cpus$cach<27,]

#Identify the best split in each of the partitioned datasets: 
#Right
getNode(cpus_FS_R$syct,log10(cpus_FS_R$perf))[[2]] 
getNode(cpus_FS_R$mmin,log10(cpus_FS_R$perf))[[2]] 
getNode(cpus_FS_R$mmax,log10(cpus_FS_R$perf))[[2]] 
getNode(cpus_FS_R$cach,log10(cpus_FS_R$perf))[[2]] 
getNode(cpus_FS_R$chmin,log10(cpus_FS_R$perf))[[2]] 
getNode(cpus_FS_R$chmax,log10(cpus_FS_R$perf))[[2]] 
getNode(cpus_FS_R$mmax,log10(cpus_FS_R$perf))[[1]] 
sort(getNode(cpus_FS_R$mmax,log10(cpus_FS_R$perf))[[4]], decreasing=FALSE) 

#Left
getNode(cpus_FS_L$syct,log10(cpus_FS_L$perf))[[2]]
getNode(cpus_FS_L$mmin,log10(cpus_FS_L$perf))[[2]] 
getNode(cpus_FS_L$mmax,log10(cpus_FS_L$perf))[[2]] 
getNode(cpus_FS_L$cach,log10(cpus_FS_L$perf))[[2]] 
getNode(cpus_FS_L$chmin,log10(cpus_FS_L$perf))[[2]] 
getNode(cpus_FS_L$chmax,log10(cpus_FS_L$perf))[[2]] 
getNode(cpus_FS_L$mmax,log10(cpus_FS_L$perf))[[1]] 
sort(getNode(cpus_FS_L$mmax,log10(cpus_FS_L$perf))[[4]], decreasing=FALSE)

#Thus, for the first right partitioned data, the best split is for the mmax value as the mid-point between 24000 and 32000; that is, at mmax = 28000. 
#Similarly, for the first left-partitioned data, the best split is the average value of 6000 and 6200, which is 6100, for the same mmax covariate

#Partition the first right part cpus_FS_R as follows: 
cpus_FS_R_SS_R <- cpus_FS_R[cpus_FS_R$mmax>=28000,] 
cpus_FS_R_SS_L <- cpus_FS_R[cpus_FS_R$mmax<28000,]

#Obtain the best split for cpus_FS_R_SS_R and cpus_FS_R_SS_L by running the following code:
getNode(cpus_FS_R_SS_R$syct,log10(cpus_FS_R_SS_R$perf))[[2]] 
getNode(cpus_FS_R_SS_R$mmin,log10(cpus_FS_R_SS_R$perf))[[2]] 
getNode(cpus_FS_R_SS_R$mmax,log10(cpus_FS_R_SS_R$perf))[[2]] 
getNode(cpus_FS_R_SS_R$cach,log10(cpus_FS_R_SS_R$perf))[[2]] 
getNode(cpus_FS_R_SS_R$chmin,log10(cpus_FS_R_SS_R$perf))[[2]]
getNode(cpus_FS_R_SS_R$chmax,log10(cpus_FS_R_SS_R$perf))[[2]] 
getNode(cpus_FS_R_SS_R$cach,log10(cpus_FS_R_SS_R$perf))[[1]]
sort(getNode(cpus_FS_R_SS_R$cach,log10(cpus_FS_R_SS_R$perf))[[4]], decreasing=FALSE) 

getNode(cpus_FS_R_SS_L$syct,log10(cpus_FS_R_SS_L$perf))[[2]] 
getNode(cpus_FS_R_SS_L$mmin,log10(cpus_FS_R_SS_L$perf))[[2]] 
getNode(cpus_FS_R_SS_L$mmax,log10(cpus_FS_R_SS_L$perf))[[2]] 
getNode(cpus_FS_R_SS_L$cach,log10(cpus_FS_R_SS_L$perf))[[2]] 
getNode(cpus_FS_R_SS_L$chmin,log10(cpus_FS_R_SS_L$perf))[[2]] 
getNode(cpus_FS_R_SS_L$chmax,log10(cpus_FS_R_SS_L$perf))[[2]] 
getNode(cpus_FS_R_SS_L$cach,log10(cpus_FS_R_SS_L$perf))[[1]]
sort(getNode(cpus_FS_R_SS_L$cach,log10(cpus_FS_R_SS_L$perf)) [[4]],decreasing=FALSE)

#For the cpus_FS_R_SS_R part, the final division is according to cach being greater than 56 or not (average of 48 and 64). 
#If the cach value in this partition is greater than 56, then perf (actually log10(perf)) ends in the terminal leaf 3, else 2. 
#However, for the region cpus_FS_R_SS_L, we partition the data further by the cach value being greater than 96.5 (average of 65 and 128). 
#In the right side of the region, log10(perf) is found as 2, and a third level split is required for cpus_FS_R_SS_L with cpus_FS_R_SS_L_TS_L.

#Partition cpus_FS_L accordingly, as the mmax value being greater than 6100 or otherwise:
cpus_FS_L_SS_R <- cpus_FS_L[cpus_FS_L$mmax>=6100,] 
cpus_FS_L_SS_L <- cpus_FS_L[cpus_FS_L$mmax<6100,] 

#The rest of the partition for cpus_FS_L is completely given next.
#The details will be skipped and the R program is given right away: 
cpus_FS_L_SS_R <- cpus_FS_L[cpus_FS_L$mmax>=6100,] 
cpus_FS_L_SS_L <- cpus_FS_L[cpus_FS_L$mmax<6100,] 
getNode(cpus_FS_L_SS_R$syct,log10(cpus_FS_L_SS_R$perf))[[2]] 
getNode(cpus_FS_L_SS_R$mmin,log10(cpus_FS_L_SS_R$perf))[[2]] 
getNode(cpus_FS_L_SS_R$mmax,log10(cpus_FS_L_SS_R$perf))[[2]] 
getNode(cpus_FS_L_SS_R$cach,log10(cpus_FS_L_SS_R$perf))[[2]] 
getNode(cpus_FS_L_SS_R$chmin,log10(cpus_FS_L_SS_R$perf))[[2]]
getNode(cpus_FS_L_SS_R$chmax,log10(cpus_FS_L_SS_R$perf))[[2]] 
getNode(cpus_FS_L_SS_R$syct,log10(cpus_FS_L_SS_R$perf))[[1]] 
sort(getNode(cpus_FS_L_SS_R$syct,log10(cpus_FS_L_SS_R$perf))[[4]], decreasing=FALSE) 

getNode(cpus_FS_L_SS_L$syct,log10(cpus_FS_L_SS_L$perf))[[2]] 
getNode(cpus_FS_L_SS_L$mmin,log10(cpus_FS_L_SS_L$perf))[[2]] 
getNode(cpus_FS_L_SS_L$mmax,log10(cpus_FS_L_SS_L$perf))[[2]] 
getNode(cpus_FS_L_SS_L$cach,log10(cpus_FS_L_SS_L$perf))[[2]] 
getNode(cpus_FS_L_SS_L$chmin,log10(cpus_FS_L_SS_L$perf))[[2]] 
getNode(cpus_FS_L_SS_L$chmax,log10(cpus_FS_L_SS_L$perf))[[2]] 
getNode(cpus_FS_L_SS_L$mmax,log10(cpus_FS_L_SS_L$perf))[[1]] 
sort(getNode(cpus_FS_L_SS_L$mmax,log10(cpus_FS_L_SS_L$perf)) [[4]],decreasing=FALSE)

cpus_FS_L_SS_R_TS_R <- cpus_FS_L_SS_R[cpus_FS_L_SS_R$syct<360,] 
getNode(cpus_FS_L_SS_R_TS_R$syct,log10(cpus_FS_L_SS_R_TS_R$ perf))[[2]] 
getNode(cpus_FS_L_SS_R_TS_R$mmin,log10(cpus_FS_L_SS_R_TS_R$ perf))[[2]] 
getNode(cpus_FS_L_SS_R_TS_R$mmax,log10(cpus_FS_L_SS_R_TS_R$ perf))[[2]] 
getNode(cpus_FS_L_SS_R_TS_R$cach,log10(cpus_FS_L_SS_R_TS_R$ perf))[[2]] 
getNode(cpus_FS_L_SS_R_TS_R$chmin,log10(cpus_FS_L_SS_R_TS_R$ perf))[[2]] 
getNode(cpus_FS_L_SS_R_TS_R$chmax,log10(cpus_FS_L_SS_R_TS_R$ perf))[[2]] 
getNode(cpus_FS_L_SS_R_TS_R$chmin,log10(cpus_FS_L_SS_R_TS_R$ perf))[[1]] 
sort(getNode(cpus_FS_L_SS_R_TS_R$chmin,log10(cpus_FS_L_SS_R_TS_ R$perf))[[4]],decreasing=FALSE)

#Using the rpart function from the rpart library we first built the regression tree for log10(perf). 
#Then, we explored the basic definitions underlying the construction of a regression tree and defined the getNode function to obtain the best split for a pair of
#regressands and a covariate. This function is then applied for all the covariates, and the best overall split is obtained; using this we get our first partition of the data, which will be in agreement with the tree given by the rpart function. 
#We then recursively partitioned the data by using the getNode function and verified that all the best splits in each partitioned data are in agreement with the one provided by the rpart function

#CART as a powerful recursive partitioning method, useful for building (non-linear) models
#CONSTRUCTION OF CLASSIFICATION TREE
=====================================
-- Classification predicts catagorial class lebel for dataset
-- Prediction predicts missing value of a dataset

Working Strategy:
-- Classification classifies data based on some training set data
-- In case of prediction, it construct a classification model and predict missing value based on the model
  
  
  
#For identifying the split of a classification tree we need to define certain measures known as impurity measures.
#The three popular measures of impurity are Bayes error, the cross-entropy function, and Gini index. 
#Let p denote the percentage of success in a dataset of size n

#We will write a short program to understand the shape of these impurity measures as a function of p:
  
p <- seq(0.01,0.99,0.01)
plot(p,pmin(p,1-p),"l",col="red",xlab="p",xlim=c(0,1),ylim=c(0,1), ylab="Impurity Measures")
points(p,-p*log(p)-(1-p)*log(1-p),"l",col="green") 
points(p,p*(1-p),"l",col="blue") 
title(main="Impurity Measures")
legend(0.6,1,c("Bayes Error","Cross-Entropy","Gini Index"),col=c("red","green","blue"),pch="-");

library(rpart);
data(kyphosis)
head(kyphosis);

#We will first build the classification tree for Kyphosis as a function of the three variables Age, Start, and Number.
#The tree will then be displayed and rules will be extracted from it. The getNode function will be defined based on the cross-entropy function, which will be applied on the raw data and the first overall optimal split obtained to partition the data.
#The process will be recursively repeated until we get the same tree as returned by the rpart function.

#Using the option of split="information", construct a classification tree based on the cross-entropy information for the kyphosis data 
ky_rpart <- rpart(Kyphosis ~ Age + Number + Start, data=kyphosis,parms=list(split="information"))
plot(ky_rpart);
text(ky_rpart);

#http://www.solver.com/classification-tree


#Extract rules from ex_rpart by using asRules
library(rattle)
asRules(ky_rpart);


## ANALYTICS VIDHYA #########
Tree based learning algorithms are considered to be one of the best and mostly used supervised learning methods. 
Tree based methods empower predictive models with high accuracy, stability and ease of interpretation. 
Unlike linear models, they map non-linear relationships quite well. 
They are adaptable at solving any kind of problem at hand (classification or regression).

#Methods like decision trees, random forest, gradient boosting are being popularly used in all kinds of data science problems

#Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works for both categorical and continuous input and output variables. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables.

#As mentioned above, decision tree identifies the most significant variable and it's value that gives best homogeneous sets of population. Now the question which arises is, how does it identify the variable and the split? To do this, decision tree uses various algorithms, which we will shall discuss in the following section

#Types of Decision Trees

1. Types of decision tree is based on the type of target variable we have. It can be of two types:
  
#Categorical Variable Decision Tree: Decision Tree which has categorical target variable then it called as categorical variable decision tree. Example:- In above scenario of student problem, where the target variable was "Student will play cricket or not" i.e. YES or NO.
#Continuous Variable Decision Tree: Decision Tree has continuous target variable then it is called as Continuous Variable Decision Tree.

Root Node, Splitting, Decision Node, Leaf/Terminal Node

Splitting: It is a process of dividing a node into two or more sub-nodes.
Decision Node: When a sub-node splits into further sub-nodes, then it is called decision node
Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting

#Advantages
- Easy to understand : Graphical Intuitive representation
- Usefull in Data Exploraton : Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. With the help of decision trees, we can create new variables / features that has better power to predict target variable. It can also be used in data exploration stage
- Less Data Cleaning Required : It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree
- It can handle both numerical and categorical variables
- Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.

#Disadvantages
- Over fitting: Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning.
- Not fit for continuous variables: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.

2. Both the trees work almost similar to each other, let's look at the primary differences & similarity between classification and regression trees:
- Regression trees are used when dependent variable is continuous. Classification trees are used when dependent variable is categorical.
- In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we'll make its prediction with mean value.
- In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling in that region. Thus, if an unseen data observation falls in that region, we'll make its prediction with mode value.
- Both the trees divide the predictor space (independent variables) into distinct and non-overlapping regions. For the sake of simplicity, you can think of these regions as high dimensional boxes or boxes.
- Both the trees follow a top-down greedy approach known as recursive binary splitting/partitioning. We call it as 'top-down' because it begins from the top of tree when all the observations are available in a single region and successively splits the predictor space into two new branches down the tree. It is known as 'greedy' because, the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to a better tree.
- This splitting process is continued until a user defined stopping criteria is reached. For example: we can tell the the algorithm to stop once the number of observations per node becomes less than 50.
- In both the cases, the splitting process results in fully grown trees until the stopping criteria is reached. But, the fully grown tree is likely to overfit data, leading to poor accuracy on unseen data. This bring 'pruning'. Pruning is one of the technique used tackle overfitting. We'll learn more about it i


3. How does a tree decide where to split?

#The decision of making strategic splits heavily affects a tree's accuracy. 
#The decision criteria is different for classification and regression trees.

Decision trees use multiple algorithms to decide to split a node in two or more sub-nodes. 
The creation of sub-nodes increases the homogeneity of resultant sub-nodes. 
In other words, we can say that purity of the node increases with respect to the target variable. 
Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

The algorithm selection is also based on type of target variables. Let's look at the four most commonly used algorithms in decision tree:

#Gini Index
Gini index says, if we select two items from a population at random then they must be of same class and probability for this is 1 if population is pure.

- It works with categorical target variable "Success" or "Failure".
- It performs only Binary splits
- Higher the value of Gini higher the homogeneity.
- CART (Classification and Regression Tree) uses Gini method to create binary splits.

#Steps to Calculate Gini for a split
- Calculate Gini for sub-nodes, using formula sum of square of probability for success and failure (p^2+q^2).
- Calculate Gini for split using weighted Gini score of each node of that split


#Chi-Square
- It is an algorithm to find out the statistical significance between the differences between sub-nodes and parent node. 
- We measure it by sum of squares of standardized differences between observed and expected frequencies of target variable.

- It works with categorical target variable "Success" or "Failure".
- It can perform two or more splits.
- Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.
- Chi-Square of each node is calculated using formula,
- Chi-square = ((Actual - Expected)^2 / Expected)^1/2
- It generates tree called CHAID (Chi-square Automatic Interaction Detector)

#Steps to Calculate Chi-square for a split:
  
- Calculate Chi-square for individual node by calculating the deviation for Success and Failure both
- Calculated Chi-square of Split using Sum of all Chi-square of success and Failure of each node of the split


#Information Gain

#Look at the image below and think which node can be described easily. I am sure, your answer is C because it requires less information as all values are similar. On the other hand, B requires more information to describe it and A requires the maximum information. In other words, we can say that C is a Pure node, B is less Impure and A is more impure
#Now, we can build a conclusion that less impure node requires less information to describe it. And, more impure node requires more information. Information theory is a measure to define this degree of disorganization in a system known as Entropy. If the sample is completely homogeneous, then the entropy is zero and if the sample is an equally divided (50% - 50%), it has entropy of one.

#Entropy can be calculated using formula:-Entropy, Decision Tree

#Here p and q is probability of success and failure respectively in that node. Entropy is also used with categorical target variable. It chooses the split which has lowest entropy compared to parent node and other splits. The lesser the entropy, the better it is.

#Steps to calculate entropy for a split:
  
  - Calculate entropy of parent node
  - Calculate entropy of each individual node of split and calculate weighted average of all sub-nodes available in split.

#Reduction in Variance

#Till now, we have discussed the algorithms for categorical target variable. Reduction in variance is an algorithm used for continuous target variables (regression problems). This algorithm uses the standard formula of variance to choose the best split. The split with lower variance is selected as the criteria to split the population:
  
#Decision Tree, Reduction in Variance

#Steps to calculate Variance:
  
- Calculate variance for each node.
- Calculate variance for each split as weighted average of each node variance.

#Overfitting is one of the key challenges faced while modeling decision trees. If there is no limit set of a decision tree, it will give you 100% accuracy on training set because in the worse case it will end up making 1 leaf for each observation. Thus, preventing overfitting is pivotal while modeling a decision tree and it can be done in 2 ways:
Setting constraints on tree size
Tree pruning

#Setting Constraints on Tree Size
#This can be done by using various parameters which are used to define a tree. First, lets look at the general structure of a decision tree:

The parameters used for defining a tree are further explained below. The parameters described below are irrespective of tool. It is important to understand the role of parameters used in tree modeling. These parameters are available in R & Python.

#Minimum samples for a node split
  Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
  Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
  Too high values can lead to under-fitting hence, it should be tuned using CV.

#Minimum samples for a terminal node (leaf)
  Defines the minimum samples (or observations) required in a terminal node or leaf.
  Used to control over-fitting similar to min_samples_split.
  Generally lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in majority will be very small.

#Maximum depth of tree (vertical depth)
  The maximum depth of a tree.
  Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
  Should be tuned using CV.

#Maximum number of terminal nodes
  The maximum number of terminal nodes or leaves in a tree.
  Can be defined in place of max_depth. Since binary trees are created, a depth of 'n' would produce a maximum of 2^n leaves.

#Maximum features to consider for split
  The number of features to consider while searching for a best split. These will be randomly selected.
  As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features.
  Higher values can lead to over-fitting but depends on case to case
  

  #Tree Pruning
  
#As discussed earlier, the technique of setting constraint is a greedy-approach. In other words, it will check for the best split instantaneously and move forward until one of the specified stopping condition is reached. 
#Let's consider the following case when you're driving:
  
#This is exactly the difference between normal decision tree & pruning. 
#A decision tree with constraints won't see the truck ahead and adopt a greedy approach by taking a left. On the other hand if we use pruning, we in effect look at a few steps ahead and make a choice.
 - So we know pruning is better. But how to implement it in decision tree? The idea is simple.
 - We first make the decision tree to a large depth.
 - Then we start at the bottom and start removing leaves which are giving us negative returns when compared from the top.
 - Suppose a split is giving us a gain of say -10 (loss of 10) and then the next split on that gives us a gain of 20. A simple decision tree will stop at step 1 but in pruning, we will see that the overall gain is +10 and keep both leaves.


#Advanced packages like xgboost have adopted tree pruning in their implementation. But the library rpart in R, provides a function to prune. Good for R users!
  
#Are tree based models better than linear models?
"If I can use logistic regression for classification problems and linear regression for regression problems, why is there a need to use trees"? Many of us have this question. And, this is a valid one too.
-Actually, you can use any algorithm. It is dependent on the type of problem you are solving. Let's look at some key factors which will help you to decide which algorithm to use:
-If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.\
- If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
- If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!  
    
#What are ensemble methods in tree based modeling ?
    
#The literary meaning of word 'ensemble' is group. 
#Ensemble methods involve group of predictive models to achieve a better accuracy and model stability. 
#Ensemble methods are known to impart supreme boost to tree based models.
    
#Like every other model, a tree based model also suffers from the plague of bias and variance. 
#Bias means, 'how much on an average are the predicted values different from the actual value.'
#Variance means, 'how different will the predictions of the model be at the same point if different samples are taken from the same population'.

#You build a small tree and you will get a model with low variance and high bias. How do you manage to balance the trade off between bias and variance ?
  
#Normally, as you increase the complexity of your model, you will see a reduction in prediction error due to lower bias in the model. As you continue to make your model more complex, you end up over-fitting your model and your model will start suffering from high variance.
#champion model should maintain a balance between these two types of errors. This is known as the trade-off management of bias-variance errors. Ensemble learning is one way to execute this trade off analysis.
  
#Some of the commonly used ensemble methods include: Bagging, Boosting and Stacking. In this tutorial, we'll focus on Bagging and Boosting in detail.
  
#What is Bagging? How does it work?
  
- Bagging is a technique used to reduce the variance of our predictions by combining the result of multiple classifiers modeled on different sub-samples of the same data set. The following figure will make it clearer:
- The steps followed in bagging are:
    
#Create Multiple DataSets:
  Sampling is done with replacement on the original data and new datasets are formed.
  The new data sets can have a fraction of the columns as well as rows, which are generally hyper-parameters in a bagging model
  Taking row and column fractions less than 1 helps in making robust models, less prone to overfitting
#Build Multiple Classifiers:
  Classifiers are built on each data set.
  Generally the same classifier is modeled on each data set and predictions are made.
#Combine Classifiers:
  The predictions of all the classifiers are combined using a mean, median or mode value depending on the problem at hand.
  The combined values are generally more robust than a single model.
  

  


  

  

















