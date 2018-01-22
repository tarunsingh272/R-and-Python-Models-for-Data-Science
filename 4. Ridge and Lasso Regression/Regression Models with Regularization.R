#http://pingax.com/regularization-implementation-r/

Regression Models with Regularization
-------------------------------------------
#In the model selection issues with the linear regression model, 
#we found that a covariate is either selected or not depending on the associated p-value. 
#However, the rejected covariates are not given any kind of consideration once the p-value 
#is lesser than the threshold. This may lead to discarding the covariates even if they 
#have some say on the regressand. 
dev.off()

#The overfitting problem
library(RSADBE)  
data(OF)
str(OF)
plot(OF$X,OF$Y,"b",col="red",xlab="X",ylab="Y");
#Both +ve and -ve slopes linearly, It apears that the output Y depends upon higher order of covariate X
#Let us fit polynomial curves of various degrees and understand the behavior of the different linear regression models.
#Terms X,X^2,X^3 are treated as different variables
#X1= X ,X2 = X^2,..., Xk=X^k
#We use option poly of lm to fit the polynomial regression models

lines(OF$X,lm(Y~poly(X,1,raw=TRUE),data=OF)$fitted.values,"b",col="green")
lines(OF$X,lm(Y~poly(X,2,raw=TRUE),data=OF)$fitted.values,"b",col="wheat")
lines(OF$X,lm(Y~poly(X,3,raw=TRUE),data=OF)$fitted.values,"b",col="yellow")
lines(OF$X,lm(Y~poly(X,6,raw=TRUE),data=OF)$fitted.values,"b",col="orange")
lines(OF$X,lm(Y~poly(X,9,raw=TRUE),data=OF)$fitted.values,"b",col="black")

#enhance the legend
legend(3,50,c("Poly 1","Poly 2","Poly 3","Poly 6","Poly 9"),
       col=c("green","wheat","yellow","orange","black"),pch=1,ncol=3)

#Initialise the following vectors
R2<-NULL;AdjR2<-NULL;FStat<-NULL;Mvar<-NULL;PolYOrder<-1:9
temp1<-NULL
#Now,fit the regression models beginning with order 1 up to 9 and extract their R2,AdjR2, F-Statistic value, and model variability
for (i in 1:9) {
  temp<-summary(lm(Y~poly(X,i,raw=T),data=OF))
  R2[i]<-temp$r.squared
  AdjR2[i]<-temp$adj.r.squared
  FStat[i]<-as.numeric(temp$fstatistic[i])
  Mvar[i]<-temp$sigma
}
cbind(PolYOrder,R2,AdjR2,FStat,Mvar)

as.numeric(lm(Y~poly(X,i,raw=T),data= OF)$coefficients)
as.numeric(lm(Y~poly(X,1,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,2,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,3,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,4,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,5,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,6,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,7,raw=T),data=OF)$coefficients)
as.numeric(lm(Y~poly(X,8,raw=T),data=OF)$coefficients)

#Fitting higher order polynomial curves gives a closer approximation of the fit.
#As the polynomial degree increases then the coefficient also increases in magnitude

#Regression Spline and ridge
------------------------------
#We will begin with a piecewise linear regression model and then consider the polynomial regression extension. The term spline refers to a thin strip of wood that can be easily bent along a curved line
  
#Bbasis Functions
#In many applications, it has been found that the transformed variables are more important 
#than the original variable itself. Thus, we need a more generic framework to consider the 
#transformations of the variables.Such a framework is provided by the basis functions 

#For a single covariate X, the set of transformations may be defined as follows:
#f(X)=sum(m=1 to M)(βm*hm(X))

#Here, hm(X) is the m-th transformation of X, and βm is the associated regression coefficient. 
#In the case of a simple linear regression model, we have h1(X)=1 and h2(X)=X . 
#For the polynomial regression model, we have hm(X)=X^m ,m=1,2,...,k , and for the logarithmic transformation h(X)=logX .

#Piece wise linear regression model
data(PW_Illus);
attach(PW_Illus);
plot(X,Y);

#It is apparent from the scatterplot display that a linear relationship between the x- and y-values 
#over the real line intervals less than 15, between 15 to 30, and greater than 30 is appropriate.
#In this particular case, we have a two-piece linear regression model.

#In general, let xa and xb denote the two points, where we believe the linear regression model has 
#the breakpoints. Further more, we denote an indicator function by Ia to represent that it equals 1 
#when the x value is greater than xa and takes the value 0 in other cases. 
#Similarly, the second breakpoint indicator Ib  is defined 
#The piecewise linear regression model is defined as follows:
  
Y=β0+β1X+β2(X-xa)Ia+β3(X-xb)Ib+error

#Calculate break points; so we select a range
break1 <- X[which(X>=12 & X<=18)] 
break2 <- X[which(X>=27 & X<=33)]

n1<-length(break1) : Candidates for candidates for being the breakpoints 
n2<-length(break2);

#We do not have a clear defining criterion to select one of the n1 or n2 x values to be the 
#breakpoints. Hence, we will run various linear regression models and select that pair of points 
#(xa, xb) to be the breakpoints, which return the least mean residual sum of squares.
#Towards this, we set up a matrix, which will have three columns with the first two columns 
#for the possible potential pair of breakpoints, and the third column will contain the mean residual sum of squares.

MSE_MAT <- matrix(nrow=(n1*n2), ncol=3);
colnames(MSE_MAT) = c("Break_1","Break_2","MSE");
curriter=0
for(i in 1:n1){
  for(j in 1:n2)  { curriter=curriter+1 
  MSE_MAT[curriter,1]<-break1[i] 
  MSE_MAT[curriter,2]<-break2[j]
  piecewise1 <- lm(Y ~ X*(X<break1[i])+X*(X>=break1[i] & X<break2[j])+X*(X>=break2[j]))
  MSE_MAT[curriter,3] <- as.numeric(summary(piecewise1)[6]) #Gives SSE 
  }
}

#To find pair of breakpoints
MSE_MAT[which(MSE_MAT[,3]==min(MSE_MAT[,3])),]
#That breakpoint which have minimum RMS error
plot(X,Y);

#Fit the piecewise linear regression model with breakpoints at (14,30) with 
pw_final <- lm(Y ~ X*(X<14)+X*(X>=14 & X<30)+X*(X>=30))

#Add the fitted values to the scatter plot with 
points(PW_Illus$X,pw_final$fitted.values,col ="red")

#The piecewise linear regression model shows a useful flexibility, and it is indeed a very useful model when there is a genuine reason 
#to believe that there are certain breakpoints in the model. This has some advantages and certain limitations too. 
#From a technical perspective,  the model is not continuous, whereas from an applied perspective, the model possesses problems in making guesses about the breakpoint values 
#and also the problem of extensions to multi-dimensional cases.


####################################
#####Ridge Regression in R
####################################

#Ridge Regression is a regularization method that tries to avoid overfitting, penalizing large 
#coefficients through the L2 Norm. For this reason, it is also called L2 Regularization.

#In a linear regression, in practice it means we are minimizing the RSS (Residual Sum of Squares) 
#added to the L2 Norm. Thus, we seek to minimize:
  
RSS(β)+ λ∑(j=1 to p)β^2j

where λ is the tuning parameter, βj are the estimated coefficients, existing p of them.

#To perform Ridge Regression in R, we will use the glmnet package
require(glmnet)
# Data = considering that we have a data frame named dataF, with its first column being the class
x <- as.matrix(dataF[,-1]) # Removes class
y <- as.double(as.matrix(dataF[, 1])) # Only class


# Fitting the model (Ridge: Alpha = 0)
set.seed(999)
cv.ridge <- cv.glmnet(x, y, family='binomial', alpha=0, parallel=TRUE, standardize=TRUE, type.measure='auc')

# Results
plot(cv.ridge)
cv.ridge$lambda.min
cv.ridge$lambda.1se
coef(cv.ridge, s=cv.ridge$lambda.min)

#In the above code, we execute logistic regression (note the family=’binomial’), in parallel (if a cluster or cores have been previously allocated), 
#internally standardizing (needed for more appropriate regularization) and wanting to observe the results of AUC (area under ROC curve). 
#Moreover, the method already performs 10-fold cross validation to choose the best λλ.

#############################
############Lasso in R
#############################
#Lasso is also a regularization method that tries to avoid overfitting penalizing large coefficients, 
#but it uses the L1 Norm. For this reason, it is also called L1 Regularization.

#This method has as great advantage the fact that it can shrink some of the coefficients to exactly 
#zero, performing thus a selection of attributes with the regularization.

#In a linear regression, in practice for the Lasso, it means we are minimizing the RSS 
#(Residual Sum of Squares) added to the L1 Norm. Thus, we seek to minimize:
  
  RSS(β)+λ∑j=1p|βj|

#where λ is the tuning parameter, βj are the estimated coefficients, existing p of them.

require(glmnet)
# Data = considering that we have a data frame named dataF, with its first column being the class
x <- as.matrix(dataF[,-1]) # Removes class
y <- as.double(as.matrix(dataF[, 1])) # Only class

# Fitting the model (Lasso: Alpha = 1)
set.seed(999)
cv.lasso <- cv.glmnet(x, y, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')

# Results
plot(cv.lasso)
plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
cv.lasso$lambda.min
cv.lasso$lambda.1se
coef(cv.lasso, s=cv.lasso$lambda.min)








