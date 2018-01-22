#http://rstudio-pubs-static.s3.amazonaws.com/2899_a9129debf6bd47d2a0501de9c0dc583d.html


Multiple Linear Regression
---------------------------
  y<-c(1,5,3,8,5,3,10,7);
  x1<-c(2,4,5,6,8,10,11,13);
  x2<-c(1,2,2,4,4,4,6,6);
  plot(x1,y);
  plot(x2,y);
  plot(x1,x2);
  mod1<-  lm(y~x1);
  mod2 <- lm(y~x2);
  mod3 <- lm(y~x1+x2);
  
  install.packages("memisc")
  library("memisc")
  require("mtable")
  mtable(mod1,mod2,mod3);
  
  data(package='RSADBE');
  library("RSADBE")
  data(Gasoline)
  str(Gasoline);
  gasoline_lm <- lm(y~.,data=Gasoline);
  summary(gasoline_lm);
  
  #Summary
  1. Overall model is significant as p value of F statistic is less than alpha(0.05)
  2. Only x1 and x3 are the most significant of all 11 variables
  3. x11 variable is the factor with value A and M only (Categorical Variable)
  4. X11M (It already splits an m-level factor variable is used to create m-1 new different variables.)

  
  gasoline_anova<-anova(gasoline_lm);
  gasoline_anova
  confint(gasoline_lm)
  
#Useful Residual Plots
#Types of Modified Residuals
#In the context of multiple linear regression models, modifications of the residuals have been
#found to be more useful than the residuals themselves. Residuals are assumed to have mean of 0 and unknown variance i.e. follow a normal distribution
# Mean Residual Sum of Squares: Sum of squares of residuals/ (n-p)
# n: Total Data points ; p: number of covariates in model
  
#Standardised Residuals: Check standardized residuals to check for normality assumptions for the residuals
#R-student Residuals: Especially used for the detection of outliers
  
#MSE of regression model 
# Useful Residual Plots
gasoline_lm_mse<-gasoline_anova$Mean[length(gasoline_anova$Mean)] #This is variance
stan_resid_gasoline <- resid(gasoline_lm)/sqrt(gasoline_lm_mse) #Standardising Residuals
#Standardizing the residuals
studentized_resid_gasoline <- resid(gasoline_lm)/(sqrt(gasoline_lm_mse*(1-hatvalues(gasoline_lm))))
#Studentizing the residuals
pred_resid_gasoline <- rstandard(gasoline_lm)
pred_student_resid_gasoline<-rstudent(gasoline_lm)
# returns the R-Student Prediction Residuals
par(mfrow=c(2,2))
plot(gasoline_fitted,stan_resid_gasoline,xlab="Fitted",ylab="Residuals")
title("Standardized Residual Plot")
plot(gasoline_fitted,studentized_resid_gasoline,xlab="Fitted",ylab="Residuals")
title("Studentized Residual Plot")
plot(gasoline_fitted,pred_resid_gasoline,xlab="Fitted",ylab="Residuals")
title("PRESS Plot")
plot(gasoline_fitted,pred_student_resid_gasoline,xlab="Fitted",ylab="Residuals")
title("R-Student Residual Plot")

#To know the reason for outlier (either in x or in y)
#Leverage Points and Influential points
#---------------------------------------
  
hatvalues(gasoline_lm);
length(gasoline_lm$coefficients)
cooks.distance(gasoline_lm)  #If grreater than 1 than influential
  
  
Multicollinearity Problem
---------------------------
#Covariates or independent variables or predictors are not linearly independent 
#then multicollinear
#Check the correlation of one variable with another (only numeric)
#The linear independence here is the sense of Linear Algebra that a vector 
#(covariate in our context) cannot be expressed as a linear combination of others.
  
If MCP then
- Wrong value of Regression coefficients
- Relevant factors cant be identified via t test or F test
- The importance of certain predictors will be undermined

mc<-round(cor(Gasoline[,-c(1,12)]),2);
install.packages("corrplot")
library(corrplot)
#http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
#Higher the correlation means there is a problem of MCP
corrplot(cor,method = "number")

install.packages("tidyr")
install.packages("dplyr")
library(dplyr)
library(tidyr)
mc[upper.tri(mc,diag=TRUE)]<-NA #Set all values in upper.tri as NA
mc<-mc %>% abs() %>% data.frame() %>% mutate(var1=row.names(mc)) %>% 
  gather(var2,cor, -var1) %>% na.omit()  

#We find the VIF (Variance Inflation Factor)
#Fucntion available in car and faraway package
library(car)
install.packages("faraway");
library(faraway);

#vif should be leass tha 5 or 10 for the covariate to be linearly independent

vif(Gasoline[,-c(1,12)]);
#x3 has highest, so remove x3
vif(Gasoline[,-c(1,4,12)])
#x10 has highest VIF so remive it and run again
vif(Gasoline[,-c(1,4,11,12)]);
vif(Gasoline[,-c(1,2,4,11,12)]);
vif(Gasoline[,-c(1,2,3,4,11,12)]);
  
#Now remove all the variables with VIF greater than 10
#Final Model
  
summary(lm(y~x4+x5+x6+x7+x8+x9,data=Gasoline));
  
#vif fucntion from car can be direstly applied to the lm object
  
#Model Selection
--------------------
-Stepwise Procedures
-Criterion based Procedures

#1. Stepwise Procedures
-------------------------
#3 methods of selecting co variates for inclusion in the final model
  a.  backward elimination
  b.  forward selection
  c.  stepwise regresion
  
1. Backward Elimination : 
  - Begin with all covariates
  - Eliminate that covariate whose p value is maxm among all covariates having p value greater than alpha(0.05)
  - Refit the model again and continue until all the covariates whose p value < 0.05
  
2. Forward Selection :
  - Begin with an empty model
  - For each covariate, obtain the p value if it is added to the model. Select the covariates with the least p value among all the covariates
  whose p value is lesser than 0.05
  - Repat until no more covariates can be updated for the model
  
best way use AIC (Akaike Information Criteria)
#Example
  ------------
#1. Create function pvalueslm which exracts the p values rlated to all covariates of an lm object
  
pvalueslm<-function(lm) {summary(lm) $coefficients[-1,4]}
  
#2. Create backwardlm function
    backwardlm <-function(lm,criticalalpha) {
    lm2=lm
    while(max(pvalueslm(lm2))>criticalalpha) {
      lm2=update(lm2,paste(".~.-",attr(lm2$terms,"term.labels")[(which(pvalueslm(lm2)==max(pvalueslm(lm2))))],sep=""))
    }
    return(lm2)
  }
  
#Our goal is to iterate until we have any more covariates with p value greater than 0.5
attr(gasoline_lm$terms,"term.labels") #Extract all covariate names
summary(gasoline_lm)$coefficents[-1,4]
(which(pvalueslm(lm2)==max(pvalueslm(lm2)))) #it identifies covariate number which has maximum p value greater than alpha
paste(".~.-",attr(),sep="" #returns the formula which have removed the unwanted covariate
        
3. Obtain the efficicient linear regression model by applying the backwardlm function with critical alpha 0.20 on the data
gasoline_lm_backward<-backwardlm(gasoline_lm,criticalalpha=0.20)
        
4. summary(gasoline_lm_backward)
    

#Forwardlm
forwardlm<-function(y,x,criticalalpha){
yx<-data.frame(y,x)
mylm<-lm(y~-., data=yx)
avail_cov<-attr(mylm$terms,"dataClasses")[-1]
minpvalues<-0
while(minpvalues<criticalalpha) {
pvalues_curr<-NULL
            for(i in 1:length(avail_cov)){
              templm<-update(mylm,paste(".~.+",names(avail_cov[i])))
              mypvalues<-summary(templm)$coeficients[,4]
              pvalues_curr<-c(pvalues_curr,mypvalues[length(mypvalues)])
            }
            minpvalues<-min(pvalues_curr)
            if(minpvalues<criticalalpha){
              include_me_in<-min(which(pvalues_curr<criticalaplha))
              mylm<-update(mylm,paste(".~.+",names(avail_cov[include_me_in])))
              avail_cov<-avail_cov[-include_me_in]
            }
          }
          return(mylm)
        }
        
5. gasoline_lm_forward<-forwardlm(Gasoline$y,Gasoline[,-1],criticalalpha=0.20)
6. summary(gasoline_lm_forward)

#AIC
step(gasoline_lm,direction="both")
#direction can be forward or backaward
      
#Linear regression model provides the best footing for the general regression problems
#When out variable is discrete,binary or multi category them lm fails