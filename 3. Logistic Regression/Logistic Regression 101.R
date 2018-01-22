#LOGISTIC REGRESSION
----------------------
- Dichotomous/Ordinal/ Binary output variable either 0 or 1, yes or no
- Error distribution is non normal
- High possibility of Heteroscedaticity
- IndependenT variables may be categorical or continuous
- Instead of ordinary least squares regression, maximum likelihood estimation is used in logistic regression
- As a result, the statistics of interest are Wald-Chi-square values instead of F or t-values.

install.packages("RSADBE")
library("RSADBE")
data(sat);
str(sat);
ls();
rm(list=ls());
Y=pass
X=Sat Score
plot(sat$Sat,sat$Pass,xlab='SAT Score',ylab="Final Result")
#Higher the sat score higher the chance of Pass
Fit Simple Linear Regression Frst
passlm<-lm(Pass~Sat,data=sat)
summary(passlm);
abline(passlm);
#make a prediction with SAT-M scores of 400 and 700
predict(passlm,newdata=list(Sat=400));
predict(passlm,newdata=list(Sat=700),interval="prediction");

#Overall p-value significant, Pr>|t| for Sat also very significant but the problem with model is that the predicted values can be both +ve or -ve but the output can be 0 or 1
#Probit Regression and Logistic Regression

Probit Regression Model
------------------------
Using latent variable also called auxillary random variable
Y*=X'b+e     ' same earlier linear regressio modelwith Y replaced by Y*.
Error term e is asumed to follow a normal distribution N(0,(var)^2)
Normal distribution : N(Mean,var)
If Mean=0 and Var=1 then Standard Normal distribution (Bell curve)

Y=1 if y*>0 otherwise Y=0
Method of Maximum Likelihood estimation is use dto detremine b. It uses log likelihood function

We use #glm# function# 

#Using the glm function and the binomial(probit) option we can find the probit model for Pass   as function of Sat score:

pass_probit<-glm(Pass~Sat, data=sat, binomial(probit))
?glm
summary(pass_probit);
install.packages("pscl")
library(pscl);
pR2(pass_probit); #Gives pseudo R2 value
predict(pass_probit,newdata=list(Sat=700),type="response");

#Summary
Pr(>|z|) for Sat has very significant say in explaining whether the student successfully completes the course or not
Pseudo R2 value (McFadden metric) indicates that approx 39.34 % of the output is explained by Sat variable. This suggests that we need to collect more information about the students

?predict.glm
type="response", type="terms", type="link"

#Logistic regression
We use log of odds as dependent variable, when the outcome variable is categorical
It simply, predicts the probability of occurence of an event by fiiting data to logit function

g(y)=b0+b1(x)
Let g(y), the Link function be p ; probability of success or failure
p= Probabilty of success for observation is denoted by p
y=b0+b1x1+b2x2....
p=(e^y)/1+(e^y)--  Logit Function - 1
q=1-p=1-(e^y)/1+(e^y) -- Probabilty of Failure - 2
divide 1 by 2 and take log on both sides 
log(p/(1-p))=y : 
  
  logit(P(Y=1|X1,..,Xp))=log(P(Y=1|X1,..,Xp)1−P(Y=1|X1,..,Xp))=β0+β1X1+..+βpXp
The unknown model parameters β0,β1,..,βp are ordinarily estimated by maximum likelihood

#The above function is the link function. 
#Logarithmic transformation on the outcome avriable allows us to model a non linear association in a linear way
#p/1-p is the odd ratio

?glm
family=binomial for glm function 
pass_logistic<-glm(Pass~Sat,data=sat,family="binomial")
summary(pass_logistic);

Deviance: is measure useful for assesing goodnes of fit and is analogous to residual sum of squares
Null Deviance : when there are no covariates/regressors; like empty model so high deviance
Residual Deviance: RD when variable is used and has large impact on the the regressand; so the residuals of such a fitted model will be much less than null deviance

Find pseudo R2 : 
  pR2(pass_logistic)
summary.glm(pass_logistic)

The overall model significance is obtained with
with(pass_logistic,pchisq(null.deviance-deviance,df.null-df.residual,lower.tail=FALSE))
#pchisq gives p value
confint(pass_logistic)
predict.glm(pass_logistic,newdata=list(Sat=400,type="response"))

#Hosmer- Lemeshow goodness of fit test statistic for logistic model
----------------------------------------------------------
install.packages("ResourceSelection")
library("ResourceSelection")
h1<-hoslem.test(pass_logistic$y,fitted(pass_logistic),g=10)
h1


#ROC Curves : Receiving Operator Curves

