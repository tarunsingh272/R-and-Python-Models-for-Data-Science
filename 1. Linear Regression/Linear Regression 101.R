#################################################################################################
#There are a large number of methods available for fitting models to continuous variables, 
#such as a linear regression [e.g., Multiple Regression, General Linear Model (GLM)], 
#nonlinear regression (Generalized Linear/Nonlinear Models), 
#regression trees (see Classification and Regression Trees), 
#CHAID, Neural Networks, etc
###################################################################################################

#http://www.statisticssolutions.com/assumptions-of-linear-regression/
#http://docs.statwing.com/interpreting-residual-plots-to-improve-your-regression/


#Linear Regression
#For modelling reln btw two variables
# y : response variable
# x : predictor varaible
# Residuals : y[i]-y'[i]=e[i]
#y'=b0+b1x  #Method of least squares is used to choose values of b0 and b1 that minimizes
#the sum of square of the residual errors
#Residual Standard Error = Root-MSE

#This task can be easily accomplished by Least Square Method. It is the most common method used for 
#fitting a regression line. It calculates the best-fit line for the observed data by minimizing 
#the sum of the squares of the vertical deviations from each data point to the line. 
#Because the deviations are first squared, when added, there is no cancelling out between 
#positive and negative values.

#At the conclusion of the chapter, you will be able to build a regression model through the following steps:
Building a linear regression model and their interpretation
Validation of the model assumptions
Identifying the effect of every single observation, covariates, as well as the output
Fixing the problem of dependent covariates
Selection of the optimal linear regression model

#To understand the regression model, we begin with n pairs of observations (X1,Y1),(X2,Y2)..(Xn,Yn) with each 
#pair being completely independent of the other.
#The core assumptions of the model are listed as follows:
- All the observations are independent
- The regressand (predictors) depends linearly on the regressors(response variable)
- The errors are normally distributed, that is N (0,var)

mod<-lm(LungCap ~ Age) #First variable Y second x variable; predicting luncap on basis of Age
summary(mod)
attributes(mod) # All attributes
mod$coefficients
plot(Age,LungCap,main="Scatterplot");
abline(mod,col=2,lwd=3);#Adding regression line to plot
plot(mod$residuals)
plot(mod)

p-value significance
=======================
#http://blog.minitab.com/blog/adventures-in-statistics/how-to-interpret-regression-analysis-results-p-values-and-coefficients
The p-value for each term tests the null hypothesis that the coefficient is equal to zero (no effect). 
A low p-value (< 0.05) indicates that you can reject the null hypothesis. 
In other words, a predictor that has a low p-value is likely to be a meaningful 
addition to your model because changes in the predictors value are related to 
changes in the response variable.

A larger p-value suggests that changes in the predictor are not associated with changes in the response.
Typically, you use the coefficient p-values to determine which terms to keep in the regression model.
P value is a statistical measure that helps scientists determine whether or not their hypotheses are correct. P values are used to determine whether the results of their experiment are within the normal range of values for the events being observed. Usually, if the P value of a data set is below a certain pre-determined amount (like, for instance, 0.05), scientists will reject the "null hypothesis" of their experiment - in other words, theyll rule out the hypothesis that the variables of their experiment had no meaningful effect on the results

If your p value is lower than your significance value, congratulations - you've proved that it's highly likely that there is a correlation between the variables you manipulated and the results you've observed. If your p value is higher than your significance value, you can't say with confidence whether the results you observed were the result of pure chance or of your experimental manipulation.
By convention, scientists usually set the significance value for their experiments at 0.05, or 5 percent.
# This means that experimental results that meet this significance level have, at most, a 5% chance of being the result of pure chance. 
#In other words, there's a 95% chance that the results were caused by the scientists manipulation of experimental variables, rather than by chance. For most experiments, being 95% sure about a correlation between two variables is seen as "successfully" showing a correlation between the two.

In other words, when p < 0.05 we say that the results are statistically significant, meaning we have strong evidence to suggest the null hypothesis is false.

Regression Coefficients :
  =========================
  b0 = Constant
b1 = Slope of the predictor or coefficient

Criteria is to minimise the sum of squared errors (Also known as method of least squares)


Check Residuals before analysing the goodness of fit
=====================================================
  The bottom line is that randomness and unpredictability are crucial components of any regression model. 
If you don’t have those, your model is not valid.
Response = (Constant + Predictors) + Error
You shouldn’t be able to predict the error for any given observation. 
The non-random pattern in the residuals indicates that the deterministic portion (predictor variables) 
of the model is not capturing some explanatory information that is “leaking” into the residuals.
The residuals should fall in a symmetrical pattern and have a constant spread throughout the range


Goodness of fit for a linear Model
=================================
Linear regression calculates an equation that minimizes the distance between the fitted line and all of the data points. 
Technically, ordinary least squares (OLS) regression minimizes the sum of the squared residuals.

A model fits the data well if the differences between the observed values and the models predicted values are small and unbiased.
Before you look at the statistical measures for goodness-of-fit, you should check the residual plots. 
Residual plots can reveal unwanted residual patterns that indicate biased results more effectively than numbers.

R-Squared : Coefficient of Determination
=========
R-squared is a statistical measure of how close the data are to the fitted regression line.
R-squared between 0 and 100%
The higher the R-squared, the better the model fits your data
R-squared cannot determine whether the coefficient estimates and predictions are biased, which is why you must assess the residual plots.
R-squared can be misleading sometimes


#MULTIPLE LINEAR REGRESSION
=============================
  As a predictive analysis, the multiple linear regression is used to explain the relationship between one continuous dependent variable from two or more independent variables.
The independent variables can be continuous or categorical (dummy coded as appropriate).

Assumptions:
  -------------  
- Data must be normally distributed
- A linear relationship is assumed between the dependent variable and the independent variables.
- The residuals are homoscedastic and approximately rectangular-shaped.
- Absence of multicollinearity is assumed in the model, so that the independent variables are not too highly correlated.

Muliticollinearity
--------------------
- state of very high intercorrelations or inter-associations among the independent variables, leading to wrong statistical inferences
- variance inflation factors (VIF), which indicate the extent to which multicollinearity is present in a regression analysis. A VIF of 5 or greater indicates a reason to be concerned about multicollinearity.

Multiple linear regression analysis helps us to understand how much will the dependent variable change, when we change the independent variables.

There are 3 major uses for Multiple Linear Regression Analysis – 
(1) causal analysis, (2) forecasting an effect, (3) trend forecasting. 

#Causality vs Correlation 
A causes b : Cause and Effect : Causality
A and B happens at same time : Coorelated : May be some other reason

Adjusted R squared
------------------
R-squared cannot determine whether the coefficient estimates and predictions are biased, which is why you must assess the residual plots. However, R-squared has additional problems that the adjusted R-squared and predicted R-squared are designed to address.

Problem 1: Every time you add a predictor to a model, the R-squared increases, even if due to chance alone. It never decreases. Consequently, a model with more terms may appear to have a better fit simply because it has more terms.

Problem 2: If a model has too many predictors and higher order polynomials, it begins to model the random noise in the data. This condition is known as overfitting the model and it produces misleadingly high R-squared values and a lessened ability to make predictions.

The adjusted R-squared is a modified version of R-squared that has been adjusted for the number of predictors in the model. 
The adjusted R-squared increases only if the new term improves the model more than would be expected by chance. It decreases when a predictor improves the model by less than expected by chance. 
The adjusted R-squared can be negative, but it’s usually not.  
It is always lower than the R-squared.

Predicted R square
------------------
A key benefit of predicted R-squared is that it can prevent you from overfitting a model. As mentioned earlier, an overfit model contains too many predictors and it starts to model the random noise.
Because it is impossible to predict random noise, the predicted R-squared must drop for an overfit model.
The predicted R-squared doesn’t have to be negative to indicate an overfit model. If you see the predicted R-squared start to fall as you add predictors, even if they’re significant, you should begin to worry about overfitting the model.

In general, an F-test in regression compares the fits of different linear models.
Unlike t-tests that can assess only one regression coefficient at a time, the F-test can assess multiple coefficients simultaneously.

The F-test of the overall significance is a specific form of the F-test. It compares a model with no predictors to the model that you specify. 
A regression model that contains no predictors is also known as an intercept-only model.

Significant F-test for overall significance in the Analysis of Variance table.
If the P value for the F-test of overall significance test is less than your significance level, you can reject the null-hypothesis and conclude that your model provides a better fit than the intercept-only model.
Typically, if you don't have any significant P values for the individual coefficients in your model, the overall F-test won't be significant either. However, in a few cases, the tests could yield different results. For example, a significant overall F-test could determine that the coefficients are jointly not all equal to zero while the tests for individual coefficients could determine that all of them are individually equal to zero.

#############################################################################
#############################################################################
###################REVISION WITH CODE#####
#############################################################################

install.packages("RSADBE");
library("RSADBE")
data(package='RSADBE');
data(IO_Time)
attach(IO_Time)

#Identify Relation between CPU Time and No of IO Operations
#Identify coefficents of regressions so that SSE sum of square error is minimized

IO_lm<-lm(CPU_Time~No_of_IO); 
class(IO_lm)  #lm object created
summary(IO_lm);

#Question 1 : Is Model Significant: Check p value and F-statistic for overall model(last row)
#p-value(overall) < 0.05 then model is significant 

#Question 2 : Check whether the independent variable and intercept term are significant or not
#Higher the stars more significant the independent variable
#This coefficient has the interpretation that for a unit increase in the number of IOs; 
#CPU_Time is expected to increase by 0.04076

#Question 3 : Now that we know that the model, as well as the independent variable, are significant, 
#we need to know how much of the variability in CPU_Time is explained by No_of_IO. 
#The answer to this question is provided by the measure R^2 (Coefficient of Determination)
#A more robust explanation, which takes into consideration the number of parameters and observations,
#is provided by Adjusted R-squared which is 98.6 percent.

#QUestion 3 : Check Residual Plots: Should be normally distributed, median close to 0
#Median close to zero means standard normal distribution.

summary(CPU_Time);

#In regression 
#Null hypothesis : all regression coefficients are 0
#Alternative hypotheis : Atleast one coefficient is not 0

#The ANOVA technique gives the answer to the latter null hypothesis of interest.
#The R functions anova and confint respectively help obtain the ANOVA table
#and confidence intervals from the lm objects.
IO_anova <- anova(IO_lm)
IO_anova

#The ANOVA table confirms that the variable No_of_IO is significant indeed. Note
#the difference of the criteria for confirming this with respect to summary(IO_lm).
#In the former case, the significance was arrived at using the t-statistics and here we have
#used the F-statistic. Precisely, we check for the variance significance of the input variable.
#We now give the tool for obtaining confidence intervals.
#The difference between ANOVA and the summary of the linear model object is in the respectively 
# p-values reported by them as Pr(>F) and Pr(>|t|).

confint(IO_lm)

#ANOVA Test
#Analysis of Variance (ANOVA) is a statistical method used to test differences between two or more means.
#inferences about means are made by analyzing variance.
#ANOVA Is About Variation
The whole purpose of Analysis of Variance is to break up the variation into component parts, 
and then look at their significance. But theres a catch: in statistics, 
Variance (the square of Standard Deviation) is not an “additive” quantity—in other words, you can’t just add the variance of two subsets and use the total as the variance of the combination.


In ANOVA, we call the collection of factors we’re using to assess variation a “model.” 
A good model ensures that the Sum of Squares due to error (a la “Miscellaneous” household expenses) is relatively small compared to that due to the factors in our model. This is why the Sum of Squares attributed to the model factors is called Explained Variation and the Sum of Squares due to error is called the Unexplained Variation.
In ANOVA, the F-ratio is the tool we use to compare each of the sources of variation
In general, the higher the F-ratio—the ratio of the variation due to any component to the variation due to error—the more significant the factor is. In our budget analogy, these factors would be our more significant the household expense categories.
Value of Alpha = 0.05
To determine significance, compare the p-value for each factor with the alpha value (0.05). 
If the p-value is less than alpha, that factor is significant, because there's less than a 5% chance we'd see that factor's F value otherwise.  
If the p-value is greater than alpha, we reject the factor's significance.
The R-squared statistic is simply the percentage of variation that can be explained by the factors included in our model. The greater the significance of the factors we can find through analysis, the greater the R-squared value (which is really a reflection of a good model). 

Now vaildate the assumptions/model 
----------------------------------
- Regression function is not linear : Plot residuals vs regressors should not be linear
- Error terms do not have constant variance : Plot of residuals to check that error terms are random no pattern
- Plot the boxplot of residuals to check for outliers, remove them and rebuilt.
- Error terms are not normally distributed. Plot normal probability in which the predicted values are plotted 
  against observed values. If the values fall along a staright line the normality assumption for errors hold 
  true

###IMportant 
IO_lm_resid <- resid(IO_lm)
par(mfrow=c(3,2))
plot(No_of_IO, IO_lm_resid,main="Plot of Residuals Vs Predictor Variable",ylab="Residuals",xlab="Predictor Variable")
plot(No_of_IO, abs(IO_lm_resid), main="Plot ofAbsolute Residual Values Vs Predictor Variable",ylab="Absolute Residuals", xlab="PredictorVariable")
# Equivalently
plot(No_of_IO, IO_lm_resid^2,main="Plot of Squared Residual Values Vs Predictor Variable", ylab="Squared Residuals", xlab="PredictorVariable")
plot(IO_lm$fitted.values,IO_lm_resid, main="Plot of Residuals Vs Fitted Values",ylab="Residuals", xlab="Fitted Values")  
plot.ts(IO_lm_resid, main="Sequence Plot ofthe Residuals")
boxplot(IO_lm_resid,main="Box Plot of the Residuals")
rpanova = anova(IO_lm)
IO_lm_resid_rank=rank(IO_lm_resid)
tc_mse=rpanova$Mean[2]
IO_lm_resid_expected=sqrt(tc_mse)*qnorm((IO_lm_resid_rank-0.375)/(length(CPU_Time)+0.25))
plot(IO_lm_resid,IO_lm_resid_expected,xlab="Expected",ylab="Residuals",main="The Normal Probability Plot")
abline(0,1)


###### Already a function in R for plotting multiple plots####
plot(IO_lm)
