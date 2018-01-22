#links
#http://www.analyticsvidhya.com/blog/2015/12/faster-data-manipulation-7-packages/
#http://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/
#http://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
#http://www.cookbook-r.com/Graphs/Plotting_distributions_(ggplot2)/
#http://www.analyticsvidhya.com/blog/2016/03/questions-ggplot2-package-r/#ten
#http://www.analyticsvidhya.com/blog/2015/04/comprehensive-guide-data-exploration-r/
#http://www.sharpsightlabs.com/data-analysis-example-r-supercars-part2/
#http://www.analyticsvidhya.com/blog/2015/05/data-visualization-resource/
#http://www.milanor.net/blog/preparing-the-data-for-modelling-with-r/
##Refer the excel for different data exploration for univariate and bivariate analysis
#http://rfunction.com/archives/1302
#https://www.youtube.com/watch?v=HeqHMM4ziXA
#1. Access the UCI machine learning repository: http://archive.ics.uci.edu/ml/.


#Categorical/Qualitative/Discrete --> Nominal and Ordinal Variables
#Numeric/Quantitative/Continuous --> Interval and Ratio Variables
#Independent Variables/Predictors/Features/ - same
#Dependent/Target/ - same

#Types of Variables
#Nominal
#Ordinal
#Interval Variables
#Ratio Variables

#Many statistics books begin by defining the different kinds of variables you might want to analyze. This scheme was developed by Stevens and published in 1946.
#A categorical variable, also called a nominal variable, is for mutual exclusive, but not ordered, categories. For example, your study might compare five different genotypes. You can code the five genotypes with numbers if you want, but the order is arbitrary and any calculations (for example, computing an average) would be meaningless.
#A ordinal variable, is one where the order matters but not the difference between values. For example, you might ask patients to express the amount of pain they are feeling on a scale of 1 to 10. A score of 7 means more pain that a score of 5, and that is more than a score of 3. But the difference between the 7 and the 5 may not be the same as that between 5 and 3. The values simply express an order. Another example would be movie ratings, from * to *****.
#A interval variable is a measurement where the difference between two values is meaningful. The difference between a temperature of 100 degrees and 90 degrees is the same difference as between 90 degrees and 80 degrees.
#A ratio variable, has all the properties of an interval variable, and also has a clear definition of 0.0. When the variable equals 0.0, there is none of that variable. Variables like height, weight, enzyme activity are ratio variables. Temperature, expressed in F or C, is not a ratio variable. A temperature of 0.0 on either of those scales does not mean 'no heat'. However, temperature in Kelvin is a ratio variable, as 0.0 Kelvin really does mean 'no heat'. Another counter example is pH. It is not a ratio variable, as pH=0 just means 1 molar of H+. and the definition of molar is fairly arbitrary. A pH of 0.0 does not mean 'no acidity' (quite the opposite!). When working with ratio variables, but not interval variables, you can look at the ratio of two measurements. A weight of 4 grams is twice a weight of 2 grams, because weight is a ratio variable. A temperature of 100 degrees C is not twice as hot as 50 degrees C, because temperature C is not a ratio variable. A pH of 3 is not twice as acidic as a pH of 6, because pH is not a ratio variable.

#OK to compute....  Nominal	Ordinal	Interval	Ratio
#frequency distribution.	Yes	Yes	Yes	Yes
#median and percentiles.	No	Yes	Yes	Yes
#add or subtract.	No	No	Yes	Yes
#mean, standard deviation, standard error of the mean.	No	No	Yes	Yes
#ratio, or coefficient of variation.	No	No	No	Yes


#The factor() function also allows you to assign an order to the nominal variables, thus making them ordinal variables. 
#This is done by setting the order parameter to TRUE and by assigning a vector with the desired level hierarchy to the argument levels

library() #To list all available packages
data() #To list all available datsets
data(package='package name') #To list all data sets in a given package
data('datset name') #To read in a datset




########Below are the steps involved to understand########
#clean and prepare your data for building your predictive model:

1.Variable Identification : Target vs Input 
2.Univariate Analysis : We explore variables one by one. Method to perform uni-variate analysis will depend on whether the variable type is categorical or continuous. 
Continuous Variables:- We need to understand the central tendency and spread of the variable. These are measured using various statistical metrics visualization   
Univariate analysis is also used to highlight missing and outlier values. 
Categorical variables, we’ll use frequency table to understand distribution of each category. We can also read as percentage of values under each category.

3.Bi-variate Analysis:We can perform bi-variate analysis for any combination of categorical and continuous variables. The combination can be: Categorical & Categorical, Categorical & Continuous and Continuous & Continuous.
Continuous & Continuous: Scatterplot (Corelation factor)
Categorical & Categorical : Two-way table, Stacked Column Chart, Chi Square Test
Categorical & Continuous : Box plot, Z test, T test, ANOVA

4. Missing values treatment: Deletion(omit), Mean/Mode/Median Imputation, Binning, KNN Imputation
5. Outlier treatment : 
  Can be : Univariate or Multivariate, 
Detect by : Using Boxplot, Histogram, Scatter Plot, 
Removing Outliers: Deleting Observations, Transforming and binning values, Imputation
6. Variable transformation
7. Variable creation : Dummy variables like categorical to numeric, Derived variables
#########

During this phase, there are many processes, such as data parsing, sorting, merging, filtering, 
missing value completion, and other processes to transform and organize the data, and enable it 
to fit into a consume structure. Later, the mapped data can be further utilized for data
aggregation, analysis, or visualization



# Data preparation using R
####STEP !##############
# Load Dataset
install.packages("rattle")
install.packages("dplyr")
install.packages("plyr")
install.packages("tidyr")
install.packages("lubridate")
install.packages("gridExtra")
install.packages("xlsx")
install.packages("rJava")
install.packages("corrplot")

library(dplyr)
library(plyr)
library(gridExtra)
library(xlsx)
library(rattle)
library(rJava);

#http://www.r-statistics.com/2012/08/how-to-load-the-rjava-package-after-the-error-java_home-cannot-be-determined-from-the-registry/

Sys.setenv("JAVA_HOME"="")

#Using weather data from rattle package
dspath<-system.file("csv","weather.csv",package="rattle")
weather<-read.csv(dspath)

#Another method to load data
#data <- read.csv("abc.csv", header='TRUE', sep=",")

#Lets have a basic look at how our data looks
dim(weather);
names(weather);
str(weather);
summary(weather);


######STEP 2#############
# Load generic variables
#We will store the datset as the generic variable ds. 
#This makes our process generic and can be used as a template

dsname<- "weather";
ds<-get(dsname);
# Or directly use ds<-weather;

#####STEP 3#############
??tbl_df #Used to add extra columns to data frame, it helps when there is large
#dataset and we accidentally print the dataset on console
#######STEp 4 ##########
class(ds);

#ds<-tbl_df(ds)
#class(ds)

head(ds);
tail(ds);

ds[sample(nrow(ds),7),] #Sample of 7 rows

str(ds)
summary(ds)

##########STEP 5 #################
######Meta Data Cleansing#########

#1.Normalise Variable Names
#Convenient to map all variables to lower case
#Use normVarNames();

names(ds);
names(ds)<-normVarNames(names(ds));
names(ds);

# Review data type/format of each variable using sapply
sapply(ds, class);

#Observation: Date is a factor instead it should be date format
#We use lubridate to convert to date format

library(lubridate);
head(ds$date);
ds$date<-ymd(as.character(ds$date))
class(ds$date)

#Review of variable roles
#We are now in a position to identify the roles played by each variable;
#We need to identify what is target, what is id (irrelevant), which is risk variable
#Watch out for output variables as inputs to modelling

vars<-names(ds);
vars
target<-"rain_tomorrow";
risk<-"risk_mm";
id<-c("date","location");


#Identify all variables which are factors
factors<-which(sapply(ds[vars],is.factor))
factors<-names(which(sapply(ds[vars],is.factor)))


#length : Get or set the length of vectors (including lists) and factors, 
#and of any other R object for which a method has been defined.

length(vars)
numeric_int_fact <-setdiff(vars,ignore) #Ignoring ignore vectors from vars 
length(numeric_int_fact)
numeric_int <- setdiff(numeric_int_fact,factors) #Removing factors from numeric int factor
length(numeric_int)
numeric_int

##########################################################################
################Data Exploration of Numeric and Integer variables#########
##########################################################################

numeric_int_exploration<- as.data.frame(numeric_int)

clss_info <- sapply(ds[numeric_int], class); #clss_info a character vector 
numeric_int_exploration<-cbind(numeric_int_exploration,clss_info) #Add the above vector to data frame

#http://www.r-bloggers.com/using-apply-sapply-lapply-in-r/
#Getting count of total observations
#len<-apply(ds,2,function(x) length(x)) #Will give an integer
len<-sapply(ds[numeric_int],function(x) length(x))
#len<-sapply(ds,function(x) length(x))
#len<-lapply(ds,function(x) length(x)) #Will give a list 

numeric_int_exploration<-cbind(numeric_int_exploration,len) #Add the above vector to data frame

#identifying no of NA records in each column or for each variable
#NAs<-apply(ds,2,function(x) length(x[is.na(x)]))
NAs<-sapply(ds[numeric_int],function(x) sum(is.na(x)))
numeric_int_exploration<-cbind(numeric_int_exploration,NAs) #Add the above vector to data frame

#In R, a missing value is often noted with the "NA" symbol, which stands for not available.
#Most functions (such as mean or sum) may output NA while encountering an NA value in the
#dataset. Though you can assign an argument such as na.rm to remove the effect of NA, it is
#better to impute or remove the missing data in the dataset to prevent propagating the effect
#of the missing value.

#Plotting missing values
install.packages("Amelia");
require(Amelia);
missmap(ds,main="Missing map")
AmeliaView()

#Imputing missing vlaues
After detecting the number of missing values within each attribute, we have to impute the
missing values since they might have a significant effect on the conclusions that can be drawn
from the data.

#Mean of all columns
mean<-sapply(ds[numeric_int],function(x) mean(x,na.rm=TRUE))
median<-sapply(ds[numeric_int],function(x) median(x,na.rm=TRUE))
sd<-sapply(ds[numeric_int],function(x) sd(x,na.rm=TRUE))
var<-sapply(ds[numeric_int],function(x) var(x,na.rm=TRUE))
min_value<-sapply(ds[numeric_int],function(x) min(x,na.rm=TRUE))
max_value<-sapply(ds[numeric_int],function(x) max(x,na.rm=TRUE))
quant_25<-sapply(ds[numeric_int],function(x) quantile(x,probs=c(0.25),na.rm=TRUE))
quant_75<-sapply(ds[numeric_int],function(x) quantile(x,probs=c(0.75),na.rm=TRUE))

#Append all records column wise in data frame numeric_int_exploration
numeric_int_exploration<-cbind(numeric_int_exploration,mean) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,median) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,sd) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,var) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,min_value) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,max_value) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,quant_25) #Add the above vector to data frame
numeric_int_exploration<-cbind(numeric_int_exploration,quant_75) #Add the above vector to data frame

##########################################################################
################Data Exploration of Factors###############################
##########################################################################

length(factors)
factors<-setdiff(factors,ignore)
factors

factors_exploration<- as.data.frame(factors)
clss_info <- sapply(ds[factors], class); #clss_info a character vector 
factors_exploration<-cbind(factors_exploration,clss_info) #Add the above vector to data frame

NAs<-sapply(ds[factors],function(x) sum(is.na(x)))
factors_exploration<-cbind(factors_exploration,NAs) #Add the above vector to data frame

#Levels for each factor

levels<-sapply(ds[factors],function(x) length(levels(x)))
factors_exploration<-cbind(factors_exploration,levels) #Add the above vector to data frame

##########################################################################
################Data Cleaning###############################
##########################################################################
##########STEP 6######## CLEANING ########
# We want to ignore some variables that are irrelevant or inappropriate for modelling
#Ignorings IDs and risks variables/ Output variablesIn statistics, estimation refers to the process by which one makes inferences about a population, based on information obtained from a sample. Point Estimate vs. Interval Estimate. Statisticians use sample statistics to estimate population parameters./
ignore<-union(id,if(exists("risk")) risk)
ignore;

#We might also identify variable that has unique value for every observation.
#These are identifiers as well and if so we should ignore
(ids<-which(sapply(ds,function(x) length(unique(x))==nrow(ds))));
ignore<-union(ignore,names(ids))
ignore;

#Ignore variables with constant values
constants<-names(which(sapply(ds[vars],function(x) all(x==x[1L]))))
ignore<-union(ignore,constants);

#We remove any variables where all of the values are missing
#We will first count the no of missing values for each variable, 
mvc<-sapply(ds[vars],function(x) sum(is.na(x)))
class(mvc)
mvn<-names(which(mvc==nrow(ds))) #get name for that variable which has every value as missing values
mvn;
ignore<-union(ignore,mvn);

# Many Missing : Ignore those variables which have more than 50% missing values
mvn<-names(which(mvc>=0.5*nrow(ds)))
mvn;
ignore<-union(ignore,mvn);
ignore

#Ignore varibales with too many levels(factor variables), or group the levels into smaller no. of levels
high_factors<-which(sapply(ds[vars],is.factor)) #Tells all factor variables in data frame
high_factors;
lvls<-sapply(factors,function(x) length(levels(ds[[x]]))) #Get levels for each factor
(many=names(which(lvls>20)));
ignore<-union(ignore,many);
lvls

# Couldnt Complete
#pdf(file="plot.pdf",width=18)
#grid.table(numeric_int_exploration)
#gri()
#grid.table(factors_exploration)
#dev.off()


#Identify Correlated Variables
mc<-cor(ds[which(sapply(ds, is.numeric))], use="complete.obs") #Creates corelation matrix
mc
mc_df<-as.data.frame(mc)

#Find all numeric variables and then use coorelation to identify which numeric variables are highly correlated
#1.when we use complete.obs, we discard the entire row if an NA is present. 
#In my example, this means we discard rows 1 and 2. 
#However, pairwise.complete.obs uses the non-NA values when calculating the correlation between V1 and V2.

names(which(sapply(ds, is.numeric))) #Only Numeric variables

library("tidyr")
mc[upper.tri(mc,diag=TRUE)]<-NA #Set all values in upper.tri as NA
mc<-mc %>% abs() %>% data.frame() %>% mutate(var1=row.names(mc)) %>% 
  gather(var2,cor, -var1) %>% 
  na.omit()

#install.packages("PerformanceAnalytics")
#library(PerformanceAnalytics)
#chart.Correlation(mc)


#Removing variables which are highly correlated
ignore<-union(ignore,c("temp_3pm","pressure_9am", "temp_9am"))

#Clean- Remove the variables
#Once we have identified the variables to be ignored , we remove from them our list of variables to use
length(vars)
var<-setdiff(vars,ignore)
length(var)

#Remove missing target
dim(ds);
sum(is.na(ds[target]))
ds<-ds[!is.na(ds[target]),] #Remove rows having NA value for target

#Remove observations that have missing values na.omit()
ods<-ds
omit<-NULL
dim(ds[var]);
sum(is.na(ds[var]));
mo<-attr(na.omit(ds[var]),"na.action")
omit<-union(omit,mo)
if(length(omit)) ds<-ds[-omit,]
dim(ds)

cleaned_ds<-ds[var]

#Restore the dataset
ds<-ods
dim(ds);


#Export to Excel in different sheets
write.xlsx(numeric_int_exploration, file="filename.xlsx", sheetName="Cont_Var_Summary")
write.xlsx(factors_exploration, file="filename.xlsx", sheetName="Categ_Var_Summary", append=TRUE)
write.xlsx(mc, file="filename.xlsx", sheetName="Corr_Matrix", append=TRUE)

##########################################################################
##Above steps include process of summarising numeric/continuous , factors/categorical,
##and removing all ids, constants, NAs, coorelated variables###############
##We also identify our target variables####################################
#############################################################################

#DATA EXPLORATION FOR DIFFERENT TYPES OF VARIABLES#
1. Univariate Analysis
#At this stage, we explore variables one by one. Method to perform uni-variate analysis 
#will depend on whether the variable type is categorical or continuous

#Continuous Variables:- In case of continuous variables, we need to understand the central tendency 
#and spread of the variable
#Visualisation Methods : Histogram and Box Plot
#Univariate analysis is also used to highlight missing and outlier values.
#For categorical variables, we'll use frequency table to understand distribution of each category
#We can also read as percentage of values under each category. It can be be measured using two metrics
#Visualisation Methods : Bar plot

univarplot <- function(data.frame)
{
  df <- data.frame
  ln <- length(names(data.frame))
  for(i in 1:ln)
  {
    mname <- substitute(df[,i])
    if(is.factor(df[,i]))
    {
      plot(df[,i],main=names(df)[i])
    }
    else
    {
      boxplot(df[,i],main=names(df)[i])
      hist(df[,i],main=names(df)[i])
    }
  }
}

univarplot(cleaned_ds)

pdf(file="plot.pdf",width=12)
par(mfrow=c(3,3),mar=c(2,1,1,1)) #my example has 9 columns
univarplot(cleaned_ds)
dev.off()


##Refer the excel for different data exploration for univariate and bivariate analysis

##BIVARIATE ANALYSIS##

##Continuous vs Continuous
#Scatterplot appropriate to summarize the relnship btw 2 numeric variables

#Appending plots to one pdf file"
num.plots <- 5
my.plots <- vector(num.plots, mode='list')

for (i in 1:num.plots) {
  plot(i)
  my.plots[[i]] <- recordPlot()
}
graphics.off()

pdf('myplots.pdf', onefile=TRUE)
for (my.plot in my.plots) {
  replayPlot(my.plot)
}
graphics.off()


#Concatenation of String
p <- function(..., sep='') {
  paste(..., sep=sep, collapse=sep)
}

#scaplot function for scatter plot
scaplot <- function(data.frame)
{
  df <- data.frame
  ln <- length(names(data.frame))
  for (i in 1:ln)
  {
    for (j in 1:ln)
    {
      ##plot(num_df[,i],num_df[,j],xlab=names(num_df)[i],ylab=names(num_df)[j],cex=0.5,col=2)
      plot(df[,i],df[,j],main=p(names(df)[i],"vs",names(df)[j]),xlab=names(df)[i],ylab=names(df)[j],cex=0.5,col=2)    
    }
  }
}  

pdf(file="plot.pdf",width=10)
par(mfrow=c(3,3),mar=c(3,1,1,1)) #my example has 9 columns

scaplot(cleaned_ds[which(sapply(cleaned_ds, is.numeric))])
#or
pairs(cleaned_ds[which(sapply(cleaned_ds, is.numeric))])

dev.off()


#Correlation plot
correlations<-cor(cleaned_ds[which(sapply(cleaned_ds, is.numeric))])
library(corrplot)
corrplot(correlations,method="color")
corrplot(correlations,type="upper",method="color")
corrplot.mixed(correlations)
#method could be color,ellipse,square,number

#Scatter plot shows the relationship between two variable but does not indicates the strength of relationship amongst them. To find the strength of the relationship, we use Correlation. Correlation varies between -1 and +1.

#-1: perfect negative linear correlation
#+1:perfect positive linear correlation and 
#0: No correlation
#Correlation can be derived using following formula:

#Correlation = Covariance(X,Y) / SQRT( Var(X)* Var(Y)) --Pearson Coelation

#Categorcial vs Categorical Data
============================================================================
  # create a contingency table
  bar_plot <- function(data.frame)
  {
    df <- data.frame
    ln <- length(names(data.frame))
    for (i in 1:ln)
    {
      for (j in 1:ln)
      { 
        if(i!=j)
        {
          ##plot(num_df[,i],num_df[,j],xlab=names(num_df)[i],ylab=names(num_df)[j],cex=0.5,col=2)
          table1<-table(df[,i], df[,j]);
          barplot(table1,main=p(names(df)[i],"vs",names(df)[j]),xlab=names(df)[i],ylab=names(df)[j])    
        }
      }
    }
  }  

fac_df<-ds[which(sapply(ds[var], is.factor))]
bar_plot(fac_df)

##Categorical vs Numerical Variable
#======================================================
#Boxplot:We can compare the distribution of a numerical variable on the basis of a categorical variable
#


##########STEP 8 ##############
########SAVE DATASET

dsdate<-paste0("_",format(Sys.Date(),"%y%m%d"));
dsdate;
dsrdata<-paste0(dsname,dsdate,".RData");
save(ds,dsname,dspath,dsdate,target,risk,id,ignore,vars,nobs,omit,inputi,inputc,numi,numc,cati,catc,file=dsrdata)
load(dsrdata)

rattle()

#################Xda Package ##################
###############################################
###############################################
install.packages("devtools")
library(devtools)
install_github("ujjwalkarn/xda")
library(xda)
numSummary(ds);
charSummary(ds);
bivariate(ds,'rain_today','rain_tomorrow');
bivariate(ds,'rain_today','cloud_9am');
Plot(ds,'rain_today')
p=c()


