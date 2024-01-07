#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# This is my first challenge and first begginer data science project. I am very grateful for the opportunity to practice on the Kaggle platform and for the lot of knowledge from many users and creators of other solutions. I used many notebooks, but also many internet sources and books to create project and understand basic topics. I am a beginner in Python and data science, so I am open to any comments to improve my solution. 
# **I would also like to thank other users for sharing their solutions, notebooks and ideas on the discussion threads, which also helped me a lot in understanding many issues.**
# My solution summary:
# 1. EDA analysis - 1st Problem - I had problem with missing data in Age column I must repeat fill missing data.
# 2nd Problem - I detect outliers but I can't delete them, because at the end length of my data doesn't fit to submission. I need to read up on this problem.
# I think that dealing with outliers could be helpful with better model accuracy and reduce large kurtosis in few columns.
# 2. Feature engineering - I create some new features and assess by correlation which will be good for the next steps.
# 3. Standard Scaler to prepare data to modeling.
# 4. Modeling and hyperparameters tuning (I know I have a messy code but I tried a lot of models. I tried to asses the best parameters by create a function or
# I'm trying manually change values of hyperparameters.)
# 5. My summary of models with accuracy based on train data score (Decision Tree, Random Forest, SVC, Logistic Regression, KNN, Extra Trees Classifier,
# Ada Boost Classifier and Gradient Boost Classifier.) I created table with selected scores(I rejected the worse ones).
# 6. I choose two models Random Forest Classifier model and Extra Trees Classifier model and I asses them with Learning Curve. I also asses in
# Validation Curve impact of max depth in both models and I choose Random Forest Classifier (which I improved changing max depth value to 3,
# because there was some problems : I had higher train accuracy than test accuracy (the difference between these two measures was to high)).
# 7. I tried the improve model on data and make a prediction.
# 

# In[2]:


# imports
from scipy import stats
from scipy.stats import norm, kurtosis
from scipy.stats import skew
from scipy.stats import pearsonr
from collections import Counter

import sklearn
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, \
                            precision_score, \
                            recall_score, \
                            f1_score, \
                            mean_squared_error, \
                            mean_absolute_error

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.metrics import jaccard_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import VotingClassifier


# set seed for reproducibility
np.random.seed(0)
print("Added imports successfully.")


# In[3]:


print("Read the train set.\n")
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.head()


# In[4]:


print("Read the test set.\n")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()


# In[5]:


IDcolumn = test_data["PassengerId"]


# In[6]:


print("Get full data.")
def concatenate_data(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

full_data = concatenate_data(train_data, test_data)
full_data = pd.DataFrame(full_data)

#Copies
full_data_copy = full_data
train_copy = train_data
test_copy = test_data


# In[7]:


#My first step is knowing and understanding data.

print("Column names, data types\n")
train_data.info(), test_data.info()

print("Number of rows, number of columns\n")
train_data.shape, test_data.shape


# Number(int and float) and object types (string and other types) are observed in train data - it will be important when choosing an machine learning model and data visualization method. After reviewing the file, the types loaded by pandas seem correct as to the visible data in the dataframe

# In[8]:


test_data.shape


# In[9]:


#Set some useful variables
print("Set and present basic variables\n")
print("Train columns\n")
train_columns = train_data.columns
print(train_columns)
print("\nTest columns\n")
test_columns = test_data.columns
print(test_columns)


# In[10]:


# Useful function to calculate skewness - numeric features

print("Skewness function")

def skewness_calc(df, variables):
    for i, var in enumerate (variables):
        print(var, df[var].skew())


# Useful function to calculate skewness - numeric features

# In[11]:


# Useful function to calculate kurtosis - numeric features

print("Kurtosis function.")

def kurtosis_calc(df, variables):
    for i, var in enumerate (variables):
        print(var, df[var].kurtosis())


# Useful function to calculate kurtosis - numeric features

# In[12]:


print("IQR_method to detect outliers.")

def IQR_method_detect(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    outliers = df[((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    
    return outliers


# Function to detect outliers.

# I detect some outliers below but I can't delete them I described this problem at the beggining on this notebook.

# In[13]:


print("Function to create ROC curve and calculate AUC value and plot it.")

def ROC_AUC_calc(y_test_df, prediction_yhat):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_df, prediction_yhat)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


# In[14]:


def accuracy_validation_calcs(df_y_test, pred_yhat, df_X_train, df_y_train, model):
    
    print("Train set Accuracy: ", metrics.accuracy_score(df_y_train, model.predict(df_X_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(df_y_test, pred_yhat))
    print("Confusion matrix: ", confusion_matrix(df_y_test,pred_yhat))
    print("Precision score: ", precision_score(df_y_test,pred_yhat))
    print("Recall score: ", recall_score(df_y_test,pred_yhat))


# In[15]:


def cross_val_score_model(model, features, target):
    result_cross_val=cross_val_score(model,features,target,cv=10,scoring='accuracy')
    result_cross_val_mean=result_cross_val.mean()
    return result_cross_val_mean
    print('The cross val score is:',round(result_cross_val.mean()*100,2))


# In[16]:


#Analyze missing data
print("Analyze missing data - Train set")
#Column with missing data summarize
count_missing_train = train_data.isnull().sum()
percentage_missing_train = train_data.isnull().sum() * 100 / len(train_data)

print((count_missing_train), "\n Percentage result \n", (percentage_missing_train))


# Observed missing values in columns: Age, Cabin and Embarked. Were the Cabin column has 77% of missing values.

# In[17]:


print("Analyze missing data - Test set")

#Column with missing data summarize
count_missing_test = test_data.isnull().sum()
percentage_missing_test = test_data.isnull().sum() * 100 / len(test_data)

print((count_missing_test), "\n Percentage result \n", (percentage_missing_test))


# Observed missing values in columns: Age, Cabin and Fare. Also the Cabin colum has a high percentage (78%) of missing values.

# In[18]:


#Visualize character of missing data - Is it regular or random? Train set
ax = msno.matrix(train_data.sample(891))


# In[19]:


#Visualize character of missing data - Is it regular or random? Test set
ax = msno.matrix(test_data.sample(418))


# A picture of missing data without obvious regularities.

# In[20]:


#Dealing with missing data - decisions
print("Dealing with missing data.")
print("Delete Cabin column - train data")

train_data = train_data.drop(columns =['Cabin'], axis=1)
train_data.head()


# Decided to delete Cabin column both in train and test set, because in both situations high percentage (above 70%) observed. It will be very hard to fill these empty cells without real data, so "Cabin" column may deteriorate the quality of the model's conclusions

# In[21]:


print("Delete Cabin column - test data")

test_data = test_data.drop(columns =['Cabin'], axis=1)
test_data.head()


# In[22]:


print("Delete PassengerId column - train data")
train_data = train_data.drop(columns =['PassengerId'], axis=1)
train_data.head()


# In[23]:


print("Delete PassengerId column - test data")
test_data = test_data.drop(columns =['PassengerId'], axis=1)
test_data.head()


# In[24]:


#Analyze variable with unique values 'Ticket', counting duplicates
print("Analyze variable with unique values 'Ticket' in train set, counting duplicates")
len(train_data['Ticket'])-len(train_data['Ticket'].drop_duplicates())


# In[25]:


print("Train data shape:",(train_data.shape), "Test data shape:", (test_data.shape))


# In[26]:


#Analyze variable with unique values 'Ticket', counting duplicates
print("Analyze variable with unique values 'Ticket' in test set, counting duplicates")
len(test_data['Ticket'])-len(test_data['Ticket'].drop_duplicates())


# In[27]:


print("Train data shape:",(train_data.shape), "Test data shape:", (test_data.shape))


# In[28]:


print("Delete Ticket column - train data")
train_data = train_data.drop(columns =['Ticket'], axis=1)
train_data.head()


# Decided to drop Ticket column in both sets, because data about ticket classes are in Pclass. Also Ticket column has a lot of duplicates.

# In[29]:


print("Train data shape:",(train_data.shape), "Test data shape:", (test_data.shape))


# In[30]:


print("Delete Ticket column - test data")
test_data = test_data.drop(columns =['Ticket'], axis=1)
test_data.head()


# In[31]:


#Check 
print("Train data shape:",(train_data.shape), "Test data shape:", (test_data.shape))


# It's all correct. Why I deleted these columns:
# Ticket - has the same data about classes like Pclass - Pclass it will be
# easier to divide by one hot encoding. Also it has a lot of duplicates.
# Cabin - this column has to many missing values - it will be hard to fill it
# without influence on model fitting.
# PassengerId - It's just set of ordinal numbers it does not provide any
# important information.
# These columns are deleted in both test and train sets.

# In[32]:


# Dealing with missing data - Fare in test set - analyze feature distribution
print("Visualize Fare distribution in test set")
counts = test_data['Fare'].value_counts(dropna=False)
fig = plt.figure(figsize =(10, 7))
counts.plot.bar(title='feat1', grid=True)


# Distribution of this feature doesn't look like normal distribution - 
# skewness and kurtosis calculation below.

# In[33]:


#Skewness and Kurtosis for Fare in test set
print("Skewness and Kurtosis for Fare in test set")
Fare_test = test_data[['Fare']]

print("Fare data kurtosis - test set\n")        
kurtosis_calc(test_data, Fare_test)
print("\nTest data skewness - test set\n")
skewness_calc(test_data, Fare_test)


# Fare in test set has leptokurtic distribution (thin and tall distribution). That might mean that Fare values have outliers. Also high value of skewness observed, which coincides with the observation from the charts, that Fare distribution in test set is not symmetrical. 

# In[34]:


#If fare has ouliers? IQR
print("Interquartlie Range to detect outliers function")

outliers_Fare = IQR_method_detect(test_data['Fare'])
print("number of outliers: "+ str(len(outliers_Fare)))

print("max outlier value: "+ str(outliers_Fare.max()))

print("min outlier value: "+ str(outliers_Fare.min()))


# Fare in test set has 55 outliers - median should be used to fill missing values.

# In[35]:


#Fill Fare 1 missing value in test set with median

print("Dealing with missing value in Fare column in test set.")
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data.head()


# Fare filled with median because mean is sensitive to outliers.

# In[36]:


test_data.isnull().sum()


# No missing data in Fare column.

# In[37]:


#Dealing with missing data in train set - Embarked- categorical feature
print("Unique values in Embarked - train")
test_data['Embarked'].value_counts().plot(kind='bar')


# 3 values - one hot encoding will be useful to turn values on numeric variables.

# In[38]:


train_data['Embarked'].describe()


# In[39]:


#Dealing with missing values (2) in Embarked in train set
print("Filling 2 missing values in Embarked train set with most frequent value.")
train_data['Embarked']=train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0])
train_data.head()


# I decided to fill 2 missing values with most frequent value "S" - because it is categorical feature.

# In[40]:


train_data.isnull().sum()


# Now Age will be analyzed. In this case I decided to make some deeper analysis, because 177 missing values in train set and 86 missing in test set values observed.

# In[41]:


train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data.head()


# In[42]:


test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})


# Survived turned to numerical variable, because I want use it to calculate correlation with Age.

# In[43]:


# Dealing with missing data - Age in train set - analyze feature distribution
print("Visualize Age distribution in train set")
counts = train_data['Age'].value_counts(dropna=False)
fig = plt.figure(figsize =(10, 7))
counts.plot.bar(title='feat1', grid=True)


# In[44]:


# Dealing with missing data - Fare in test set - analyze feature distribution
print("Visualize Age distribution in test set")
counts1 = test_data['Age'].value_counts(dropna=False)
fig = plt.figure(figsize =(10, 7))
counts1.plot.bar(title='feat1', grid=True)


# Observed outliers in Age in test set - decided to fill missing values by median, because mean is sensitive to outliers.

# In[45]:


#Skewness and Kurtosis for Age in train set
print("Skewness and Kurtosis for Age in train set\n")

print("Age data kurtosis - train set\n")        
kurtosis_calc(train_data, train_data[['Age']])
print("\nAge data skewness - train set\n")
skewness_calc(train_data, train_data[['Age']])


# In[46]:


#Skewness and Kurtosis for Age in test set
print("Skewness and Kurtosis for Age in test set\n")

print("Age data kurtosis - test set\n")        
kurtosis_calc(test_data, test_data[['Age']])
print("\nAge data skewness - test set\n")
skewness_calc(test_data, test_data[['Age']])


# Data are moderately skewed.
# Low kurtosis in a data set is an indicator that data has lack of outliers.
# Despite this results I will use IQR method, because Age has a lot of missing values, it could have influence of this results.
# 

# In[47]:


IQR_method_detect(train_data['Age'])

outliers_Age = IQR_method_detect(train_data['Age'])
print("number of outliers: "+ str(len(outliers_Age)))

print("max outlier value: "+ str(outliers_Age.max()))

print("min outlier value: "+ str(outliers_Age.min()))


# In[48]:


IQR_method_detect(test_data['Age'])

outliers_Age_test = IQR_method_detect(test_data['Age'])
print("number of outliers: "+ str(len(outliers_Age_test)))

print("max outlier value: "+ str(outliers_Age_test.max()))

print("min outlier value: "+ str(outliers_Age_test.min()))


# Despite results using skewness and kurtosis values, IQR method shows 11 
# outliers. That's why I will be using median to fill missing values in Age. In test set only 2 outliers observed, but it could be remembered that test set 
# includes less data and it should be treated the same, because it will be
# represent bahavior of training set.

# In[49]:


#Finding correlations with Age, because Age has so many missing values
print("Correlation between numeric values vs Age - train set")
matrix_corrs = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
matrix_corrs


# In[50]:


print("Visualisation of correlation between numeric values vs Age - train set")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# Pclass(-0.37), SibSp(-0.31) - they are negatively correlated with Age (medium strength). I decided that other variables have to small strength of correlation with Age. I'll expand further analysis of SibSp and Pclass

# In[51]:


#Calculate the mean and median of Age it will be useful to get some comparisons to 
#find the best method to fill a lot of missing values in this column.
print("Age - Mean and Median - train set.")
print(train_data.Age.mean())
print(train_data.Age.median())


# In[52]:


print("Age - Mean and Median - test set.")
print(test_data.Age.mean())
print(test_data.Age.median())


# In[53]:


#Trying to find pattern to fill missing values in Age
age_groupby_pclass = train_data.groupby(['Pclass']).Age.agg([len, min, max, 'mean', 'median'])
age_groupby_pclass


# In[54]:


age_groupby_pclass_test = test_data.groupby(['Pclass']).Age.agg([len, min, max, 'mean', 'median'])
age_groupby_pclass_test


# In[55]:


#Trying to find pattern to fill missing values in Age
age_groupby_sibsp = train_data.groupby(['SibSp']).Age.agg([len, min, max, 'mean', 'median'])
age_groupby_sibsp


# In[56]:


age_groupby_sibsp_test = test_data.groupby(['SibSp']).Age.agg([len, min, max, 'mean', 'median'])
age_groupby_sibsp_test


# In[57]:


(train_data.groupby(['Pclass', 'SibSp']))['Age'].median()


# In[58]:


train_data.groupby(['Pclass', 'SibSp'])['Age'].describe()


# In[59]:


missing_Age = train_data[train_data['Age'].isnull()]
median_Age_grouped_train = train_data.groupby(['Pclass','SibSp'])['Age'].median()

def Age_filled(feature):
    if pd.isnull(feature['Age']):
        return median_Age_grouped_train[feature['Pclass'],feature['SibSp']]
    else:
        return feature['Age']

train_data['Age'] = train_data.apply(Age_filled, axis=1)


# In[60]:


train_data.isnull().sum()


# In[61]:


train_data['Age']=train_data['Age'].replace(np.nan, 26.0)
train_data.tail()


# I've tried different ways, functions but always I've got 7 missing values, so
# I fill these gaps with calculate median by replace method.

# In[62]:


train_data.isnull().sum()


# In[63]:


test_data['Age'] = test_data.apply(Age_filled, axis=1)
test_data.isnull().sum()


# In[64]:


test_data['Age'] = test_data['Age'].replace(np.nan, 26.0)
test_data.isnull().sum()


# In[65]:


categorical_var_train = train_data.select_dtypes(include=['object']).columns.tolist()
categorical_var_test = test_data.select_dtypes(include=['object']).columns.tolist()
numerical_var_train = train_data.select_dtypes(include=['int','float']).columns.tolist()
numerical_var_test = test_data.select_dtypes(include=['int', 'float']).columns.tolist()
print("\nCategorical features-train\n")
print(categorical_var_train)
print("\nNumerical features-train\n")
print(numerical_var_train)
print("\nCategorical features-test\n")
print(categorical_var_test)
print("\nNumerical features-test\n")
print(numerical_var_test)


# In[66]:


#Dealing with outliers - numeric features

print("Fare data kurtosis - train set\n")        
kurtosis_calc(train_data, numerical_var_train)


# Conclusions: SibSp, Parch, Fare have high value of kurtosis - these features will go to deeper outliers analysis.

# In[67]:


#Dealing with outliers - numeric features

print("Fare data kurtosis - test set\n")        
kurtosis_calc(test_data, numerical_var_test)


# Conclusions: SibSp, Parch, Fare have high value of kurtosis - these features will go to deeper outliers analysis.

# In[68]:


print("Visualize Fare distribution in train set")
counts = train_data['Fare'].value_counts(dropna=False)
fig = plt.figure(figsize =(10, 7))
counts.plot.bar(title='feat1', grid=True)


# In[69]:


print("Visualize SibSp distribution in train set")
counts = train_data['SibSp'].value_counts(dropna=False)
fig = plt.figure(figsize =(10, 7))
counts.plot.bar(title='feat1', grid=True)


# In[70]:


# Dealing with missing data - Fare in test set - analyze feature distribution
print("Visualize Parch distribution in train set")
counts = train_data['Parch'].value_counts(dropna=False)
fig = plt.figure(figsize =(10, 7))
counts.plot.bar(title='feat1', grid=True)


# In[71]:


print("Interquartlie Range to detect outliers in SibSp train set")

outliers_SibSp_train = IQR_method_detect(train_data['SibSp'])
print("number of outliers: "+ str(len(outliers_SibSp_train)))

print("max outlier value: "+ str(outliers_SibSp_train.max()))

print("min outlier value: "+ str(outliers_SibSp_train.min()))


# In[72]:


print("Interquartlie Range to detect outliers in Pclass train set")

outliers_SibSp_train = IQR_method_detect(train_data['Pclass'])
print("number of outliers: "+ str(len(outliers_SibSp_train)))

print("max outlier value: "+ str(outliers_SibSp_train.max()))

print("min outlier value: "+ str(outliers_SibSp_train.min()))


# Check if IQR method is right - in this case it is correct method to detect outliers.

# In[73]:


print("Interquartlie Range to detect outliers in SibSp test set")

outliers_SibSp_test = IQR_method_detect(test_data['SibSp'])
print("number of outliers: "+ str(len(outliers_SibSp_test)))

print("max outlier value: "+ str(outliers_SibSp_test.max()))

print("min outlier value: "+ str(outliers_SibSp_test.min()))


# In[74]:


print("Interquartlie Range to detect outliers in Parch train set")

outliers_Parch_train = IQR_method_detect(train_data['Parch'])
print("number of outliers: "+ str(len(outliers_Parch_train)))

print("max outlier value: "+ str(outliers_Parch_train.max()))

print("min outlier value: "+ str(outliers_Parch_train.min()))


# In[75]:


print("Interquartlie Range to detect outliers in Parch test set")

outliers_Parch_test = IQR_method_detect(test_data['Parch'])
print("number of outliers: "+ str(len(outliers_Parch_test)))

print("max outlier value: "+ str(outliers_Parch_test.max()))

print("min outlier value: "+ str(outliers_Parch_test.min()))


# In[76]:


print("Interquartlie Range to detect outliers in Fare train set")

outliers_Fare_train = IQR_method_detect(train_data['Fare'])
print("number of outliers: "+ str(len(outliers_Fare_train)))

print("max outlier value: "+ str(outliers_Fare_train.max()))

print("min outlier value: "+ str(outliers_Fare_train.min()))


# These calculations show that these features have problem with outliers in
# both sets. These outliers should be deleted. Fare test also has oulliers - it was calculated above.

# In[77]:


# OneHotEncoder
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Embarked'].values.reshape(-1, 1)))
train_data[['Embarked - Cherbourg', 'Embarked - Queenstown', 'Embarked - Southampton']] = pd.DataFrame(result, index = train_data.index)


train_data.head()


# In[78]:


result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Embarked'].values.reshape(-1, 1)))
test_data[['Embarked - Cherbourg', 'Embarked - Queenstown', 'Embarked - Southampton']] = pd.DataFrame(result, index = test_data.index)


test_data.head()


# In[79]:


result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Sex'].values.reshape(-1, 1)))
train_data[['Female', 'Male']] = pd.DataFrame(result, index = train_data.index)

train_data.head()


# In[80]:


result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Sex'].values.reshape(-1, 1)))
test_data[['Female', 'Male']] = pd.DataFrame(result, index = test_data.index)

test_data.head()


# In[81]:


print("One Hot Encoding for Pclass - better understanding of class (socio-economic status) meaning in model.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Pclass'].values.reshape(-1, 1)))
train_data[['1st class', '2nd class', '3rd class']] = pd.DataFrame(result, index = train_data.index)

train_data.head()


# In[82]:


result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Pclass'].values.reshape(-1, 1)))
test_data[['1st class', '2nd class', '3rd class']] = pd.DataFrame(result, index = test_data.index)

test_data.head()


# In[83]:


#Extracting title from name and use One Hot Encoding to turn it to numeric variable
print("Feature engineering")
print("Extracting Title from Name in train set.")
# create a new feature to extract title names from the Name column
train_data['Title'] = train_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
Dict_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
# map titles and view value frequency
train_data.Title = train_data.Title.map(Dict_titles)
print(train_data.Title.value_counts())


# Even though more women survived, there were more men in the ship. And 6 unique values observed - it will be One Hot Encoding in use.

# In[84]:


#Extracting title from name and use One Hot Encoding to turn it to numeric variable
print("Feature engineering")
print("Extracting Title from Name in test set.")
# create a new feature to extract title names from the Name column
test_data['Title'] = test_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
Dict_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
# map titles and view value frequency
test_data.Title = test_data.Title.map(Dict_titles)
print(test_data.Title.value_counts())


# In[85]:


# OneHotEncoder
print("Change Title to numerical features in train set.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Title'].values.reshape(-1, 1)))

train_data[['Master', 'Miss', 'Mr', 'Mrs', 'Officer', 'Royalty']] = pd.DataFrame(result, index = train_data.index)
train_data.head()


# In[86]:


# OneHotEncoder
print("Change Title to numerical features in test set.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Title'].values.reshape(-1, 1)))

test_data[['Master', 'Miss', 'Mr', 'Mrs', 'Officer', 'Royalty']] = pd.DataFrame(result, index = test_data.index)
test_data.head()


# In[87]:


#Combine Parch i SibSp - create new feature
print("Create Family Size - new feature by combining Parch i SibSp in train and test set")

train_data['Family Size'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['Family Size'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data.head()


# In[88]:


train_data['Is_Alone'] = 0
train_data.loc[train_data['Family Size'] == 1, 'Is_Alone'] = 1
train_data.head()


# In[89]:


test_data['Is_Alone'] = 0
test_data.loc[test_data['Family Size'] == 1, 'Is_Alone'] = 1


# In[90]:


# OneHotEncoder
print("Change Title to numerical features in test set.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Is_Alone'].values.reshape(-1, 1)))


test_data[['With someone', 'Alone']] = pd.DataFrame(result, index = test_data.index)
test_data.head()


# In[91]:


# OneHotEncoder
print("Change Title to numerical features in test set.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Is_Alone'].values.reshape(-1, 1)))


train_data[['With someone', 'Alone']] = pd.DataFrame(result, index = train_data.index)
train_data.head()


# In[92]:


print("Couples")
train_data['Is_Couple'] = 0
train_data.loc[train_data['Title'] == 'Mrs', 'Is_Couple'] = 1
#train_data['Is_Couple'] = train_data['Is_Couple'].loc[train_data['Title'] == 'Mrs'] = 1
train_data.head()


# In[93]:


print("Couples")
test_data['Is_Couple'] = 0
test_data.loc[test_data['Title'] == 'Mrs', 'Is_Couple'] = 1
#train_data['Is_Couple'] = train_data['Is_Couple'].loc[train_data['Title'] == 'Mrs'] = 1
test_data.head()


# In[94]:


# OneHotEncoder
print("Change Title to numerical features in test set.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Is_Couple'].values.reshape(-1, 1)))

test_data[['Couple', 'NotCouple']] = pd.DataFrame(result, index = test_data.index)
test_data.head()


# In[95]:


# OneHotEncoder
print("Change Title to numerical features in test set.")
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Is_Couple'].values.reshape(-1, 1)))

train_data[['Couple', 'NotCouple']] = pd.DataFrame(result, index = train_data.index)
train_data.head()


# In[96]:


bins = np.linspace(min(train_data['Age']), max(train_data['Age']), 4)
group_names = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
group_names = {'child': 0-18, 'adult': 19-60, 'senior': 60-79}

train_data['Age-binned'] = pd.cut(train_data['Age'], bins, labels=group_names, include_lowest=True )
train_data[['Age','Age-binned']].head()


# In[97]:


bins = np.linspace(min(test_data['Age']), max(test_data['Age']), 4)
group_names = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
group_names = {'child': 0-18, 'adult': 19-60, 'senior': 60-79}

test_data['Age-binned'] = pd.cut(test_data['Age'], bins, labels=group_names, include_lowest=True )
test_data[['Age','Age-binned']].head()


# In[98]:


# OneHotEncoder
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(train_data['Age-binned'].values.reshape(-1, 1)))

train_data[['Child', 'Adult', 'Senior']] = pd.DataFrame(result, index = train_data.index)
train_data.head()


# In[99]:


test_data.head()


# In[100]:


# OneHotEncoder for encoding Embarked in train and test set
result = pd.DataFrame(OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit_transform(test_data['Age-binned'].values.reshape(-1, 1)))
test_data[['Child', 'Adult', 'Senior']] = pd.DataFrame(result, index = test_data.index)


test_data.head()


# In[101]:


print("Delete unnecessary columns - train data.")

train_data = train_data.drop(columns =['Name'], axis=1)
train_data.head()


# In[102]:


print("Delete unnecessary columns - train data.")

train_data = train_data.drop(columns =['Embarked', 'Age-binned', 'Is_Alone', 'Is_Couple', 'NotCouple', 'Title'], axis=1)
train_data.head()


# As shown above train and test set still have categorical features. I deleted these column, because I changed both these columns values to numerical.

# In[103]:


print("Delete unnecessary columns - train data.")

test_data = test_data.drop(columns =['Name'], axis=1)


# In[104]:


print("Delete unnecessary columns - train data.")

test_data = test_data.drop(columns =['Embarked', 'Age-binned', 'Is_Alone', 'Is_Couple', 'NotCouple', 'Title'], axis=1)


# In[105]:


num_vares = train_data.select_dtypes(include=['int','float']).columns.tolist()


# In[106]:


num_vares_test = train_data.select_dtypes(include=['int','float']).columns.tolist()


# In[107]:


def ploters(variables, df):
    fig, axes = plt.subplots(10,3, figsize=(8, 30))
    flattened_axes = fig.axes
    

    for i, var in enumerate(variables):
        ax = flattened_axes[i]
        i = np.array([i])
        sns.regplot(x=var, y='Survived', data=df, ax=ax,)

    plt.show()


# In[108]:


#Analyze selected numerical features (without PassengerId)
print("Visualization to explore numerical features")


def multiple_histograms(df, features, rows, cols):
    figure=plt.figure()
    figure, axes = plt.subplots(10,3, figsize=(15, 15))
    flattened_axes = fig.axes
    
    for i, feat in enumerate(features):
        ax=figure.add_subplot(rows,cols,i+1)
        df[feat].hist(bins=10,ax=ax)
        ax.set_title(feat + "Distribution")

    plt.show()



# In[109]:


print("Visualise all features vs Survived feature.")
ploters(num_vares, train_data)


# In[110]:


#Survival rate
Survived = (sum(train_data['Survived'])/len(train_data['Survived'].index))*100
Survived


# 38% - value of Survived Rate

# In[111]:


basic_vars = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
title_vars = train_data[['Master', 'Miss', 'Mr', 'Mrs', 'Officer', 'Royalty', 'Survived']]
family_alone_vars = train_data[['Family Size', 'With someone', 'Alone', 'Couple', 'Survived']]
socio_economic_status_vars = train_data[['1st class', '2nd class', '3rd class', 'Survived']]
age_grouped_vars = train_data[['Child', 'Adult', 'Senior', 'Survived']]
gender_vars = train_data[['Female', 'Male', 'Survived']]
embarked_vars = train_data[['Embarked - Cherbourg', 'Embarked - Queenstown', 'Embarked - Southampton','Survived']]


# In[112]:


print("Correlation matrix - basic vars.")
matrix_corrs = basic_vars.corr()
matrix_corrs


# In[113]:


print("Correlation matrix - basic vars.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[114]:


print("Correlation matrix - title vars.")
print("Correlation between numeric values vs Age - train set")
matrix_corrs = title_vars.corr()
matrix_corrs


# In[115]:


print("Correlation matrix - title vars.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[116]:


print("Correlation matrix - family and alone vars.")
print("Correlation between numeric values vs Age - train set")
matrix_corrs = family_alone_vars.corr()
matrix_corrs


# In[117]:


print("Correlation matrix - family and alone vars.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[118]:


print("Correlation matrix - socio-economic status.")
matrix_corrs = socio_economic_status_vars.corr()
matrix_corrs


# In[119]:


print("Correlation matrix - socio-economic status.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[120]:


print("Correlation matrix - age grouped.")
print("Correlation between numeric values vs Age - train set")
matrix_corrs = age_grouped_vars.corr()
matrix_corrs


# In[121]:


print("Correlation matrix - age grouped.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[122]:


print("Correlation matrix - gender.")
matrix_corrs = gender_vars.corr()
matrix_corrs


# In[123]:


print("Correlation matrix - gender.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[124]:


print("Correlation matrix - embarked.")
matrix_corrs = embarked_vars.corr()
matrix_corrs


# In[125]:


print("Correlation matrix - embarked.")
plt.figure(figsize=(12, 10))
sns.heatmap(data = matrix_corrs,cmap='BrBG', annot=True, linewidths=0.2)


# In[126]:


print("Deleting highly correlated features.")
train_data = train_data.drop(columns =['Male', 'With someone'], axis=1)
test_data = test_data.drop(columns =['Male', 'With someone'], axis=1)
train_data.head()


# I deleted highly correlated columns/features which values of correlation coefficient was higher than 0,95. 

# In[127]:


original_train_set_without_survived = train_data.drop("Survived", axis=1)
original_train_set_with_only_survived = train_data["Survived"]


# In[128]:


X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    original_train_set_without_survived, original_train_set_with_only_survived, train_size=0.6, test_size=0.4, random_state=0)
print(X_train_selected.shape, y_train_selected.shape)
print(X_test_selected.shape, y_test_selected.shape)


# In[129]:


print("Standard Scaler.")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_selected = sc.fit_transform(X_train_selected)

X_test_selected = sc.transform(X_test_selected)

Test = sc.transform(test_data)


# In[130]:


print('Modeling and ROC, AUC, accuracy scores.\n')

print("Decision Tree Classifier")

TitanicTree_selected = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
TitanicTree_selected


# In[131]:


TitanicTree_selected.fit(X_train_selected,y_train_selected)
predTitanicTree_selected = TitanicTree_selected.predict(X_test_selected)

print (predTitanicTree_selected[0:5])
print (y_test_selected [0:5])


# In[132]:


print("Decisions tree hyperparameters tuning by Grid Search.")
params = {'criterion':['gini','entropy'],
          'max_depth': [3, 5, 7],
          'min_samples_split': [2, 4, 8],
          'max_leaf_nodes': [3, 8, 12, 18, 22, 26]}

grid_object_1 = GridSearchCV(DecisionTreeClassifier(), params, cv=3, n_jobs=3)

grid_object_1.fit(X_train_selected,y_train_selected)


# show best parameter for classifier
Dec_tree_params = grid_object_1.best_params_
Dec_tree_params 


# In[133]:


Dec_tree_est = grid_object_1.best_estimator_
Dec_tree_est


# In[134]:


print("Train model with better hyperparameters.")
TitanicTree_tuning = DecisionTreeClassifier(criterion='gini', max_depth = 5, max_leaf_nodes = 8, min_samples_split = 4)
TitanicTree_tuning


# In[135]:


TitanicTree_tuning.fit(X_train_selected,y_train_selected)
predTitanicTree_tuning = TitanicTree_tuning.predict(X_test_selected)

print (predTitanicTree_tuning [0:5])
print (y_test_selected [0:5])


# In[136]:


accuracy_validation_calcs(y_test_selected, predTitanicTree_tuning, X_train_selected, y_train_selected, TitanicTree_tuning)


# In[137]:


ROC_AUC_calc(y_test_selected,predTitanicTree_tuning)


# In[138]:


decision_tree_cross = cross_val_score_model(TitanicTree_tuning, original_train_set_without_survived, original_train_set_with_only_survived)
decision_tree_cross


# In[139]:


# RFC Parameters tunning 

rf_class = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini", "entropy"]}


rfc_grid = GridSearchCV(rf_class,param_grid = rf_param_grid, cv=3, scoring="accuracy", n_jobs= 3, verbose = 1)

rfc_grid.fit(X_train_selected,y_train_selected)

rfc_best = rfc_grid.best_estimator_

# Best score
print(rfc_grid.best_score_)
Rand_For_params = rfc_grid.best_params_
Rand_For_params


# In[140]:


random_forest_classifier_100 = RandomForestClassifier(bootstrap=False, n_estimators=100, random_state=0, criterion='gini', max_depth=None, min_samples_split=10, max_features=10, min_samples_leaf=10)

random_forest_classifier_100.fit(X_train_selected, y_train_selected)

y_pred_100 = random_forest_classifier_100.predict(X_test_selected)


# In[141]:


accuracy_validation_calcs(y_test_selected, y_pred_100, X_train_selected, y_train_selected, random_forest_classifier_100)


# In[142]:


ROC_AUC_calc(y_test_selected,y_pred_100)


# In[143]:


random_forest_cross = cross_val_score_model(random_forest_classifier_100, original_train_set_without_survived, original_train_set_with_only_survived)
random_forest_cross


# In[144]:


print("K nearest neighbor (KNN)")

 
neigh = KNeighborsClassifier(n_neighbors = 9).fit(X_train_selected,y_train_selected)
neigh


# In[145]:


param_neighbors = {
    'n_neighbors': [3, 5, 10, 12, 16],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [30, 40, 80]}

grid_object0 = GridSearchCV(KNeighborsClassifier(), param_neighbors, cv=3, n_jobs=3, scoring='accuracy')
grid_object0.fit(X_train_selected,y_train_selected)


# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_object0.best_params_)
print("Best score: ", grid_object0.best_score_)



# In[146]:


print('Testing models accuracy on all features.\n')

print("K nearest neighbor (KNN)")

neigh = KNeighborsClassifier(n_neighbors = 3, leaf_size=30, algorithm = 'ball_tree').fit(X_train_selected,y_train_selected)
neigh


# In[147]:


y_hat_1 = neigh.predict(X_test_selected)
y_hat_1 [0:5]


# In[148]:


accuracy_validation_calcs(y_test_selected, y_hat_1, X_train_selected, y_train_selected, neigh)


# In[149]:


ROC_AUC_calc(y_test_selected,y_hat_1)


# In[150]:


cross_val_score_model(neigh, original_train_set_without_survived, original_train_set_with_only_survived)


# In[151]:


from sklearn.model_selection import GridSearchCV
grid_params = { 'n_neighbors' : [2,4,5,8,10,12,14,16, 20,22],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose = 1, cv=4, n_jobs = -1)
g_res = gs.fit(X_train_selected, y_train_selected)
g_res.best_score_


# In[152]:


g_res.best_params_


# In[153]:


Neigh_est = g_res.best_estimator_
Neigh_est


# In[154]:


print("K nearest neighbor (KNN)")

neigh_tuned = KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan', weights = 'uniform').fit(X_train_selected,y_train_selected)
neigh_tuned


# In[155]:


y_hat_knn_tuned = neigh_tuned.predict(X_test_selected)
y_hat_knn_tuned [0:5]


# In[156]:


accuracy_validation_calcs(y_test_selected, y_hat_knn_tuned, X_train_selected, y_train_selected, neigh_tuned)


# In[157]:


ROC_AUC_calc(y_test_selected,y_hat_knn_tuned)


# In[158]:


neigh_cross = cross_val_score_model(neigh_tuned, original_train_set_without_survived, original_train_set_with_only_survived)
neigh_cross


# In[159]:


LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_selected,y_train_selected)
LR


# In[160]:


yhat_log = LR.predict(X_test_selected)
yhat_log [0:5]


# In[161]:


yhat_prob_selected = LR.predict_proba(X_test_selected)
yhat_prob_selected [0:5]


# In[162]:


jaccard_score(y_test_selected, yhat_log,pos_label=0)


# In[163]:


accuracy_validation_calcs(y_test_selected, yhat_log, X_train_selected, y_train_selected, LR)


# In[164]:


ROC_AUC_calc(y_test_selected,yhat_log)


# In[165]:


lr_cross = cross_val_score_model(LR, original_train_set_without_survived, original_train_set_with_only_survived)
lr_cross


# In[166]:


print("SVM (Support Vector Machines)")

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train_selected, y_train_selected) 


# In[167]:


yhat_selected_SVC = clf.predict(X_test_selected)
yhat_selected_SVC [0:5]


# In[168]:


accuracy_validation_calcs(y_test_selected, yhat_selected_SVC, X_train_selected, y_train_selected, clf)


# In[169]:


ROC_AUC_calc(y_test_selected,yhat_selected_SVC)


# In[170]:


SVC_cross = cross_val_score_model(clf, original_train_set_without_survived, original_train_set_with_only_survived)
SVC_cross


# In[171]:


param_SVC = {
    'C': [0.1, 50, 70],
    'kernel': ['rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']}

grid_object_SVC = GridSearchCV(svm.SVC(), param_SVC, cv=3, n_jobs=3, scoring='accuracy')
grid_object_SVC.fit(X_train_selected,y_train_selected)


# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_object_SVC.best_params_)
print("Best score: ", grid_object_SVC.best_score_)



# In[172]:


SVC_best_est = grid_object_SVC.best_estimator_
print(SVC_best_est)


# In[173]:


print("SVM (Support Vector Machines)")

clf_tuned = svm.SVC(C=0.1, gamma='scale', kernel='rbf')
clf_tuned.fit(X_train_selected, y_train_selected) 


# In[174]:


yhat_tuned = clf_tuned.predict(X_test_selected)
yhat_tuned [0:5]


# In[175]:


accuracy_validation_calcs(y_test_selected, yhat_tuned, X_train_selected, y_train_selected, clf_tuned)


# In[176]:


ROC_AUC_calc(y_test_selected,yhat_tuned)


# In[177]:


cross_val_score_model(clf_tuned, original_train_set_without_survived, original_train_set_with_only_survived)


# In[178]:


rf = RandomForestClassifier(n_estimators=400, max_depth=None, max_leaf_nodes=None, random_state=None, class_weight=None, warm_start=False, oob_score=False, verbose = 0, min_samples_leaf=1, min_samples_split = 2)
rf.fit(X_train_selected, y_train_selected)
rf_prediction = rf.predict(X_test_selected)

score = metrics.accuracy_score(y_test_selected, rf_prediction)
print(score)


# In[179]:


param_Rand_For = {
    'criterion': ['gini', 'entropy'],
    'min_impurity_decrease': [0.0,1],
    'min_weight_fraction_leaf': [0.0,0.5],
    'n_estimators' : [10,50, 100, 300],
    'max_depth' : [1, 10],
     'min_samples_split':[2,6]
}
    
    
grid_object_rand = GridSearchCV(RandomForestClassifier(), param_Rand_For, cv=3, n_jobs=3, error_score = 'raise')
grid_object_rand.fit(X_train_selected,y_train_selected)


# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", grid_object_rand.best_params_)
print("Best score: ", grid_object_rand.best_score_)



# I create another version to assess deeper problems with hyperparameters.

# In[180]:


rf_tuned = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=6,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
rf_tuned.fit(X_train_selected, y_train_selected)
rf_tuned_prediction = rf_tuned.predict(X_test_selected)


# In[181]:


accuracy_validation_calcs(y_test_selected, rf_tuned_prediction, X_train_selected, y_train_selected, rf_tuned)


# In[182]:


ROC_AUC_calc(y_test_selected,rf_tuned_prediction)


# In[183]:


random_forest_tuned_cross = cross_val_score_model(rf_tuned, original_train_set_without_survived, original_train_set_with_only_survived)
random_forest_tuned_cross


# In[184]:


print("Extra Tree Classifier.")
#X, y = make_classification(n_features=4, random_state=0)
clf_extra = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf_extra.fit(X_train_selected, y_train_selected)
ExtraTreesClassifier(random_state=0)
clf_extra.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0]])
#array([1])

pred_clf_extra = clf_extra.predict(X_test_selected)

print (pred_clf_extra [0:5])
print (y_test_selected [0:5])


# In[185]:


accuracy_validation_calcs(y_test_selected, pred_clf_extra, X_train_selected, y_train_selected, clf_extra)


# In[186]:


ROC_AUC_calc(y_test_selected,pred_clf_extra)


# In[187]:


cross_val_score_model(clf_extra, original_train_set_without_survived, original_train_set_with_only_survived)


# In[188]:


print("Decisions tree hyperparameters tuning by Grid Search.")
params = {'criterion':['gini','entropy'],
          'max_depth': [3, 5, 7],
          'min_samples_split': [2, 4, 8],
          'max_leaf_nodes': [3, 8, 12, 18, 22, 26],
          'n_estimators' : [20, 200, 400]}

grid_object2 = GridSearchCV(ExtraTreesClassifier(), params, cv=3, n_jobs=3)

grid_object2.fit(X_train_selected,y_train_selected)
Extra_params1 = grid_object2.best_params_
Extra_params1


# In[189]:


Extra_est = grid_object2.best_estimator_
Extra_est


# In[190]:


print("Extra Tree Classifier tuned hyperparameters.")
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
#X, y = make_classification(n_features=4, random_state=0)
clf_extra_tuned = ExtraTreesClassifier(n_estimators=20, criterion='gini', random_state=0,max_depth=5,max_leaf_nodes=30,min_samples_split= 8)
clf_extra_tuned.fit(X_train_selected, y_train_selected)
ExtraTreesClassifier(random_state=0)
clf_extra_tuned.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0]])
#array([1])

pred_clf_extra_tuned = clf_extra_tuned.predict(X_test_selected)

print (pred_clf_extra_tuned [0:5])
print (y_test_selected [0:5])


# In[191]:


accuracy_validation_calcs(y_test_selected, pred_clf_extra_tuned, X_train_selected, y_train_selected, clf_extra_tuned)


# In[192]:


ROC_AUC_calc(y_test_selected,pred_clf_extra_tuned)


# In[193]:


extra_tree_cross = cross_val_score_model(clf_extra_tuned, original_train_set_without_survived, original_train_set_with_only_survived)
extra_tree_cross


# In[194]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1), n_estimators=30,
            learning_rate=0.4, random_state=42)
ada_boost.fit(X_train_selected, y_train_selected)


# In[195]:


y_pred_final_ada = ada_boost.predict(X_test_selected)


# In[196]:


accuracy_validation_calcs(y_test_selected, y_pred_final_ada, X_train_selected, y_train_selected, ada_boost)


# In[197]:


ROC_AUC_calc(y_test_selected,y_pred_final_ada)


# In[198]:


cross_val_score_model(ada_boost, original_train_set_without_survived, original_train_set_with_only_survived)


# In[199]:


print("Decisions tree hyperparameters tuning by Grid Search.")
params_ada = {'random_state': [10, 42, None],
          'learning_rate': [0.1, 0.5, 1.0],
          'n_estimators' : [20, 100, 300]}

grid_ada_b = GridSearchCV(AdaBoostClassifier(), params_ada, cv=3, n_jobs=3)


grid_ada_b.fit(X_train_selected,y_train_selected)

Adab_params = grid_ada_b.best_params_
Adab_params


# In[200]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost_tuned = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=1), n_estimators=100,
            learning_rate=0.1, random_state=10)
ada_boost_tuned.fit(X_train_selected, y_train_selected)


# In[201]:


y_pred_tuned_ada = ada_boost_tuned.predict(X_test_selected)


# In[202]:


accuracy_validation_calcs(y_test_selected, y_pred_tuned_ada, X_train_selected, y_train_selected, ada_boost_tuned)


# In[203]:


ROC_AUC_calc(y_test_selected,y_pred_tuned_ada)


# In[204]:


adab_cross = cross_val_score_model(ada_boost_tuned, original_train_set_without_survived, original_train_set_with_only_survived)
adab_cross


# In[205]:


from sklearn.ensemble import GradientBoostingClassifier
gb_clf=GradientBoostingClassifier(n_estimators=20,learning_rate=0.9,max_depth=2,max_features=2,random_state=5)
gb_clf.fit(X_train_selected,y_train_selected)
y_pred_gb=gb_clf.predict(X_test_selected)
accuracy_score(y_pred_gb,y_test_selected)


# In[206]:


accuracy_validation_calcs(y_test_selected, y_pred_gb, X_train_selected, y_train_selected, gb_clf)


# In[207]:


ROC_AUC_calc(y_test_selected,y_pred_gb)


# In[208]:


gradientb_cross = cross_val_score_model(gb_clf, original_train_set_without_survived, original_train_set_with_only_survived)
gradientb_cross


# In[209]:


print("Decisions tree hyperparameters tuning by Grid Search.")
params_gradient = {'loss': ['log_loss', 'exponential'],
          'learning_rate': [0.1, 0.5, 1.0],
          'n_estimators' : [20, 100, 300],
          'random_state': [10, 42, None],
          'subsample': [0.1, 1.0],
          'criterion' : ['friedman_mse','squared_error']}

gradient_grid = GridSearchCV(GradientBoostingClassifier(), params_gradient, cv=3, n_jobs=3)


gradient_grid.fit(X_train_selected,y_train_selected)

 #'max_depth':np.arange(1,21).tolist()[0::2],
          #'min_samples_split':np.arange(2,11).tolist()[0::2],
           #   'max_leaf_nodes':np.arange(3,26).tolist()[0::2]}

# show best parameter for classifier
gradient_grid_params = gradient_grid.best_params_
gradient_grid_params


# In[210]:


from sklearn.ensemble import GradientBoostingClassifier
gradient_clf=GradientBoostingClassifier(n_estimators=20,learning_rate=0.1,random_state=10, criterion='friedman_mse', loss='exponential', subsample = 1.0)
gradient_clf.fit(X_train_selected,y_train_selected)
y_pred_gradient=gradient_clf.predict(X_test_selected)
accuracy_score(y_pred_gradient,y_test_selected)


# In[211]:


from sklearn.ensemble import GradientBoostingClassifier
gradient_clf=GradientBoostingClassifier(n_estimators=20,learning_rate=0.1,random_state=10, criterion='friedman_mse', loss='log_loss', subsample = 1.0)
gradient_clf.fit(X_train_selected,y_train_selected)
y_pred_gradient=gradient_clf.predict(X_test_selected)
accuracy_score(y_pred_gradient,y_test_selected)


# In[212]:


accuracy_validation_calcs(y_test_selected, y_pred_gradient, X_train_selected, y_train_selected, gradient_clf)


# In[213]:


ROC_AUC_calc(y_test_selected,y_pred_gradient)


# In[214]:


gradientb_tuned_cross = cross_val_score_model(gradient_clf, original_train_set_without_survived, original_train_set_with_only_survived)
gradientb_tuned_cross


# In[215]:


x = original_train_set_without_survived
y = original_train_set_with_only_survived


# In[216]:


print("Accuracy score for selected models.")
ML_models = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'KNN', 
              'Logistic Regression', 'SVC', 'Random Forest Tuned', 
              'Extra Trees Classifier', 'Ada Boost Classifier', 
              'Gradient Boost Classifier', 'Gradient Boost Classifier Tuned'],
    'Score': [decision_tree_cross, random_forest_cross, neigh_cross, 
              lr_cross, SVC_cross, random_forest_tuned_cross, 
              extra_tree_cross, adab_cross, gradientb_cross,gradientb_tuned_cross]})
ML_models.sort_values(by='Score',ascending=False)


# "The best model accuracy (based on train is Random Forest).
# Hovewer train and test accuracy suggest that may be some problem with overfitting.
# I will check it on visualisations.
# 
# Train set Accuracy:  0.8614232209737828
# Test set Accuracy:  0.8207282913165266
# Confusion matrix:  [[198  23]
#  [ 41  95]]
# Precision score:  0.8050847457627118
# Recall score:  0.6985294117647058

# In[217]:


print("Analyze Overfitting and Underfitting.")
from yellowbrick.model_selection import (LearningCurve)
fig, ax = plt.subplots(figsize=(6,4))
lc = LearningCurve(RandomForestClassifier(bootstrap=False, n_estimators=100, random_state=0, criterion='gini', max_depth=None, min_samples_split=10, max_features=10, min_samples_leaf=10),
                  cv=10)
lc.fit(x,y)
ax.legend("Learning Curve for Random Forest Model.")


# In[218]:


print("Validation Curve for Random Forest - max depth inspect.")
from yellowbrick.model_selection import (ValidationCurve)
fig, ax = plt.subplots(figsize=(6,4))
vc = ValidationCurve(RandomForestClassifier(bootstrap=False, n_estimators=100, random_state=0, criterion='gini', max_depth=None, min_samples_split=10, max_features=10, min_samples_leaf=10),
                  param_name = 'max_depth', param_range =np.arange(1,20),cv=10, n_jobs= -1)
vc.fit(x,y)
ax.legend("Vaidation Curve for Random Forest Model - max depth influence.")
vc.ax.set(
xlabel='max_depth', ylabel='score')


# In[219]:


print("Overfitting and Underfitting.")
from yellowbrick.model_selection import (LearningCurve)
fig, ax = plt.subplots(figsize=(6,4))
lc = LearningCurve(RandomForestClassifier(bootstrap=False, n_estimators=100, random_state=0, criterion='gini', max_depth=3, min_samples_split=10, max_features=10, min_samples_leaf=10),
                  cv=10)
lc.fit(x,y)
ax.legend("Learning Curve for Random Forest Model.")


# I change the max_depth to 3 and it improve fitting.

# In[220]:


print("Overfitting and Underfitting.")
from yellowbrick.model_selection import (LearningCurve)
fig, ax = plt.subplots(figsize=(6,4))
lc = LearningCurve(ExtraTreesClassifier(n_estimators=20, criterion='gini', random_state=0,max_depth=5,max_leaf_nodes=30,min_samples_split= 8),
                  cv=10)
lc.fit(x,y)
ax.legend("Learning Curve for Random Forest Model.")


# In[221]:


print("Validation Curve for Random Forest - max depth inspect.")
from yellowbrick.model_selection import (ValidationCurve)
fig, ax = plt.subplots(figsize=(6,4))
vc = ValidationCurve(ExtraTreesClassifier(n_estimators=20, criterion='gini', random_state=0,max_depth=5,max_leaf_nodes=30,min_samples_split= 8),
                  param_name = 'max_depth', param_range =np.arange(1,20),cv=10, n_jobs= -1)
vc.fit(x,y)
ax.legend("Vaidation Curve for Random Forest Model - max depth influence.")
vc.ax.set(
xlabel='max_depth', ylabel='score')


# In[222]:


print("Overfitting and Underfitting.")
from yellowbrick.model_selection import (LearningCurve)
fig, ax = plt.subplots(figsize=(6,4))
lc = LearningCurve(ExtraTreesClassifier(n_estimators=20, criterion='gini', random_state=0,max_depth=3,max_leaf_nodes=30,min_samples_split= 8),
                  cv=10)
lc.fit(x,y)
ax.legend("Learning Curve for Random Forest Model.")


# In[223]:


print("Training and predicting from selected and improved model.")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(bootstrap=False, n_estimators=100, random_state=0, criterion='gini', max_depth=3, 
          min_samples_split=10, max_features=10, min_samples_leaf=10)

rf_model.fit(x,y)


# In[224]:


rf_model.score(x,y)


# In[225]:


Test = sc.fit_transform(Test)
Test1 = pd.read_csv("../input/titanic/test.csv")

output = pd.DataFrame({"PassengerId": IDcolumn, "Survived":rf_model.predict(Test)})
output.PassengerId = output.PassengerId.astype(int)
output.Survived = output.Survived.astype(int)

output.to_csv("TitanicSubmission.csv", index=False)
print("Your submission was successfully saved!")
output.head(10)

