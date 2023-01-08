import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

############## DATA CLEANING #####################################################
df=pd.read_csv(r'D:\Coding\Python\Machine Learning\Algorithms\Bengaluru_House_Data.csv')
# print(df.isnull().sum())
a=df.isnull().sum()/df.shape[0]*100
n=a[a>17].keys()
df=df.drop(columns=n)

# Numerical
num_var=df.select_dtypes(include=['int64','float64']).columns
print(df[num_var])
im=SimpleImputer(strategy='mean')
im.fit(df[num_var])
df[num_var]=im.transform(df[num_var])
print(df[num_var].isnull().sum())

# Categorical 
cat_var=df.select_dtypes(include='O').columns
imp=SimpleImputer(strategy='most_frequent')
imp.fit(df[cat_var])
df[cat_var]=imp.transform(df[cat_var])

print(df.isnull().sum().sum())

################# DATA PREPROCESSING ###########################################
df2=df.drop(columns=df[cat_var])
# print(df2)
################## DATA SPlITING ################################################
X=df2.drop(columns='price', axis=1)
y=df2['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=69)

################## FEATURE SCALING ####################################################
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

################### TRAINING ####################################################
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, y_train)

# print(lr.coef_) # used to  print feature coefeciant that our model has learned


print(lr.intercept_)

################# PREDICTION ###############################################
pre=lr.predict(X_test)
print(pre) # the predicted values 
print(y_test) # the original values 

score=lr.score(X_test, y_test) # shows you the accuracy percentage of your model
print(score*100)
 










