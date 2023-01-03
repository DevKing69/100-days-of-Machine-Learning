import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=sb.load_dataset('titanic')
# print(df.head())
df2=df[['survived','pclass','age','parch']]
print(df2.head())

df3=df2.fillna(df2.mean())
print(df3)

x=df3.drop('survived', axis=1)
y=df3['survived']
print('shape of x=',x.shape)
print('shape of y=',y.shape)

x_train,y_train,x_test,y_test=train_test_split(x,y, 
test_size=0.2, # this specifies the percentage of data we want for test dataset (0.2 means 20%)
random_state=51) # this takes 50 values 

 
sc=StandardScaler()
sc.fit(x_train)
print(sc.mean_)
print(x_train.describe())

x_train_sc=sc.transform(x_train)
x_test_sc=sc.transform(x_test)
# print(x_train)

x_train_sc=pd.DataFrame(x_train_sc, columns=["pclass","age","parch"])
x_train_sc=pd.DataFrame(x_test_sc, columns=["pclass","age","parch"])

print(x_train_sc.describe())


















