from google.colab import drive

call .flush_and_unmountmount("/content/drive", force_remount=True)

drive.mount('/content/drive')

import pandas as pd 

import numpy as np

import sklearn as sk

import seaborn as sns

data=pd.read_pdf("/content/drive")

df=data.head(10)
import matplotlib.pyplot as plt

plt.bar (df['Age'],4)
plt.scatter(df['Age'],df['CreditScore'])
plt.scatter(df['Age'],df['CreditScore'],df['Tenure'])
data.describe()
data.isnull().sum()
sns.boxplot(data['Age'])
q=data.quantile(q=[0.75,0.5])
iqr=q.iloc[0]-q.iloc[1]
iqr
l=q.iloc[1]-(1.5*iqr)
l['Age']
u=q.iloc[1]+(1.5*iqr)
u['Age']
data['Age']=np.where(data['Age']>u['Age'],u['Age'],np.where(data['Age']<l['Age'],l['Age'],data['Age']))
sns.boxplot(data['Age'])
df.info()
from sklearn.preprocessing import LabelEncoder
from collections import Counter as count
le=LabelEncoder()
data['Surname']=le.fit_transform(data['Surname'])
data
data['Geography']=le.fit_transform(data['Geography'])
data['Gender']=data['Gender'].replace(['Male','Female'],[0,1])
data
x=data.iloc[:,0:13]
x
y=data['Exited']
y
from sklearn.preprocessing import scale
scale(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train

x_train.shape

y_train

y_train.shape

x_test

x_test.shape

y_test

y_test.shape