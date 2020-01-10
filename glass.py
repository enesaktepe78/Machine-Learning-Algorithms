#!/usr/bin/env python
# coding: utf-8

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# In[525]:

glass_df = pd.read_csv("glass.csv")
glass_df

sns.set(style="whitegrid", font_scale=1.8)
plt.subplots(figsize = (15,8))
sns.countplot('Type',data=glass_df).set_title('Count of Glass Types')
Y= glass_df["Type"]
Y.head()

# In[526]:

sns.set(style="whitegrid", font_scale=1.2)
plt.subplots(figsize = (20,15))
plt.subplot(3,3,1)
sns.boxplot(x='Type', y='RI', data=glass_df)
plt.subplot(3,3,2)
sns.boxplot(x='Type', y='Na', data=glass_df)
plt.subplot(3,3,3)
sns.boxplot(x='Type', y='Mg', data=glass_df)
plt.subplot(3,3,4)
sns.boxplot(x='Type', y='Al', data=glass_df)
plt.subplot(3,3,5)
sns.boxplot(x='Type', y='Si', data=glass_df)
plt.subplot(3,3,6)
sns.boxplot(x='Type', y='K', data=glass_df)
plt.subplot(3,3,7)
sns.boxplot(x='Type', y='Ca', data=glass_df)
plt.subplot(3,3,8)
sns.boxplot(x='Type', y='Ba', data=glass_df)
plt.subplot(3,3,9)
sns.boxplot(x='Type', y='Fe', data=glass_df)
plt.show()

# In[527]:

glass_df.info()

# In[528]:

glass_df = glass_df.drop("Type",axis=1)
glass_df.head()

# In[529]:

glass_df.describe()

# In[530]:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(glass_df,Y, test_size=0.3,random_state=0)

print(x_train.shape)

# In[531]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
sc1=accuracy_score(y_test, y_pred)

# In[532]:

from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score
clf = GaussianNB()
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
sc2=accuracy_score(y_test, y_pred)

# In[533]:

from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import accuracy_score
clf = MLPClassifier(max_iter=10000)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
sc3=accuracy_score(y_test, y_pred)

# In[534]:

pd.DataFrame([['Decision Tree',sc1],['Naive Bayes',sc2],['Neural Networks',sc3],],
                                 columns=['Model','Accuracy'])
print(pd.DataFrame([['Decision Tree',sc1],['Naive Bayes',sc2],['Neural Networks',sc3],],
                                 columns=['Model','Accuracy']))

# In[ ]:




