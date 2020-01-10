#!/usr/bin/env python
# coding: utf-8

# In[29]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

diabetes = pd.read_csv('diabetes.csv')
diabetes.columns
# In[30]:
diabetes.head()
# In[31]:
print("Diabetes data set dimensions : {}".format(diabetes.shape))
# In[32]:
diabetes.groupby('Outcome').size()
# In[33]:

sns.countplot(diabetes['Outcome'],label="Count")

# In[34]:

diabetes.info()

# In[35]:

diabetes.isnull().sum()
diabetes.isna().sum()

# In[36]:

diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
print(diabetes_mod.shape)

# In[37]:

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome

# In[38]:

models = []
models.append(('K-Nearest', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Neural Networks', MLPClassifier()))
models.append(('Naive Bayes', GaussianNB()))

# In[39]:

# In[40]:
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
# In[41]:
names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Model': names, 'Accuracy': scores})
print(tr_split)
# In[42]:
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')

