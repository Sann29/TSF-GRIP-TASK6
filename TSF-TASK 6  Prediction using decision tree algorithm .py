#!/usr/bin/env python
# coding: utf-8

# # Prediction using Decision Tree Algorithm (Level-Intermediate)

# ###### Author : Shaikh Saniya Ayub
# Task6 : Prediction using Decision Tree Algorithm
# GRIP @ The Sparks Foundation
# Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks, and even multioutput tasks.For the given ‘Iris’ dataset, I created the Decision Tree classifier and visualized it graphically. The purpose of this task is if we feed any new data to this classifier, it would be able to predict the right class accordingly.  
# 
# Technical Stack : Sikit Learn, Numpy Array, Seaborn, Pandas, Matplotlib

# Prediction using Decision Tree Algorithm
# 
# ● Create the Decision Tree classifier and visualize it graphically.
# 
# ● The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# # Importing liabraries and dataset

# In[4]:


import sklearn.datasets as datasets
import pandas as pd


# Import dataset

# In[5]:


iris=datasets.load_iris()


# In[6]:


X = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[7]:


X.head()


# In[8]:


X.tail()


# In[9]:


X.info()


# In[10]:


X.describe()


# In[11]:


X.isnull().sum()


# # Data Visualization comparing various features

# In[32]:


import seaborn as sns

import matplotlib.pyplot as plt


# In[34]:


# Input data Visualization
sns.pairplot(X)


# # Decision Tree Model Training

# In[12]:


Y = iris.target
Y


# split dataset into train and test sets

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# Defining the Decision Tree Algorithm

# In[14]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)

print('Decision Tree Classifer Created Successfully')


# In[15]:


y_predict = dtc.predict(X_test)


# Constructing confusion matrix

# In[16]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)


# In[17]:


from sklearn import tree
import matplotlib.pyplot as plt


# visualizing the Decision tree

# In[18]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa','versicolor','virginica']

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)

tree.plot_tree(dtc, feature_names = fn, class_names = cn, filled = True);


# # Predicting the class output for some random values of petal and sepal length and width

# In[30]:


import sklearn.metrics as sm


# In[31]:


# Model Accuracy
print("Accuracy:",sm.accuracy_score(y_test, y_predict))


# The accuracy of this model is 1 or 100% since I have taken all the 4 features of the iris dataset for creating the decision tree model.

# # Conclusion
# I was able to successfully carry-out prediction using Prediction using Decision Tree Algorithm and was able to evaluate the model's accuracy score.
# Thank You

# # THANK YOU

# In[ ]:




