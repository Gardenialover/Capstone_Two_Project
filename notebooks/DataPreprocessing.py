#!/usr/bin/env python
# coding: utf-8

# ## loading data

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

# data manipulation and math

import numpy as np
import scipy as sp
import pandas as pd

# plotting and visualization

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
#preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder as OHE
from imblearn.over_sampling import SMOTE

# modeling
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import os


# In[2]:


df = pd.read_csv("../data/wine_df_eda.csv")
df.head()


# In[3]:


df.drop(df.columns[[0]], inplace=True, axis=1)
df.head()


# ## Preprocess categorical features

# In[4]:


new_df = pd.get_dummies(df,drop_first = True,dtype='int' )
new_df.tail(20)


# In[5]:


new_df["quality"] = new_df["quality"].apply(lambda x : 1 if x >=7 else 0)
new_df["quality"].value_counts()


# ## Scaling

# In[6]:


X = new_df.drop(["quality","wine_type_white"],axis = 1)

#Save the X labels 
wine_index = X.index
#Save the column names
wine_columns = X.columns
scaler= MinMaxScaler()
scale_X = scaler.fit_transform(X)


# In[7]:


#Create a new dataframe from `scale-X`
X_df = pd.DataFrame(scale_X, columns=wine_columns)
X_df.head()


# In[8]:


# verify scaling
X_df.mean()


# ## feature selections for modeling

# In[9]:


new_df.drop("wine_type_white",axis=1).corr()['quality'].sort_values(ascending=False)


# it shows  alcohol, fixed_acidity_ration are positivie correlated to quality. Also, in previous EDA, The density vs residual_sugar,density vs chlorides, density vs fixed_acidity, totol_sulfur_dioxide vs residual_sugar, free_sulfur_dioxide vs residual_sugar shows strong positive correlation. Therefore the following features will be selected for furture modeling.

# In[10]:


#intially, should use several features or all features? when data leakage happen?
X = X_df[["alcohol","fixed_acidity_ratio", "density","residual_sugar", "chlorides","total_sulfur_dioxide"]]
X


# In[11]:


y =new_df["quality"]


# ## proportion of classes

# In[12]:


class_counts = new_df["quality"].value_counts()
class_counts


# In[13]:


class_percentages = pd.Series([(x / new_df.shape[0]) * 100.00 for x in class_counts])


# In[14]:


fig, ax = plt.subplots()
ax.bar(class_counts.index, class_counts)
ax.set_xticks([ 0,1])
ax.set_xticklabels(class_percentages.index.astype(str) + '\n' + ' ' +
                   class_percentages.round(0).astype(str) + '%')
ax.set_ylabel('Count')
ax.set_xlabel('wine quality')
ax.set_title('wine quality',
              fontsize = 10)
plt.show()


# It shows the wine quality data is imbalanced, need to resampling using synthetic minority oversampling technique for data balancing

# In[15]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

oversample = SMOTE(random_state = 40)
X_os, y_os = oversample.fit_resample(X_df, y)
sns.countplot(x=y_os)
plt.xticks([0,1], ['poor quality','good quality'])
plt.title("Types of Wine")
plt.show()


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_os,
                                                y_os,
                                                test_size = 0.20,
                                                random_state = 123)
X_train.shape, X_test.shape


# In[ ]:




