#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


NewsPred = pd.read_csv("C:/Users/deepak/Desktop/Git/Project/train.csv")
NewsPred.head()


# In[ ]:


NewsPred.columns


# In[ ]:


X = NewsPred[["n_tokens_title","n_unique_tokens",'avg_positive_polarity','self_reference_min_shares',
              "n_tokens_content","n_non_stop_words","n_non_stop_unique_tokens","num_hrefs","num_imgs",
              "num_videos",'weekday_is_monday', 'weekday_is_tuesday','weekday_is_wednesday',
              'weekday_is_thursday', 'weekday_is_friday','weekday_is_saturday', 'weekday_is_sunday']]
y = NewsPred['Popularity']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.20,random_state=0)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[ ]:


y_pred=logreg.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:

with open('model.pkl','wb') as model_pkl:
    pickle.dump(logreg, model_pkl)

