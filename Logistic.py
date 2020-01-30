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


"""
X = NewsPred[['n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
       'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs',
       'num_self_hrefs', 'num_imgs', 'num_videos', 'average_token_length',
       'num_keywords', 'data_channel_is_lifestyle',
       'data_channel_is_entertainment', 'data_channel_is_bus',
       'data_channel_is_socmed', 'data_channel_is_tech',
       'data_channel_is_world', 'kw_min_min', 'kw_max_min', 'kw_avg_min',
       'kw_min_max', 'kw_max_max', 'kw_avg_max', 'kw_min_avg', 'kw_max_avg',
       'kw_avg_avg', 'self_reference_min_shares', 'self_reference_max_shares',
       'self_reference_avg_sharess', 'weekday_is_monday', 'weekday_is_tuesday',
       'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
       'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'LDA_00',
       'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
       'global_sentiment_polarity', 'global_rate_positive_words',
       'global_rate_negative_words', 'rate_positive_words',
       'rate_negative_words', 'avg_positive_polarity', 'min_positive_polarity',
       'max_positive_polarity', 'avg_negative_polarity',
       'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
       'title_sentiment_polarity', 'abs_title_subjectivity',
       'abs_title_sentiment_polarity']]
    
    """
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

