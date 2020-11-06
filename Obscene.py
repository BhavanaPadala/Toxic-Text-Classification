#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('cleaned_data.csv')


# In[3]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train['obscene'], test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = vect_word = TfidfVectorizer( lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1),dtype=np.float32)
x_dtm = vect.fit_transform(X_train.values.astype('U')) 
test_dtm = vect.transform(X_test)


# In[5]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
lr = LogisticRegression(C=3.5,solver = 'sag')
sub = {}
bud = {}
y = y_train
lr.fit(x_dtm,y)
y_pred_x = lr.predict(test_dtm)
bud['obscene'] = y_pred_x
y_prob = lr.predict_proba(test_dtm)[:,1]
sub['obscene']= y_prob
print(accuracy_score(y_test, y_pred_x))
print(pd.DataFrame(sub))


# In[7]:


import pickle


# In[8]:


with open('obscene_model.pkl','wb') as f:
    pickle.dump(lr,f)


# In[ ]:




