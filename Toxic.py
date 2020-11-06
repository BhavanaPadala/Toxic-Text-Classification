#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


train = pd.read_csv('cleaned_data.csv')


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train['toxic'], test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[21]:


print(type(X_train))


# In[30]:


s = str(X_test[:1])


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = vect_word = TfidfVectorizer( lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1),dtype=np.float32)
#x_dtm = vect.fit_transform(X_train[:1].values.astype('U')) 
test_dtm = vect.fit_transform([s])


# In[34]:


test_dtm


# In[15]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
lr = LogisticRegression(C=3.5,solver = 'sag')
sub = {}
bud = {}
y = y_train
lr.fit(x_dtm,y)
y_pred_x = lr.predict(test_dtm)
bud['toxic'] = y_pred_x
y_prob = lr.predict_proba(test_dtm)[:,1]
sub['toxic']= y_prob
print(accuracy_score(y_test, y_pred_x))
print(pd.DataFrame(sub))


# In[16]:


import pickle


# In[17]:


with open('toxic_model.pkl','wb') as f:
    pickle.dump(lr,f)


# In[ ]:




