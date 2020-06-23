#!/usr/bin/env python
# coding: utf-8

# # important libraries

# In[1]:


import numpy as np
import pandas as pd
import os
import re
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle

# Nltk for tekenize and stopwords
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import TreebankWordTokenizer
from nltk import RegexpTokenizer
nltk.download("stopwords")


# **reading data**

# In[2]:


data = pd.read_csv("data.csv")
data1 = pd.read_csv("data1.csv")


# **Exploring data**

# In[3]:


data.columns


# In[4]:


data1.columns


# In[5]:


data.drop("selected_text" , axis = 1 , inplace = True)


# In[6]:


train = pd.concat([data , data1] , axis = 0 )


# In[7]:


train.shape


# In[8]:


train.isnull().sum()


# In[9]:


train = train.dropna()


# In[10]:


count = train.sentiment.value_counts()
percentage = train.sentiment.value_counts(normalize = True)
df = pd.DataFrame({"count":count , "percentage":percentage})


# **Looking sentimentwise**

# In[11]:


df.head()


# In[13]:


train.describe()


# # cleaning part

# In[14]:


def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

train = standardize_text(train, "text")


# In[15]:


def rep(text):
    grp = text.group(0)
    if len(grp) > 1:
        return grp[0:1] # can change the value here on repetition
def unique_char(rep,sentence):
    convert = re.sub(r'(\w)\1+', rep, sentence) 
    return convert


# In[16]:


train['text']=train['text'].apply(lambda x : unique_char(rep,x))


# In[17]:


train.head()


# In[18]:


tokenizer = RegexpTokenizer(r"\w+")


# In[19]:


train.text = train["text"].apply(lambda x:tokenizer.tokenize(x))


# In[20]:


train.head()


# In[21]:


stop = set(stopwords.words('english'))


# In[22]:


train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))


# In[23]:


train.head()


# In[24]:


tokenizer = TreebankWordTokenizer()
stemmer = nltk.WordNetLemmatizer()
train["text"] = train["text"].apply(lambda x: tokenizer.tokenize(x))
train["text"] = train["text"].apply(lambda x:" ".join([stemmer.lemmatize(token) for token in x]))


# # model building part

# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# In[27]:


le = LabelEncoder()
train["sentiment"] = le.fit_transform(train.sentiment)


# In[28]:


train.sentiment.unique()


# In[29]:


corpus = train["text"].tolist()
labels = train["sentiment"].tolist()


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(corpus,labels, test_size=0.2, random_state=40)


# In[32]:


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(x_train)
X_test_tfidf = tfidf_vectorizer.transform(x_test)


# In[34]:


pickle.dump(tfidf_vectorizer, open('tfidf-transform.pkl', 'wb'))


# In[35]:


from sklearn.linear_model import LogisticRegression
clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state=40)
clf_tfidf.fit(X_train_tfidf, y_train)

y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)


# In[36]:


filename = 'tweet-sentiment-lc-model.pkl'
pickle.dump(clf_tfidf, open(filename, 'wb'))


# In[ ]:




