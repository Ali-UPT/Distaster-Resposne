#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[16]:


import nltk
nltk.download(['punkt', 'wordnet'])


# In[17]:


# import libraries

import nltk
import re
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import ExtraTreesClassifier
import pickle


# In[3]:


# load data from database
name= 'sqlite:///data/DisasterResponse.db'
engine = create_engine(name)
df = pd.read_sql('DisasterResponse', con=engine)


# In[4]:


#  X=Message column,Y=Your newly created 36 coulmn of categoies
X = df['message'].values
Y = df.drop(['id', 'message','original','genre'], axis=1)
Y.related.replace(2,1,inplace=True)
#converting the values to int
for c in Y.columns:
    Y[c]  = Y[c].apply(lambda x : int(x))


# ### 2. Write a tokenization function to process your text data

# In[6]:


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@/.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(X):
    detected_urls = re.findall(url_regex, X)
    for url in detected_urls:
        message = X.replace(url, "urlplaceholder")

    tokens = word_tokenize(X)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[7]:


def get_metrics(test_v, predicted_v):
    """
    get_metrics calculates f1 score, accuracy and recall

    Args:
        test_value (list): list of actual values
        predicted_value (list): list of predicted values

    Returns:
        dictionray: a dictionary with accuracy, f1 score, precision and recall
    """
    accuracy = accuracy_score(test_v,predicted_v)
    precision =round( precision_score(test_v,predicted_v,average='micro'))
    recall = recall_score(test_v,predicted_v,average='micro')
    f1 = f1_score(test_v,predicted_v,average='micro')
    return {'Accuracy':accuracy, 'f1 score':f1,'Precision':precision, 'Recall':recall}


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[8]:


#We can generate a multi-output data with a make_multilabel_classification function. The target dataset 
#contains 1 feature (x= in this case 'message' column), 36 classes (y= in this case those 36 categories)

dtc =  DecisionTreeClassifier(random_state=0,  max_depth=2, min_samples_split=3)
pipeline = Pipeline([
('vect', CountVectorizer(tokenizer=tokenize)),
('tfidf', TfidfTransformer()),
('clf', MultiOutputClassifier(estimator=dtc))
])


# In[9]:


dtc.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# #### Before spliting our dataset into train and test sets, let's print out our lable column
# 

# In[10]:


np.unique(Y)


# #### Now it is time for spliting our dataset into train and test sets

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)
pipeline_fit = pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[13]:


predicted_y_train = pipeline_fit.predict(X_train)
predicted_y_test = pipeline_fit.predict(X_test)


# In[25]:


results_for_train = []
for i,column in enumerate(y_train.columns):
    result = get_metrics(y_train.loc[:,column].values,predicted_y_train[:,i])
    results_for_train.append(result)
df_train_results = pd.DataFrame(results_for_train)
display(df_train_results)


# In[27]:


results_for_test = []
for i,column in enumerate(y_test.columns):
    result = get_metrics(y_test.loc[:,column].values,predicted_y_test[:,i])
    results_for_test.append(result)
df_test_results = pd.DataFrame(results_for_test)
df_test_results


# #### Please note that we have Two classes (0 or one) for our Y values in this dataset. In addition, we have 36 attribute for this dataset (shown above) which will be used for prediction, except Sample Code Number which is the id number.

# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[28]:


scaler = StandardScaler()
pipeline = Pipeline([
   ('vect', CountVectorizer(tokenizer=tokenize)),
   ('tfidf', TfidfTransformer()),
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', MLPClassifier(max_iter=50))
])

parameters = {
    'clf__hidden_layer_sizes': (100,),
    'clf__activation': ['relu'],
    'clf__solver': ['adam'],
    'clf__alpha': [5],
    'clf__learning_rate': ['constant'],
    'clf__learning_rate_init': [0.001],
    'clf__shuffle': [True],
    'clf__beta_2': [0.999],
}

model_T1 = GridSearchCV(pipeline, param_grid=parameters)

model_T1.fit(X_train, y_train)
predicted_y_test = model_T1.predict(X_test)
predicted_y_train = model_T1.predict(X_train)


# #### Let's see what our allowable parameters for the Pipeline are and then we can select/update our paramteres dictionary based upon that:

# In[29]:


pipeline.get_params()


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[30]:


results_for_train = []
for i,column in enumerate(y_train.columns):
    result = get_metrics(y_train.loc[:,column].values,predicted_y_train[:,i])
    results_for_train.append(result)
df_train_results = pd.DataFrame(results_for_train)
display(df_train_results)


# In[31]:


results_for_test = []
for i,column in enumerate(y_test.columns):
    result = get_metrics(y_test.loc[:,column].values,predicted_y_test[:,i])
    results_for_test.append(result)
df_test_results = pd.DataFrame(results_for_test)
df_test_results


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[32]:


pipeline = Pipeline([
   ('vect', CountVectorizer(tokenizer=tokenize)),
   ('tfidf', TfidfTransformer()),
     ('scaler_2', MaxAbsScaler()),
   ('clf', ExtraTreesClassifier(random_state=0))
])
parameters = {
 'clf__n_estimators': [5],
}

model_T2 = GridSearchCV(pipeline, param_grid = parameters)
model_T2.fit(X_train, y_train)
predicted_y_test = model_T2.predict(X_test)
predicted_y_train = model_T2.predict(X_train)


# #### Let's see what our allowable parameters for the Pipeline are and then we can select/update our paramteres dictionary based upon that:

# In[33]:


pipeline.get_params()


# ### Test the model once more 

# In[34]:


results_for_train = []
for i,column in enumerate(y_train.columns):
    result = get_metrics(y_train.loc[:,column].values,predicted_y_train[:,i])
    results_for_train.append(result)
df_train_results = pd.DataFrame(results_for_train)
display(df_train_results)


# In[35]:


results_for_test = []
for i,column in enumerate(y_test.columns):
    result = get_metrics(y_test.loc[:,column].values,predicted_y_test[:,i])
    results_for_test.append(result)
df_test_results = pd.DataFrame(results_for_test)
df_test_results


# ### 9. Export your model as a pickle file

# In[36]:


pickle.dump(model_T1, open('classifier.pkl', 'wb'))


# #### After the above steps, one can see a file with the name model_pkl in the directory:

# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




