#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st


from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# In[3]:


st.title("Streamlit Example")


# In[4]:


#get_ipython().system('streamlit run st_trial.ipynb')


# In[5]:
st.write("""
# Explore classifiers
Which one is the best
""")

# In[6]


#st.selectbox("Select dataset", ("Iris", 'Breast cancer', 'Wine dataset'))


dataset_name = st.sidebar.selectbox("Select dataset", ("Iris", 'Breast cancer', 'Wine dataset'))
#st.sidebar.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", 'SVM', 'Random Forest'))

# In[]



def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name =='Breast cancer':
        data = datasets.load_breast_cancer()
    elif dataset_name == 'Wine dataset':
        data = datasets.load_wine()
        
    X = data.data
    y = data.target
    return X,y

X, y = get_dataset(dataset_name)

st.write("Shape of Dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


# In[]

def add_parameters(clf):
    params = {}
    if clf == 'KNN':
        k = st.sidebar.slider('K', 1,15)
        params['K'] = k
    elif clf == 'SVM':
        c = st.sidebar.slider('C', 0.01,10.0)
        params['C'] = c
    elif clf == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2,15)
        n_estimators = st.sidebar.slider('n_estimators', 1,100)
        
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
        
    return params

# In[]



params = add_parameters(classifier_name)

# In[]

def get_classifier(clf_name,params):
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
        
    elif clf_name == 'SVM':
        clf = SVC(C = params['C'])

    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators = params['n_estimators'],
                                     max_depth = params['max_depth'], random_state=123)
        
    return clf

clf = get_classifier(classifier_name, params)

# In[]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc*100}%")


# In[]

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()

plt.scatter(x1, x2, c = y, alpha = 0.8, cmap = 'viridis')

plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.colorbar()

st.pyplot(fig)





