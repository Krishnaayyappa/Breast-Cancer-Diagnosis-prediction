# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:28:29 2021

@author: krish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import confusion_matrix
from scipy.stats import beta
from scipy.stats import f 

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn import svm
# use seaborn plotting style defaults
import seaborn as sns; sns.set()

#Reading the CSV file
df = pd.read_csv("Breast_cancer_data.csv")

#display columns
print(df.columns)
#display the null values
df.isnull().sum()

#diagnosis variable of breast cancer dataset
Y= df['diagnosis']

df.drop(['diagnosis'],axis=1,inplace=True)


#normalize data
df = (df - df.mean())/df.std()
# Displaying DataFrame columns.
df.columns
# Some basic information about each column in the DataFrame 
df.info()

#bservations and variables
observations = list(df.index)
variables = list(df.columns)

#visualisation of the data using a box plot
sns.boxplot(data=df, orient="h", palette="Set2")

#pairplot
sns.pairplot(df)

#Covariance
dfc = df - df.mean() #centered data
plt. figure()
ax = sns.heatmap(dfc.cov(), cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
plt.title('Covariance matrix')

#Principal component analysis
pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)
plt. figure()
plt.scatter(Z[:,0], Z[:,1], c='r')
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
for label, x, y in zip(observations, Z[:,0], Z[:,1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom')

#Eigenvectors
A = pca.components_.T
plt. figure()
plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$');
for label, x, y in zip(variables, A[:,0],A[:,1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
        textcoords='offset points', ha='right', va='bottom')
    
plt. figure()
plt.scatter(A[:,0],A[:,1],marker='o',c=A[:,2],s=A[:,3]*500,
    cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(variables,A[:,0],A[:,1]):
    plt.annotate(label,xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
#Eigenvalues
Lambda = pca.explained_variance_ 
#Scree plot
plt. figure()
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda/sum(Lambda), 'ro-', lw=2)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance')

#Explained variance
ell = pca.explained_variance_ratio_
plt. figure()
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

#Biplot
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[:,0]; A2 = A[:,1]; Z1 = Z[:,0]; Z2 = Z[:,1]
plt. figure()
for i in range(len(A1)):
# arrows project features as vectors onto PC axes
    plt.arrow(0, 0, A1[i]*max(Z1), A2[i]*max(Z2),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(A1[i]*max(Z1)*1.02, A2[i]*max(Z2)*1.02,variables[i], color='r')

for i in range(len(Z1)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.scatter(Z1[i], Z2[i], c='g', marker='o')
    #plt.text(Z1[i]*1.2, Z2[i]*1.2, observations[i], color='b')
plt.xlabel('PC1')
plt.ylabel('PC2')


plt.figure()
comps = pd.DataFrame(A,columns = variables)
sns.heatmap(comps,cmap='RdYlGn_r', linewidths=0.5, annot=True, 
            cbar=True, square=True)
ax.tick_params(labelbottom=False,labeltop=True)
plt.title('Principal components')


# logistic regression on Original dataset
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5,shuffle=(True))
logisticRegr = LogisticRegression()
print('\n')
print('Logistic Regression prediction results ---->\n')

for train_ix, test_ix in kfold.split(df):
    #X_train, X_test = df[train_ix], df[test_ix]
    X_train, X_test = df.loc[np.intersect1d(df.index, train_ix)], \
                df.loc[np.intersect1d(df.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
   
    
    logisticRegr.fit(X_train, y_train)
    logisticRegr_prediction = logisticRegr.predict(X_test)
    
    # Accuracy score is the simplest way to evaluate

    print('Accuracy = ', accuracy_score(y_test,logisticRegr_prediction))
    print(confusion_matrix(y_test,logisticRegr_prediction))
    print(classification_report(y_test,logisticRegr_prediction))
    print('Precision = ',precision_score(y_test,logisticRegr_prediction))
    print('Recall = ',recall_score(y_test,logisticRegr_prediction))

#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

#Naive Bayes on Original data set
Gaussian = GaussianNB()
print('\n')
print('GaussianNB prediction results ---->\n')
    
for train_ix, test_ix in kfold.split(df):
    X_train, X_test = df.loc[np.intersect1d(df.index, train_ix)], \
                df.loc[np.intersect1d(df.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
    
    
    Gaussian.fit(X_train, y_train)


    Gaussian_prediction = Gaussian.predict(X_test)


    # Accuracy score is the simplest way to evaluate
    print('Accuracy = ', accuracy_score(y_test,Gaussian_prediction))
    print(confusion_matrix(y_test,Gaussian_prediction))
    print(classification_report(y_test,Gaussian_prediction))
    print('Precision = ',precision_score(y_test,Gaussian_prediction))
    print('Recall = ',recall_score(y_test,Gaussian_prediction))

#Transformed dataset
Z1 = pd.DataFrame(Z)

#logistic regression on transformed dataset
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5,shuffle=(True))
logisticRegr = LogisticRegression()
print('\n')
print('Logistic Regression prediction results ---->\n')

for train_ix, test_ix in kfold.split(Z1):
    #X_train, X_test = df[train_ix], df[test_ix]
    X_train, X_test = Z1.loc[np.intersect1d(Z1.index, train_ix)], \
                Z1.loc[np.intersect1d(Z1.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
   
    
    logisticRegr.fit(X_train, y_train)
    logisticRegr_prediction = logisticRegr.predict(X_test)
    
    # Accuracy score is the simplest way to evaluate

    print('Accuracy = ', accuracy_score(y_test,logisticRegr_prediction))
    print(confusion_matrix(y_test,logisticRegr_prediction))
    print(classification_report(y_test,logisticRegr_prediction))
    print('Precision = ',precision_score(y_test,logisticRegr_prediction))
    print('Recall = ',recall_score(y_test,logisticRegr_prediction))
    


#naive bayes on transformed dataset
Gaussian = GaussianNB()
print('\n')
print('GaussianNB prediction results ---->\n')
    
for train_ix, test_ix in kfold.split(Z1):
    X_train, X_test = Z1.loc[np.intersect1d(Z1.index, train_ix)], \
                Z1.loc[np.intersect1d(Z1.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
    
    
    Gaussian.fit(X_train, y_train)


    Gaussian_prediction = Gaussian.predict(X_test)


    # Accuracy score is the simplest way to evaluate
    print('Accuracy = ', accuracy_score(y_test,Gaussian_prediction))
    print(confusion_matrix(y_test,Gaussian_prediction))
    print(classification_report(y_test,Gaussian_prediction))
    print('Precision = ',precision_score(y_test,Gaussian_prediction))
    print('Recall = ',recall_score(y_test,Gaussian_prediction))
    

#dataset containing first two principal components
PC = pd.DataFrame(Z[:, 0:2])


#logistic regression on first two PCS

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5,shuffle=(True))
logisticRegr = LogisticRegression()
print('\n')
print('Logistic Regression prediction results ---->\n')

for train_ix, test_ix in kfold.split(PC):
    #X_train, X_test = df[train_ix], df[test_ix]
    X_train, X_test = PC.loc[np.intersect1d(PC.index, train_ix)], \
                PC.loc[np.intersect1d(PC.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
   
    
    logisticRegr.fit(X_train, y_train)
    logisticRegr_prediction = logisticRegr.predict(X_test)
    
    # Accuracy score is the simplest way to evaluate

    print('Accuracy = ', accuracy_score(y_test,logisticRegr_prediction))
    print(confusion_matrix(y_test,logisticRegr_prediction))
    print(classification_report(y_test,logisticRegr_prediction))
    print('Precision = ',precision_score(y_test,logisticRegr_prediction))
    print('Recall = ',recall_score(y_test,logisticRegr_prediction))
    
    
    
#Naive Bayes on First two PCs
Gaussian = GaussianNB()
print('\n')
print('GaussianNB prediction results ---->\n')
    
for train_ix, test_ix in kfold.split(PC):
    X_train, X_test = PC.loc[np.intersect1d(PC.index, train_ix)], \
                PC.loc[np.intersect1d(PC.index, test_ix)]
    y_train, y_test = Y[np.intersect1d(Y.index, train_ix)], \
        Y[np.intersect1d(Y.index, test_ix)]
    
    
    Gaussian.fit(X_train, y_train)


    Gaussian_prediction = Gaussian.predict(X_test)


    # Accuracy score is the simplest way to evaluate
    print('Accuracy = ', accuracy_score(y_test,Gaussian_prediction))
    print(confusion_matrix(y_test,Gaussian_prediction))
    print(classification_report(y_test,Gaussian_prediction))
    print('Precision = ',precision_score(y_test,Gaussian_prediction))
    print('Recall = ',recall_score(y_test,Gaussian_prediction))
    
