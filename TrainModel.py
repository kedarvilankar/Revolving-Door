# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:32:06 2020

@author: Kedarpv

A Random Forest classifier model to predict whether an employee would quit or 
not.
"""


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import seaborn as sns


# function to compute number of days an employee is employed . 
#If an employee has not quit then Jan 01, 2016 (last date in the dataset) is considered to be the  current date.
def get_employed(row):
    date_time_str = 'Jan 01 2016'
    date_time_obj = dt.datetime.strptime(date_time_str, '%b %d %Y')
    if math.isnan(row['quit_year']):
        return (date_time_obj- row['join_date']).days
    else:
        return (row['quit_date']- row['join_date']).days

df = pd.read_csv('data/employee_retention.csv')
df = df.drop(columns=['idx'])

# convert dates to datetime object
df['join_date'] = pd.to_datetime(df['join_date'])
df['quit_date'] = pd.to_datetime(df['quit_date'])

# create new features join_year and quit_year, 
# dayes employed and quit (0 for still employed and 1 for quit)
df['join_year'] = df['join_date'].dt.year
df['quit_year'] = df['quit_date'].dt.year
df['days_employed']=df.apply(lambda row: get_employed(row),axis=1) 
df['quit'] = ~df['quit_date'].isnull()*1


# drop missing values 
df = df.dropna(subset=['salary', 'seniority'])

# prepare data to train classification model 
X = df[['dept','seniority','salary','days_employed']]
y = df['quit']
# one hot encode dept
X = pd.get_dummies(X,prefix=['dept'])

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = \
    train_test_split(X, y, test_size = 0.25, random_state = 42)

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
from sklearn.metrics import accuracy_score, f1_score
acc_score = accuracy_score(test_labels, predictions)
F1_sc = f1_score(test_labels, predictions)

print('Model Accuracy = ', acc_score)
print('F1 Score = ', F1_sc)

clf = RandomForestClassifier(random_state=42)
clf.fit(X,y)
#print(clf.feature_importances_)

# Feature importance
feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_imp)
ax = feature_imp.plot.bar()
ax.set_ylabel('Feature importance')

# plot histogram of the importance features for employees who quit and stay
plt.figure()
plt.hist(X[y==1]['days_employed'], alpha=0.5,  color  = 'r', label='Quit')
plt.hist(X[y==0]['days_employed'], alpha=0.5, color  = 'g', label='Stay')
plt.xlabel('# Days emplyed')
plt.ylabel('# Employees')
plt.legend()

plt.figure()
plt.hist(X[y==1]['seniority'], alpha=0.5, color  = 'r', label='Quit')
plt.hist(X[y==0]['seniority'], alpha=0.5, color  = 'g', label='Stay')
plt.xlabel('Seniority')
plt.ylabel('# Employees')
plt.legend()

plt.figure()
plt.hist(X[y==1]['salary'], alpha=0.5, color  = 'r', label='Quit')
plt.hist(X[y==0]['salary'], alpha=0.5, color  = 'g', label='Stay')
plt.xlabel('Salary')
plt.ylabel('# Employees')
plt.legend()


# plot probablity of an employee quiting as a function of number of days employed
n_bins=200
fig, ax = plt.subplots()
n, bins, patches = ax.hist(X[y==1]['days_employed'], 100, density=True,
                           histtype='step', cumulative=True )
ax.set_ylabel('P(Quit)')
ax.set_xlabel('# Days emplyed')