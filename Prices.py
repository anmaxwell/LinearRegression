#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:14:25 2019

@author: account1
"""

#%%

# imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# this allows plots to appear directly in the notebook FOR NOTEBOOK ONLY!!!
# %matplotlib inline

#%%

# read data into a DataFrame
data = pd.read_csv('Prices.csv', index_col=0)
data.head()

#%%

# print the shape of the DataFrame
data.shape


#%%

# look at the correlation of all the columns (1 shows direct correlation)
data.corr()

#%%

# visualize the relationships using scatterplots
fig, ax = plt.subplots(1, 5, sharey=True, figsize=(12, 4))
data.plot(kind='scatter', x='Area', y='Cost', ax=ax[0])
data.plot(kind='scatter', x='Bed1', y='Cost', ax=ax[1])
data.plot(kind='scatter', x='Region', y='Cost', ax=ax[2])
data.plot(kind='scatter', x='Bed', y='Cost', ax=ax[3], xticks=(3,4))
data.plot(kind='scatter', x='Det', y='Cost', ax=ax[4], xticks=(0,1))

#%%

#print(ax[1])

#%%
    
fig, ax = plt.subplots(1, 1, sharey=True, figsize=(4, 4))
data[data.Det==1].plot(kind='scatter', x='Bed', y='Cost', ax=ax, color="Red", 
    marker='x', label='detached')
data[data.Det==0].plot(kind='scatter', x='Bed', y='Cost', ax=ax, xticks=(3,4), 
    marker='.', label='semi')


#%%

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 8))
data[data.Det==1].plot(kind='scatter', x='Region', y='Cost', ax=ax, 
    color="Red", marker='x', label='detached')
data[data.Det==0].plot(kind='scatter', x='Region', y='Cost', ax=ax, 
    xticks=(3,4), marker='.', label='semi')



#%%

# split the data to training and test sets using 70% train to 30% test data
train_set_full, test_set = train_test_split(data, test_size=0.3)
train_set_full.head()

# create a copy of the training set to play with
train_set=train_set_full.copy()

#%%

#create the input matrix and the output for training
train_output=train_set["Cost"]
train_input=train_set.drop(["Cost"], axis=1)

#%%

#look at the features
train_input.head()

#%%

#look at the 
train_output.head()

#%%

lin_reg = LinearRegression()
lin_reg.fit(train_input, train_output)

#%%

print("Coefficients: ", lin_reg.coef_)
print("Intercept: ", lin_reg.intercept_)


#%%

#create the input matrix and the output for testing
test_output=test_set["Cost"]
test_input=test_set.drop(["Cost"], axis=1)


#%%

#find the score of the model for the test set
lin_reg.score(test_input, test_output)


#%%

#read in a csv for a new house to see what the prediction will be
new_house = pd.read_csv('new_house.csv', index_col=0)
lin_reg.predict(new_house)

#%%

#to look at the difference between the predicted and actual values for the 
#test set

pred = lin_reg.predict(test_input)

for i in zip(test_output, pred):
    print(i)



