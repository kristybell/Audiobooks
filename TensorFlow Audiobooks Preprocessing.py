#!/usr/bin/env python
# coding: utf-8

# # Practical Example: Audiobooks

# ### Preprocess the data. Balance the dataset. Create 3 datasets: training, validation, and test. Save the newly created sets in a tensor friendly format (e.g. *.npz)

# Since we are dealing with real life data, we will need to preprocess it a bit. This is the relevant code, which is not that hard, but refers to data engineering more than machine learning.
# 
# If you want to know how to do that, go through the code and the comments. In any case, this should do the trick for all datasets organized in the way: many inputs, and then 1 cell containing the targets (all supervized learning datasets).
#     
# Note that we have removed the header row, which contains the name of the categories. We simply want the data.

# ### Extract the Data from the .csv File

# In[1]:


import numpy as np
from sklearn import preprocessing  # use sklearn capabilities for standardizing the inputs

# load the .csv file
raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter =',')

unscaled_inputs_all = raw_csv_data[:,1:-1]  # [rows : columns] takes all columns excluding the i.d. and the targets (0 column and the last one (minus first))

# record the targets 
targets_all = raw_csv_data[:,-1]  # the last column contains the targets


# ### Balance the Dataset

# In[2]:


# 1. count the number of targets that are '1'
# 2. keep as many 0s as there are 1s

# if we sum all the targets, we will get the number of targets that are 1s
num_one_targets = int(np.sum(targets_all))

# set a counter for targets which are 0s
zero_targets_counter = 0

# record the indices to be removed which for now is empty but we want it to be a list or a tuple so we put empty brackets to iterate over the data set and balance it 'i' in range targets
indices_to_remove = []

# the shape of 'targets_all' on axis = 0 , is basically the length of the vector
# if the target at position 'i' is 0, and the number of zeroes is bigger than the number of 1s, we'll know the indices of all data points to be removed
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
       zero_targets_counter += 1
       if zero_targets_counter > num_one_targets:
          indices_to_remove.append(i)
        
# 'np.delete(array, obj to delete, axis)' -> a method that deletes an object along an axis
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis = 0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis = 0)


# ### Standardize the Inputs

# In[3]:


# 'preprocessing.scale(X)' -> a method that standardizes an array along an axis
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)


# ### Shuffle the Data

# In[4]:


# a little trick is to shuffle the inputs and the targets
# keeps the same information but in a random order
# since we are batching, we must shuffle the data 
# if ordered by date: inside a batch is homogeneous, but between batches it is very heterogeneous
# this will confuse the stochastic gradient descent


# In[5]:


# 'np.arange([start],stop)' -> a method that returns evenly spaced values within a given interval
shuffled_indices = np.arange(scaled_inputs.shape[0])

# 'np.random.shuffle(X)' -> a method that shuffles the numbers in a given sequence
np.random.shuffle(shuffled_indices)

# create shuffled inputs and targets to the scaled inputs and targets
shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]


# ### Split the Dataset into Train, Validation, and Test

# In[10]:


samples_count = shuffled_inputs.shape[0]

# using the 80 - 10 - 10 split
# make sure values are integers
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

# extract from the big data set
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[:train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[:train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# useful to check if we have balanced the dataset
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)


# ### Save the Three Datasets in *.npz

# In[11]:


np.savez('Audiobooks_data_train', inputs = train_inputs, targets = train_targets)
np.savez('Audiobooks_data_validation', inputs = validation_inputs, targets = validation_targets)
np.savez('Audiobooks_data_test', inputs = test_inputs, targets = test_targets)


# In[12]:


# each time we reun the code in this sheet, we will preprocess the data once again (forgetting the previous preprocessing)


# In[ ]:


# This code can be used to preprocess any dataset that has two classes
# To use this code to preprocess a dataset with more than two classifiers, must balance the data set classes instead of two

