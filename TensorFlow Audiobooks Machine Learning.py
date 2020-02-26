#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 50 hidden units in the hidden layers provide enough complexity
# dont want to put too many units initially as we want to complete the learning as fast as possible


# # Practical Example: Audiobooks

# ## Problem

# You are given data from an Audiobook app. Logically, it related only to the audio version of books. Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company.
# 
# The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertising to him/her. If we can focus our efforts ONLY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.
# 
# You have a .csv summarizing the data. There are several variables: Customer ID, Book length in mins_avg (average of all purchases, Book length in minutes_sum (sum of all purchases, Proce Paid_avg (average of all purchases), Price paid_sum (sum of all purchases, Review (a Boolean variable, Review (out of 10), Total minutes listened, Completion (from 0 to 1), Support requests (number), and Last visited minus purchase date (in days).
# 
# So there are the inputs (excluding customer ID, as it is completely arbitrary. It's more like a name, than a number).
# 
# The targets are a Boolean variable (so 0 or 1). We are taking a period of 2 years in our iputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months, 6 months sound like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information.
# 
# The task is simple: create a machine learning algorithm, which is able to predict if a customer will buy again.
# 
# This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s.

# ## Create the Machine Learning Algorithm

# ### Import the Relevant Libraries

# In[2]:


import numpy as np
import tensorflow as tf


# ### Data

# In[3]:


# load the data that was saved in the Preprocessing Algorithm as *.npz in 2 tuple form [inputs, targets]
# data to be loaded into a temporary variable 'npz' that is in an array form
npz = np.load('Audiobooks_data_train.npz')

# extract into a new variable
# expect all inputs to be floats
# 'np.ndarray.astype()' -> creates a copy of the array, cast to a specific type
train_inputs = npz['inputs'].astype(np.float)
# our targets are only 0s and 1s, but we are not completely certain about their data type
train_targets = npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('Audiobooks_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


# ### Model

# Outline, optimizers, loss, early stopping and training

# In[4]:


# copy model from MNITS problem and make necessary adjustments

input_size = 10   # 10 predictors
output_size = 2   # targets are either '0s' of '1s'
hidden_layer_size = 50 

model = tf.keras.Sequential([
#                            tf.keras.layers.Flatten(input_shape = (28,28,1)),             removed this line of code for the data has been preprocessed
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(output_size,activation='softmax')
                            ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100

max_epochs = 100

# by default, this object will monitor the validation loss and stop the training process the first time the validation loss starts increasing
# 'tf.keras.callbacks.EarlyStopping(patience)' -> configures the early stopping mechanism of the algorithm
# 'patience' -> let's us decide how many consecutive increase we can tolerate
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
model.fit(train_inputs,
         train_targets,
         batch_size = batch_size,
         epochs = max_epochs,
         callbacks = [early_stopping],
         validation_data = (validation_inputs, validation_targets),
         verbose = 2)


# In[5]:


# val_loss increases and then decreases, thus probably overfitted
# need to set u an early stopping mechanism with TensorFlow
# CALLBACKS -> functions called at certain points during model training
# declare a new parameter before fitting the model and call it 'early_stopping'

# val_loss consistently decreases until the last epoch, but only increase slightly (0.044)
# rerun the code using 'patience=2'


# In[6]:


# the first final val_accuracy has increased, so the machine has learned

# we too raw data:
# 1. should not make a lot of sense to anyone who is not in this business
# 2. where many variables were binary
# 3. had missing values
# 4. where the order of magnitude had nothing in common

# it is extremely hard to predict human behavior
# the ML algorithm we create here is a new tool in your arsenal that has given you an incredible edge
# it is a skill that an easily apply in any business out there
# have leveraged AI to reach a business insight


# ### Test the Model

# In[7]:


# declare two variables
# 'model.evaluate()' -> returns the loss value and metircs values for the model in 'test mode'
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets) # returns the final accuracy of the model


# In[8]:


print('\nTest Loss: {0:.2f}. Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))


# In[9]:


# naturally, the final accuracy is close to the validation accuracy as we didn't fiddle too much with the hyperparameters
# Note: sometimes one can get a test accuracy higher than the vali


# In[ ]:




