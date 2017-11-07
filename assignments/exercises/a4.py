""" for the heart prediction
"""

# coding: utf-8

# In[364]:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import utils
import time

# Step 1: read in data from the .csv file
df=pd.read_csv('../../data/heart.csv', usecols = [0,1,2,3,4,5,6,7,8],skiprows = [0],header=None)
l = pd.read_csv('../../data/heart.csv',usecols = [9],skiprows = [0],header=None)


# In[365]:

convert = {'Present': 1, 'Absent' : 0}


# In[366]:

df.iloc[:,4] = df.iloc[:,4].map(convert)


# In[367]:

data = np.float32(df.values)
labels = np.int64(l.values)
labels = pd.get_dummies(labels.reshape(1,len(labels))[0]).values


# In[368]:

rows = len(data)


# In[369]:

X = tf.placeholder(tf.float32, [1,9], name='X')
Y = tf.placeholder(tf.int64, [1,2], name='Y')


# In[370]:

w = tf.Variable(tf.random_normal(shape=[9, 2], stddev=0.01), dtype = tf.float32, name = 'w')
b = tf.Variable(tf.zeros([1,2]), name = 'b')


# In[371]:

logits = tf.matmul(X,w) + b


# In[372]:

entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y, name = 'loss')
loss = tf.reduce_mean(entropy)


# In[373]:

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


# In[374]:

with tf.Session() as sess:
	start_time = time.time()
	# Step 7: initialize the necessary variables, in this case, w and b
	# TO - DO	
	sess.run(w.initializer)
	sess.run(b.initializer)
	# Step 8: train the model
	for i in range(50): # run 100 epochs
		total_loss = 0
		for j in range(rows):
			# Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
			# TO DO: write sess.run()
			_,l = sess.run([optimizer,loss], {X:data[j].reshape(1,9), Y:labels[j].reshape(1,2)})     
			total_loss += l
		print("Epoch {0}: {1}".format(i, total_loss/rows))
	w, b = sess.run([w, b])
	print('Total time: {0} seconds'.format(time.time() - start_time))
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
	n_batches = 1
	total_correct_preds = 0
	for k in range(rows):
		X_batch, Y_batch = data[k].reshape(1,9), labels[k].reshape(1,2)
		accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
		total_correct_preds += sess.run(tf.reduce_mean(accuracy_batch))	
	
	print('Accuracy {0}'.format(total_correct_preds/rows))

