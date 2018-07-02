# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 03:43:10 2017

@author: Bhumika and Sunita
"""

import zipfile
import os
from PIL import Image
import numpy as np
import tensorflow  as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.filterwarnings("ignore")

#Print details of each team member
print("UBitName\t=\tBhumika Khatwani\t\tSunita Pattanayak")
print("personNumber\t=\t50247656\t\t\t50249134")
#Extract MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#initialising hyperparams
K = 10
iterations = 100
learningRate = 0.00001
lam = 0.5852
batch_size = 500
input_size = 784 #input 
number_of_neurons = 200 # hidden layer
output_size = 10 # output layer
batch_size_nn=500 
learning_rate_nn=0.01
iterations_nn = 100


W = np.zeros((input_size, K))

#Defining MNIST image and label arrays for train,test,validate data
mnist_train_labels = np.array(mnist.train.labels)
mnist_train_images =  np.array(mnist.train.images)
mnist_valid_images =  np.array(mnist.validation.images)
mnist_valid_labels =  np.array(mnist.validation.labels)
mnist_test_labels =  np.array(mnist.test.labels)
mnist_test_images =  np.array(mnist.test.images)

#Extracting file from USPS dataset
filename="proj3_images.zip"

#Defining height,width for resizing the images to 28x28 like MNIST digits
height=28
width=28

#Defining path for extracting dataset zip file
extract_path = "usps_data"

#Defining image,label list
images = []
img_list = []
labels = []

#method for softmax regression
def softmax(z):
    e = (z - np.max(z))
    return (np.exp(e).T/(np.sum(np.exp(e), axis=1))).T

#calculating negative log likelihood for cross entropy   
def negative_log_likelihood(w,x,y):
    ynk = softmax(np.dot(x,w))
    tnk = y
    cross_entropy = -np.sum((tnk)*np.log(ynk))
    return cross_entropy

#calculate gradient descent
def gradient(w,x,y,l,lam):
    ynk = softmax(np.dot(x,w))
    grad = - np.dot(x.T,(y - ynk)) + lam*w
    w = w - l*grad
    return w

#method to calculate accuracy
def getAccuracy(x,y,w):
    prob = softmax(np.dot(x,w))
    m = x.shape[0]
    accuracy = 0
    for i in range(m):
        predict = np.argmax(prob[i])
        if(predict == y[i].tolist().index(1)):
            accuracy = accuracy + 1
    return accuracy/m

#CNN using tensor flow
def tf_conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def tf_max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def tf_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def tf_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Extracting given dataset file    
with zipfile.ZipFile(filename, 'r') as zip:
    zip.extractall(extract_path)

#Extracting labels,images array needed for training    
for root, dirs, files in os.walk("."):
    path = root.split(os.sep)
        
    if "Numerals" in path:
        image_files = [fname for fname in files if fname.find(".png") >= 0]
        for file in image_files:
            labels.append(int(path[-1]))
            images.append(os.path.join(*path, file)) 

#Resizing images like MNIST dataset   
for idx, imgs in enumerate(images):
    img = Image.open(imgs).convert('L') 
    img = img.resize((height, width), Image.ANTIALIAS)
    img_data = list(img.getdata())
    img_list.append(img_data)

#Storing image and labels in arrays to be used for training   
USPS_img_array = np.array(img_list)
USPS_img_array = np.subtract(255, USPS_img_array)
USPS_label_array = np.array(labels)

#converting into one_hot array with int datatype
nb_classes = 10
targets = np.array(USPS_label_array).reshape(-1)
aa = np.eye(nb_classes)[targets]
USPS_label_array = np.array(aa, dtype=np.int32)

#Logical regression using softmax
losses = []
for i in range(0,iterations):
    loss = negative_log_likelihood(W,mnist_train_images,mnist_train_labels)
    losses.append(loss)
    W =  gradient(W, mnist_train_images, mnist_train_labels, learningRate, lam)

print("Graph for Logistic Regression :")  
plt.plot(losses)
plt.show()

print("\n \n Accuracy for logistic regression :")
print("MNIST Training Accuracy",getAccuracy(mnist_train_images,mnist_train_labels,W))
print("MNIST Test Accuracy",getAccuracy(mnist_test_images,mnist_test_labels,W))
print("MNIST Validation Accuracy",getAccuracy(mnist_valid_images,mnist_valid_labels,W))

print("USPS Test Accuracy",getAccuracy(USPS_img_array,USPS_label_array,W))

#Single neural network
nn_mnist_train_labels = tf.placeholder(tf.float32, [None, 10])
nn_mnist_train_images = tf.placeholder(tf.float32, [None, 784])

# weights and bias for hidden layer
nn_W1 = tf.Variable(tf.random_normal([input_size, number_of_neurons], stddev=0.3), name = 'W1')
nn_B1 = tf.Variable(tf.random_normal([number_of_neurons]))
# weights and bias for output layer
nn_W2 = tf.Variable(tf.random_normal([number_of_neurons, output_size], stddev=0.3), name = 'W2')
nn_B2 = tf.Variable(tf.random_normal([output_size]))

# output of the hidden layer
input_hidden = tf.add(tf.matmul(nn_mnist_train_images, nn_W1), nn_B1)
output_hidden = tf.sigmoid(input_hidden)

# softmax function
predict =  tf.nn.softmax(tf.add(tf.matmul(output_hidden,nn_W2),nn_B2))

# Cross Entropy error function
predict_adjusted = tf.clip_by_value(predict, 1e-20, 0.99999999)
cross_entropy = -tf.reduce_sum(nn_mnist_train_labels * tf.log(predict_adjusted))

# gardient calculation
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_nn).minimize(cross_entropy)

with tf.Session() as session:
   # initialise the variables
   session.run(tf.global_variables_initializer())
   number_of_batch = int(len(mnist.train.labels)/batch_size_nn)
   for i in range(iterations_nn):
        avg_cost = 0
        for j in range(number_of_batch):
            batch_images, batch_labels = mnist.train.next_batch(number_of_batch)
            _, cost = session.run([gradient_descent, cross_entropy], 
                        feed_dict={nn_mnist_train_images: batch_images, nn_mnist_train_labels: batch_labels})
        avg_cost = cost / number_of_batch
        
   count_predictions = tf.equal(tf.argmax(nn_mnist_train_labels, 1), tf.argmax(predict, 1))
   get_accuracy = tf.reduce_mean(tf.cast(count_predictions, tf.float32))
   
   print("\n \n Accuracy for SNN :")
   print("MNIST test accuracy : ")    
   print(session.run([get_accuracy], feed_dict={nn_mnist_train_images: mnist.test.images, nn_mnist_train_labels: mnist.test.labels}))
   print("USPS test accuracy : ")
   print(session.run([get_accuracy], feed_dict={nn_mnist_train_images: USPS_img_array, nn_mnist_train_labels: USPS_label_array}))

#Convolutional Neural network
#first convolutional layer
x_p = tf.placeholder(tf.float32, shape=[None, 784])
W_conv1 = tf_weight_variable([5, 5, 1, 32])
b_conv1 = tf_bias_variable([32])
x_image = tf.reshape(x_p, [-1,28,28,1])
h_conv1 = tf.nn.relu(tf_conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf_max_pool_2x2(h_conv1)
    
#second convolutional layer
W_conv2 = tf_weight_variable([5, 5, 32, 64])
b_conv2 = tf_bias_variable([64])
h_conv2 = tf.nn.relu(tf_conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tf_max_pool_2x2(h_conv2)
    
#Densely Connected Layer
W_fc1 = tf_weight_variable([7 * 7 * 64, 1024])
b_fc1 = tf_bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
#Readout
W_fc2 = tf_weight_variable([1024, 10])
b_fc2 = tf_bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
#Training
y_p = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_p, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_p, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for idx in range(iterations):
        batch = mnist.train.next_batch(batch_size)
        if idx % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_p: batch[0],y_p: batch[1],keep_prob: 1.0})            
        train_step.run(feed_dict={x_p: batch[0], y_p: batch[1],keep_prob: 0.5})
    print("\n \n Accuracy for CNN :")
    print('MNIST test accuracy : %g' % accuracy.eval(feed_dict={x_p: mnist.test.images,y_p: mnist.test.labels,keep_prob: 1.0}))
    print('USPS test accuracy  :  %g' % accuracy.eval(feed_dict={x_p: USPS_img_array,y_p: USPS_label_array,keep_prob: 1.0}))
