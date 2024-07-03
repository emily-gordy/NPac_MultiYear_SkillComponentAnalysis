#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:28:19 2024

@author: egordon4
"""

import tensorflow as tf
import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


def build_CNN_full(inputdata,outputdata,settings,random_seed):
    
    filters = settings["filters"]
    kernel_size = settings["kernel_size"]
    activation = settings["activation"]
    maxpools = settings["maxpools"]
    hiddens = settings["hiddens"]
    dropout_rate = settings["dropout_rate"]
    
    n_convs = len(filters)
    n_layers = len(hiddens)

    if len(inputdata.shape)==3:     
        inputshape = (inputdata.shape[1],inputdata.shape[2],1)
    elif len(inputdata.shape)==4:
        inputshape = (inputdata.shape[1],inputdata.shape[2],inputdata.shape[3])
    if len(outputdata.shape)>1:
        outputshape = (outputdata.shape[1],outputdata.shape[2])
        outputshape_flat = np.prod(outputshape)
    else:
        outputshape_flat = 1
    
    # define the model
    inputs = tf.keras.layers.Input(shape=inputshape) 

    # add convolutions    
    x=tf.keras.layers.Conv2D(filters[0], kernel_size[0], activation=activation,
                              kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed))(inputs)
    x=tf.keras.layers.AveragePooling2D((maxpools[0], maxpools[0]))(x)
    x=tf.keras.layers.BatchNormalization()(x)
    
    for conv in range(1,n_convs):
        x=tf.keras.layers.Conv2D(filters[conv], kernel_size[conv], activation='relu',
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed))(x)
        x=tf.keras.layers.AveragePooling2D((maxpools[conv], maxpools[conv]))(x)
        
        x=tf.keras.layers.BatchNormalization()(x)
    
    # flatten
    x = tf.keras.layers.Flatten()(x)
    
    # add dense layers
    x = tf.keras.layers.Dense(hiddens[0], activation=activation,
                # bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                kernel_regularizer=tf.keras.regularizers.L2(l2=settings["ridgepen"]))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate,name='d2')(x)

    for layer in range(1,n_layers):
        x = tf.keras.layers.Dense(hiddens[layer], activation=activation,
                        bias_initializer=tf.keras.initializers.Zeros(),
                        # bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
                        kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate,name='d_dense'+str(layer))(x)

    if len(outputdata.shape)>1:
        x = tf.keras.layers.Dense(outputshape_flat, activation="linear",
                        bias_initializer=tf.keras.initializers.Zeros(),
                        kernel_initializer=tf.keras.initializers.Zeros())(x)
        outputs = tf.keras.layers.Reshape(outputshape)(x)
    else:
        outputs = tf.keras.layers.Dense(outputshape_flat, activation="linear",
                        bias_initializer=tf.keras.initializers.Zeros(),
                        kernel_initializer=tf.keras.initializers.Zeros())(x)
        
    
    CNNmodel = tf.keras.Model(inputs=inputs,outputs=outputs,name='EmilysCNN')
    
    return CNNmodel


# class CNN(nn.Module):
#     def __init__(self, input_shape, output_shape, settings):
#         super(CNN, self).__init__()

#         self.filters = settings["filters"]
#         self.kernel_size = settings["kernel_size"]
#         self.activation = settings["activation"]
#         self.maxpools = settings["maxpools"]
#         self.hiddens = settings["hiddens"]
#         self.dropout_rate = settings["dropout_rate"]
#         self.ridgepen = settings["ridgepen"]
        
#         self.n_convs = len(self.filters)
#         self.n_layers = len(self.hiddens)

#         if len(input_shape) == 3:
#             self.inputshape = (1,input_shape[1], input_shape[2])
#         elif len(input_shape) == 4:
#             self.inputshape = (input_shape[1], input_shape[2], input_shape[3])
        
#         if len(output_shape) > 1:
#             self.outputshape = (output_shape[1], output_shape[2])
#             self.outputshape_flat = torch.prod(torch.tensor(self.outputshape))
#         else:
#             self.outputshape_flat = 1

#         # define the model
#         self.conv_layers = nn.ModuleList()
#         self.conv_layers.append(nn.Conv2d(self.inputshape[0], self.filters[0], 
#                                           self.kernel_size[0], padding="same",bias=False))
#         self.conv_layers.append(nn.AvgPool2d(self.maxpools[0]))
#         self.conv_layers.append(nn.BatchNorm2d(self.filters[0]))

#         for i in range(1, self.n_convs):
#             self.conv_layers.append(nn.Conv2d(self.filters[i-1], self.filters[i], 
#                                               self.kernel_size[i], padding="same",bias=False))
#             self.conv_layers.append(nn.AvgPool2d(self.maxpools[i]))
#             self.conv_layers.append(nn.BatchNorm2d(self.filters[i]))

#         self.flatten = nn.Flatten()

#         self.flatdims = reduce_pool(self.inputshape,self.filters,self.maxpools)

#         self.dense_layers = nn.ModuleList()
#         self.dense_layers.append(nn.Linear(int(self.flatdims[0]*self.flatdims[1]*self.flatdims[2]), self.hiddens[0]))
#         self.dense_layers.append(nn.BatchNorm1d(self.hiddens[0]))
#         # self.dense_layers.append(nn.Dropout(self.dropout_rate))

#         for i in range(1, self.n_layers):
#             self.dense_layers.append(nn.Linear(self.hiddens[i-1], self.hiddens[i]))
#             self.dense_layers.append(nn.BatchNorm1d(self.hiddens[i]))
#             # self.dense_layers.append(nn.Dropout(self.dropout_rate))

#         self.output_layer = nn.Linear(self.hiddens[-1], self.outputshape_flat)
#         # nn.init.zeros_(self.output_layer.weight)

#     def forward(self, x):
#         for layer in self.conv_layers:
#             x = F.relu(layer(x))
        
#         x = self.flatten(x)
        
#         for layer in self.dense_layers:
#             x = F.relu(layer(x))

#         x = self.output_layer(x)

#         if len(self.outputshape) > 1:
#             x = x.view(-1, *self.outputshape)
        
#         return x

# # Usage:
# # Create an instance of the CNN model
# # model = CNN(input_shape, output_shape, settings, random_seed)
# # input_tensor = torch.randn(batch_size, *input_shape) # Example input tensor
# # output = model(input_tensor) # Output tensor

# def reduce_pool(inputshape,filters,pools):
#     # only works for square pooling
#     dimreduce = []
#     dimreduce.append([filters[0],inputshape[1]//pools[0],inputshape[2]//pools[0]])

#     for i in range(len(pools)-1):
#         dimreduce.append([filters[i+1],dimreduce[i][1]//pools[i+1],dimreduce[i][2]//pools[i+1]])

#     return dimreduce[-1]



