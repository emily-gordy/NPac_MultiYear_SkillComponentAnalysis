#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:28:19 2024

@author: egordon4
"""

import tensorflow as tf
import numpy as np

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
    # x=tf.keras.layers.MaxPooling2D((maxpools[0], maxpools[0]))(x)
    x=tf.keras.layers.MaxPooling2D((maxpools[0], maxpools[0]))(x)
    x=tf.keras.layers.BatchNormalization()(x)
    
    for conv in range(1,n_convs):
        x=tf.keras.layers.Conv2D(filters[conv], kernel_size[conv], activation='relu',
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed))(x)
        x=tf.keras.layers.MaxPooling2D((maxpools[conv], maxpools[conv]))(x)
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

