# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 08:42:50 2022

@author: vsthorkildsen
"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Add, Flatten, Dense, Reshape, BatchNormalization, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers




def build_unet_w_penalty(input_size=(None,None,1), penalty=0):
    inputs=Input(input_size)
    c1=Conv2D(16,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(inputs)
    c1=Conv2D(16,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c1)
    p1=MaxPool2D(pool_size=(2,2))(c1)

    
    c2=Conv2D(32,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(p1)
    c2=Conv2D(32,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c2)
    p2=MaxPool2D(pool_size=(2,2))(c2)
    
    
    
    c3=Conv2D(64,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(p2)
    c3=Conv2D(64,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c3)
    p3=MaxPool2D(pool_size=(2,2))(c3)


    c4=Conv2D(128,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(p3)
    c4=Conv2D(128,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c4)

    up1=Conv2DTranspose(64, (3,3), strides=(2,2),padding="same",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c4)
    up1 = Concatenate()([up1, c3])
    c5=Conv2D(64,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(up1)
    c5=Conv2D(64,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c5)
    
    up2=Conv2DTranspose(32, (3,3), strides=(2,2),padding="same",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c5)
    up2 = Concatenate()([up2, c2])
    c6=Conv2D(32,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(up2)
    c6=Conv2D(32,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c6)
    
    up3=Conv2DTranspose(16, (3,3), strides=(2,2),padding="same",kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c6)
    up3 = Concatenate()([up3, c1])
    c7=Conv2D(16,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(up3)
    c7=Conv2D(16,(3,3), activation='relu', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c7)
        
    c8=Conv2D(1,(1,1), activation='linear', padding='same',kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l=penalty), use_bias=True)(c7)
    
    model = Model(inputs=[inputs], outputs=[c8])
    model.summary(line_length=150)
    return model

