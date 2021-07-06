# Copyright 2021 Nokia
# Licensed under the Creative Commons Attribution 
# Non Commercial 4.0 International License
# SPDX-License-Identifier: CC-BY-NC-4.0


import tensorflow as tf
import numpy as np

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense          
        
    
def mixup_augmenter( x_train, y_train, batch_size=64, alpha = 0.2 ):
    
    N_half_batch = int(batch_size/2)
    
    x_train_01 = x_train[:N_half_batch]
    y_train_01 = y_train[:N_half_batch]
    x_train_02 = x_train[N_half_batch:]
    y_train_02 = y_train[N_half_batch:]
    
    lam = np.random.beta(alpha,alpha)
    x_mixup = lam*x_train_01 + (1.-lam)*x_train_02
    y_mixup = lam*y_train_01 + (1.-lam)*y_train_02
    
    return (x_mixup, y_mixup)


def cross_entropy( y_true, y_pred ):
    '''
    Cross entropy loss for not necessary one-hot encoded labels.
    '''
    
    y_pred_ = tf.clip_by_value(y_pred,1e-10,1.0)
    
    ce_loss = -tf.reduce_mean( tf.reduce_sum( tf.multiply(tf.cast(y_true,tf.float32),tf.log(y_pred_)), axis=1 ) )
    
    return ce_loss
    

def define_plain_cnn(input_shape=(32,32,1), num_classes=40, mixup=False):
    
    inp = Input(shape=input_shape)
    
    conv_01 = Conv2D( filters=32, kernel_size=(8,8), strides=(2,2), padding='same' )(inp)
    conv_02 = Conv2D( filters=64, kernel_size=(5,5), strides=(1,1), padding='same' )(conv_01)
    conv_03 = Conv2D( filters=128, kernel_size=(5,5), strides=(2,2), padding='same' )(conv_02)
    conv_04 = Conv2D( filters=256, kernel_size=(3,3), strides=(2,2), padding='same' )(conv_03)
    
    gap =  GlobalAveragePooling2D()(conv_04)
    
    out = Dense( units=num_classes, activation='softmax' )(gap)
    
    classificator = Model(inputs=inp, outputs=out)
    
    loss = None
    
    if mixup:
        loss = cross_entropy
    else:
        loss='categorical_crossentropy'
    
    classificator.compile( loss=loss, optimizer='Adam', metrics=['accuracy'] )
    
    return classificator  



def train_network( network_model, train_images, train_labels, test_images, test_labels, num_of_epochs=1000, batch_size=500, mixup=False ):
    
    sample_num = train_images.shape[0]
    max_accuracy = 0
    
    for epoch in range(num_of_epochs):
        
        idx = np.linspace(0,sample_num-1,sample_num).astype('int32')
        np.random.shuffle(idx)
        train_images = train_images[idx]
        train_labels = train_labels[idx]
        
        it = 0
        while it <= sample_num-batch_size:
            
            current_train_set = train_images[it:it+batch_size]
            current_train_labels = train_labels[it:it+batch_size]
            
            if mixup:
                current_train_set, current_train_labels = mixup_augmenter( current_train_set, current_train_labels, batch_size=batch_size, alpha = 0.2 )
            
            
            network_model.train_on_batch( current_train_set, current_train_labels )
            
            it = it + batch_size
            
        if epoch%5 == 0:
            
            loss, accuracy = network_model.evaluate( test_images, test_labels )
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                
            print( epoch, loss, accuracy )
            
    return max_accuracy
