# Copyright 2021 Nokia
# Licensed under the Creative Commons Attribution 
# Non Commercial 4.0 International License
# SPDX-License-Identifier: CC-BY-NC-4.0

from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import ReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
from keras.utils import to_categorical
import numpy as np


def define_encoder():
    
    encoder = Sequential()
    encoder.add( Conv2D(32,(3,3),strides=(2,2),padding='same',input_shape=(32,32,1)) )
    encoder.add( ReLU() )
    
    encoder.add( Conv2D(32,(3,3),strides=(1,1),padding='same') )
    encoder.add( BatchNormalization() )
    encoder.add( ReLU() )
    
    encoder.add( Conv2D(64,(3,3),strides=(2,2),padding='same') )
    encoder.add( BatchNormalization() )
    encoder.add( ReLU() )
    
    encoder.add( Conv2D(64,(3,3),strides=(1,1),padding='same') )
    encoder.add( BatchNormalization() )
    encoder.add( ReLU() )
    
    encoder.add( Conv2D(128,(3,3),strides=(2,2),padding='same') )
    encoder.add( BatchNormalization() )
    encoder.add( ReLU() )
    
    return encoder


def define_decoder():
    
    decoder = Sequential()
    decoder.add( Conv2DTranspose(64, (3,3), strides=(2,2), padding='same') )
    decoder.add( BatchNormalization() )
    decoder.add( ReLU() )
    
    decoder.add( Conv2DTranspose(64,(3,3), strides=(1,1),padding='same') )
    decoder.add( BatchNormalization() )
    decoder.add( ReLU() )
    
    decoder.add( Conv2DTranspose(32,(3,3), strides=(2,2),padding='same') )
    decoder.add( BatchNormalization() )
    decoder.add( ReLU() )
    
    decoder.add( Conv2DTranspose(32,(3,3), strides=(1,1),padding='same') )
    decoder.add( BatchNormalization() )
    decoder.add( ReLU() )
    
    decoder.add( Conv2DTranspose(1,(3,3), strides=(2,2),padding='same') )
    decoder.add( Activation('tanh') )
    
    return decoder


def define_autoencoder(encoder, decoder):
    
    decoder_output = decoder(encoder.output)
    autoencoder = Model( inputs=encoder.input, outputs=decoder_output )
    autoencoder.compile( optimizer='Adam', loss='mean_squared_error', metrics=['mean_squared_error'] )
    
    return autoencoder


def define_generator(decoder):
    
    latent_dim = 4*4*128
    latent_input_rep = Input(shape=(latent_dim,))
    decoder_input = Reshape((4,4,128))(latent_input_rep)
    generated_output = decoder(decoder_input)
    
    generator = Model( inputs=latent_input_rep, outputs=generated_output )
    
    return generator


def define_discriminator(encoder,class_num=40):
    disc = Flatten()(encoder.output)
    disc = Dense(units=(class_num+1))(disc)
    disc = Activation('softmax')(disc)
    discriminator = Model( inputs=encoder.input, outputs=disc )
    
    discriminator.trainable = True
    discriminator.compile( optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    
    return discriminator

def define_gan( generator, discriminator ):
    
    disc_output = discriminator(generator.output)
    gan_model = Model( inputs=generator.input, outputs=disc_output )
    
    discriminator.trainable = False
    gan_model.compile( optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    
    return gan_model


def class_conditional_latent_vector_generator( latent_features, labels ):
    
    classes = np.unique( labels )
    num_of_classes = classes.shape[0]
    
    means = np.zeros((num_of_classes,np.shape(latent_features)[1],np.shape(latent_features)[2],np.shape(latent_features)[3]))
    deviations = np.zeros((num_of_classes,np.shape(latent_features)[1],np.shape(latent_features)[2],np.shape(latent_features)[3]))
    
    for idx, current_class in np.ndenumerate(classes):
        current_idx = np.where( labels==current_class )
        current_features = latent_features[current_idx]
        means[idx[0]] = np.mean( current_features, axis=0 )
        deviations[idx[0]] = np.std( current_features, axis=0 )
        
    return means, deviations

def rebalance_counts( y_train ):
    '''
    Determine the number of instances to be generated from each class in order to rebalance the data set
    '''
    unique_entries, counts = np.unique(y_train,return_counts=True)
    N = np.shape(y_train)[0]
    
    max_arg = np.argmax(counts)
    max_val = counts[max_arg]
    count_to_be_generated = [ max_val-counts[i] for i in unique_entries ]
    
    init_sum = np.sum(counts) + np.sum(count_to_be_generated)
    
    diff = 2*N - init_sum
    
    extra_count_to_be_generated = int(np.divide(diff,np.shape(counts)[0]))
    
    count_to_be_generated = [ val+extra_count_to_be_generated for val in count_to_be_generated ]
    
    idx = []
    for ind, val in np.ndenumerate(count_to_be_generated):
        idx += [unique_entries[ind[0]]]*val
    
    return idx
    

def generate_fake_latent_samples( num_of_classes, sample_size, latent_dim, latent_means, latent_std, y_train=None, is_balancing=False ):
    
    if is_balancing:
        idx = rebalance_counts( y_train=y_train )
        sample_size = np.shape(idx)[0]
    else:
        idx = np.random.randint(0, num_of_classes, sample_size)
        
    fake_latent_vectors = np.multiply(np.random.randn(sample_size, latent_dim),np.reshape(latent_std[idx],(sample_size,latent_dim))) + np.reshape(latent_means[idx],(sample_size,latent_dim))
    
    return fake_latent_vectors, idx

def gan_training(generator, discriminator, gan_model, images, labels, latent_means, latent_std, num_of_epochs, num_of_batches, latent_dim ):
    
    num_of_classes = np.unique(labels).shape[0]
    num_of_steps = num_of_epochs*num_of_batches
    batch_size = int(images.shape[0]/num_of_batches)
    half_batch_size = int( batch_size/2 )
    
    generator.trainable = True
    
    for step in range(num_of_steps):
        
        print(step,'/',num_of_steps)
        
        # train discriminator on real data
        '''discriminator.trainable = True
        discriminator.compile( optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'] )'''
        
        idx = np.random.randint( 0, images.shape[0], half_batch_size )
        current_images = images[idx]
        labels_real = labels[idx]
        current_labels_real = to_categorical(labels_real,num_of_classes+1)
        discriminator.train_on_batch( current_images, current_labels_real )
        
        # train discriminator on fake data
        fake_latent_vectors, _ = generate_fake_latent_samples( num_of_classes, half_batch_size, latent_dim, latent_means, latent_std )
        fake_images = generator.predict( fake_latent_vectors )
        labels_fake = np.zeros( (half_batch_size,1) ) + num_of_classes
        discriminator.train_on_batch( fake_images, to_categorical(labels_fake,num_of_classes+1) )
        
        # train gan (discriminator weights are fixed)
        '''discriminator.trainable = False'''
        '''gan_model.compile( optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'] )'''
        
        fake_latent_vectors, classes = generate_fake_latent_samples( num_of_classes, half_batch_size, latent_dim, latent_means, latent_std )
        fake_images = generator.predict( fake_latent_vectors )
        labels_fake = to_categorical( classes, num_of_classes+1 )
        gan_model.train_on_batch(fake_latent_vectors,labels_fake)
