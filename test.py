# Copyright 2021 Nokia
# Licensed under the Creative Commons Attribution 
# Non Commercial 4.0 International License
# SPDX-License-Identifier: CC-BY-NC-4.0


import numpy as np
import keras
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from numpy import linalg as LA
from multiprocessing import Pool
from keras import backend as K
import autoencoder as dcae
import cnn
import acgan
import resnet50
import bagan
import randaugment

def train_data_format(train_images,train_labels,N):

    indices = np.random.choice( a=train_images.shape[0], size=N, replace=False )

    train_set_images = train_images[indices]
    train_set_labels = train_labels[indices]

    return (train_set_images,train_set_labels)


def _test_basic(train_set_images,train_set_labels,
               extended_train_set_images, extended_train_set_labels,
               test_images, test_labels,is_resnet=False):
    
    import tensorflow as tf
    
    train_set_labels_cat = keras.utils.to_categorical(train_set_labels, 43); 
    extended_train_set_labels_cat=keras.utils.to_categorical(extended_train_set_labels,43);
    test_labels_cat = keras.utils.to_categorical(test_labels, 43);
    
    if not is_resnet:
        
        classifier_01 = cnn.define_plain_cnn( input_shape=(32,32,3), num_classes=43 )
        accuracy_base = cnn.train_network( classifier_01, train_set_images, train_set_labels_cat, test_images, test_labels_cat, num_of_epochs=0, batch_size=500, mixup=False )
        
        classifier_02 = cnn.define_plain_cnn( input_shape=(32,32,3), num_classes=43 )
        accuracy_augmented = cnn.train_network( classifier_02, extended_train_set_images, extended_train_set_labels_cat, test_images, test_labels_cat, num_of_epochs=1000, batch_size=500, mixup=False )
    
    else:
        classifier_01 = resnet50.define_resnet50( input_shape=(32,32,3), num_classes=43 )
        accuracy_base = resnet50.train_network( classifier_01, train_set_images, train_set_labels_cat,test_images, test_labels_cat, num_of_epochs=0, batch_size=500 )
    
        classifier_02 = resnet50.define_resnet50( input_shape=(32,32,3), num_classes=43 )
        accuracy_augmented = resnet50.train_network( classifier_02, extended_train_set_images, extended_train_set_labels_cat, test_images, test_labels_cat, num_of_epochs=2000, batch_size=500 )

    return accuracy_base, accuracy_augmented

def _augmenter(train_set_images, train_set_labels, M):
    '''
    Run data set augmentation using the deep guided interpolation approach.
    '''
    import tensorflow as tf

    rnd_st = np.random.randint(1,100); name='augmenter_network_' + str(rnd_st)
    network = dcae.autoencoder(name=name,image_set_01=train_set_images,image_set_02=train_set_images,image_set_02_labels=train_set_labels)
    network.optimize( num_of_epochs_phase_01 = 15000, num_of_epochs_phase_02=25000, num_of_epochs_phase_03=0, learning_rate_phase_01=0.001, learning_rate_phase_02=0.001,)
    new_images, new_labels = network.augmentation(model_to_restore=name+'_ae_se_final', num_of_samples=M)
    
    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    return (extended_train_set_images, extended_train_set_labels)

def _augmenter_random_interpolation(train_set_images, train_set_labels, M):
    '''
    Run data set augmentation by randomly interpolating latent space representations of data points.
    '''
    import tensorflow as tf

    rnd_st = np.random.randint(1,100); name='augmenter_network_' + str(rnd_st)
    network = dcae.autoencoder(name=name,image_set_01=train_set_images,image_set_02=train_set_images,image_set_02_labels=train_set_labels)
    network.optimize( num_of_epochs_phase_01 = 15000, num_of_epochs_phase_02=0, num_of_epochs_phase_03=0, learning_rate_phase_01=0.001, learning_rate_phase_02=0.001,)
    new_images, new_labels = network.random_augmentation(model_to_restore=name+'_ae_final', num_of_samples=M)
    
    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    return (extended_train_set_images, extended_train_set_labels)

def _augmenter_knn_interpolation(train_set_images, train_set_labels, M):
    '''
    Run data set augmentation by interpolating neighbor instances.
    '''
    import tensorflow as tf

    rnd_st = np.random.randint(1,100); name='augmenter_network_' + str(rnd_st)
    network = dcae.autoencoder(name=name,image_set_01=train_set_images,image_set_02=train_set_images,image_set_02_labels=train_set_labels)
    network.optimize( num_of_epochs_phase_01 = 15000, num_of_epochs_phase_02=0, num_of_epochs_phase_03=0, learning_rate_phase_01=0.001, learning_rate_phase_02=0.001,)
    new_images, new_labels = network.feature_space_clustered_augmentation(model_to_restore=name+'_ae_final', num_of_samples=M)
    
    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    return (extended_train_set_images, extended_train_set_labels)

def _augmenter_extrapolation(train_set_images, train_set_labels, M):
    '''
    Run data set augmentation by interpolating neighbor instances.
    '''
    import tensorflow as tf

    rnd_st = np.random.randint(1,100); name='augmenter_network_' + str(rnd_st)
    network = dcae.autoencoder(name=name,image_set_01=train_set_images,image_set_02=train_set_images,image_set_02_labels=train_set_labels)
    network.optimize( num_of_epochs_phase_01 = 15000, num_of_epochs_phase_02=0, num_of_epochs_phase_03=0, learning_rate_phase_01=0.001, learning_rate_phase_02=0.001,)
    new_images, new_labels = network.feature_space_extrapolated_augmentation(model_to_restore=name+'_ae_final', num_of_samples=M)
    
    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    return (extended_train_set_images, extended_train_set_labels)

def _augmenter_acgan(train_set_images, train_set_labels, M):
    '''
    ACGAN data augmentation
    '''
    # size of the latent space
    latent_dim = 100
    
    # create the discriminator
    discriminator = acgan.define_discriminator( in_shape=(32,32,3), n_classes=43 )
    # create the generator
    generator = acgan.define_generator( latent_dim, n_classes=43 )
    
    # create the gan
    gan_model = acgan.define_gan(generator, discriminator)
   
    # train model
    acgan.train(generator, discriminator, gan_model, [train_set_images, train_set_labels], latent_dim, n_epochs=40000 )
    
    # generate new samples
    [new_images, new_labels], _ = acgan.generate_fake_samples(generator, latent_dim=latent_dim, n_samples=M)
    
    # create augmented train set
    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    return (extended_train_set_images, extended_train_set_labels)

def _augmenter_bagan(train_set_images, train_set_labels, M):
    '''
    BAGAN data augmentation
    '''
    encoder = bagan.define_encoder()
    decoder = bagan.define_decoder()

    autoencoder = bagan.define_autoencoder( encoder, decoder )
    autoencoder.fit( x=train_set_images, y=train_set_images, batch_size=50, epochs=300 )
    
    latent_features = encoder.predict( train_set_images )
    
    latent_means, latent_stds = bagan.class_conditional_latent_vector_generator( latent_features=latent_features, labels=train_set_labels )
   
    generator = bagan.define_generator(decoder)
    discriminator = bagan.define_discriminator(encoder,class_num=43)
    
    bagan_model = bagan.define_gan( generator=generator, discriminator=discriminator )
 
    bagan.gan_training(generator=generator, discriminator=discriminator, gan_model=bagan_model, images=train_set_images, labels=train_set_labels,
                 latent_means=latent_means, latent_std=latent_stds, num_of_epochs=8000, num_of_batches=50, latent_dim=4*4*128 )
 
    latent_features, generated_class_labels = bagan.generate_fake_latent_samples( num_of_classes=43, sample_size=M, latent_dim=4*4*128, latent_means=latent_means, latent_std=latent_stds, y_train=train_set_labels, is_balancing=True )
    generated_fake_images = generator.predict( latent_features )
    
    # create augmented train set
    extended_train_set_images = np.vstack( (train_set_images, generated_fake_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, generated_class_labels) )
    
    return (extended_train_set_images, extended_train_set_labels)

def _augmenter_manual(train_set_images, train_set_labels, M):
    '''
    Manual augmentation using elementary image processing operations.
    '''
    from keras.preprocessing.image import ImageDataGenerator

    data_gen = ImageDataGenerator(rotation_range=20,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=0.2,
                                 zoom_range=0.0,vertical_flip=False,
                                 fill_mode='nearest')

    data_iterator = data_gen.flow( train_set_images, train_set_labels, batch_size=M )
    new_images, new_labels = None, None

    for data in data_iterator:
        new_images, new_labels = data
        break

    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    return (extended_train_set_images, extended_train_set_labels)

def _randaugment(train_set_images, train_set_labels, M):
    '''
    Perform the RandAugment data augmentation method on the input data set.
    '''
    
    N = 0
    
    if M is None:
        N = train_set_images.shape[0]
    else:
        N = M
    
    new_images = np.zeros_like(train_set_images, dtype=np.float64)
    new_labels = np.zeros_like(train_set_labels, dtype=np.int64)
    
    for i in range(N):
        current_image = (train_set_images[i] * 255).astype(np.uint8)
        distorted_image_tensor = randaugment.distort_image_with_randaugment( image=current_image,
                                                                          num_layers = 2,
                                                                          magnitude=2 )

        distorted_image = K.eval( distorted_image_tensor ).astype('float32')/255.
      
        new_images[i] = distorted_image
        new_labels[i] = train_set_labels[i]
      
    extended_train_set_images = np.vstack( (train_set_images, new_images) )
    extended_train_set_labels = np.hstack( (train_set_labels, new_labels) )

    return (extended_train_set_images, extended_train_set_labels)


def test(M,train_images,train_labels,test_images,test_labels):
    '''
    Test augmenter algorithms against plain data set based classification.
    '''
  
    # data augmentation
    with Pool(1) as p:
        extended_train_set_images, extended_train_set_labels = p.apply(_augmenter_knn_interpolation,(train_images, train_labels, M))
    
    # test on plain CNN classifier
    with Pool(1) as p:
        base_accuracy_plain, augmented_accuracy_plain = p.apply(_test_basic,(train_images, train_labels, extended_train_set_images,
                                                                extended_train_set_labels, test_images, test_labels,False))
    
    # test on RESNET50
    with Pool(1) as p:
        base_accuracy_resnet50, augmented_accuracy_resnet50 = p.apply(_test_basic,(train_images, train_labels, extended_train_set_images,
                                                                extended_train_set_labels, test_images, test_labels,True))


    return base_accuracy_plain, augmented_accuracy_plain, base_accuracy_resnet50, augmented_accuracy_resnet50


def training(name, images, labels, test_images=None, test_labels=None):
    '''
    DEPRECATED at the moment!
    '''
    
    num_of_classes = np.unique(labels).shape[0]

    for _ in range(10):

        if (test_images is None) and (test_labels is None):
            x_train, x_test, y_train, y_test = ms.train_test_split( images, labels, test_size=0.33,
                                                                   shuffle=True, random_state=42)
        else:
            x_train = images
            y_train = labels
            x_test = test_images
            y_test = test_labels

        for i in range(num_of_classes):

            current_class_indices = np.where(y_train==i)[0]
            current_size = current_class_indices.shape[0]

            for subsample_num in [ int(current_size*j) for j in [1., 0.95, 0.85, 0.75, 0.50, 0.25, 0.1] ]:
                
                indices_to_restore = np.random.choice(a=current_size, size=subsample_num,
                                                     replace=False)

                current_class_indices_to_remove = np.delete(current_class_indices, indices_to_restore)

                current_x_train = np.delete( x_train, current_class_indices_to_remove, axis=0 )
                current_y_train = np.delete( y_train, current_class_indices_to_remove, axis=0 )
                
                base_accuracy, augmented_accuracy = test(M=subsample_num,
                                                         train_images=current_x_train,train_labels=current_y_train,
                                                         test_images=x_test,test_labels=y_test)


                with open(name + '_basic_performance_class_' + str(i) + '_sample_' + str(subsample_num) + '.txt', 'a') as f:
                    f.write("%s\n" % base_accuracy)

                with open(name + '_augmented_performance_class_' + str(i)  + '_sample_' + str(subsample_num) + '.txt', 'a') as f:
                    f.write("%s\n" % augmented_accuracy)