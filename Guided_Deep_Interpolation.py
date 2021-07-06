# Copyright 2021 Nokia
# Licensed under the Creative Commons Attribution 
# Non Commercial 4.0 International License
# SPDX-License-Identifier: CC-BY-NC-4.0


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class autoencoder():
'''
DCASE network and Guided Deep Interpolation augmentation implementation. The implemented network is compatible with input image set of dimensionality Nx32x32xC
where N is the number of samples and C is the number of channels.
'''
    
    def __init__(self, name, image_set_01, image_set_02, image_set_02_labels, phase_01_batch_size=500 ,phase_02_batch_size = 500,
                 learning_rate_phase_01=0.001, learning_rate_phase_02 = 0.001,
                 num_of_epochs_phase_01 = 5000, num_of_epochs_phase_02=5000,
                 initializer=tf.contrib.layers.variance_scaling_initializer()):
        
        self._name = name
        
        self._train_set_01 = image_set_01
        self._train_set_02 = image_set_02
        self._train_set_02_labels = image_set_02_labels
        
        self._phase_01_batch_size = phase_01_batch_size
        self._phase_02_batch_size = phase_02_batch_size
        
        self._learning_rate_phase_01 = learning_rate_phase_01
        self._learning_rate_phase_02 = learning_rate_phase_02
        self._num_of_epochs_phase_01 = num_of_epochs_phase_01
        self._num_of_epochs_phase_02 = num_of_epochs_phase_02
        
        self._sample_size_phase_01 = self._train_set_01.shape[0]
        self._sample_size_phase_02 = self._train_set_02.shape[0]
        
        self._sample_width = self._train_set_01.shape[1]
        self._sample_height = self._train_set_01.shape[2]
        self._input_channel_num = self._train_set_01.shape[3]
        
        self._X = tf.placeholder(dtype=tf.float32, shape=[None,self._sample_width,self._sample_height,self._input_channel_num])
        self._Z_01 = tf.placeholder(dtype=tf.float32, shape=[None,self._sample_width,self._sample_height,self._input_channel_num])
        self._Z_02 = tf.placeholder(dtype=tf.float32, shape=[None,self._sample_width,self._sample_height,self._input_channel_num])
        
        self._fc_hidden_01_input_len = 120*4*4 
        self._fully_connected_input_len_disc = 120*4*4
        self._fully_connected_input_len_en_02 = 120*4*4
        
        self.encoder_weights = {
            'conv_weights_01': tf.get_variable(name='encoder_conv_weights_01',shape=[3,3,self._input_channel_num,30],initializer=initializer),
            'conv_weights_02': tf.get_variable(name='encoder_conv_weights_02', shape=[3,3,30,30], initializer=initializer),
            'conv_weights_03': tf.get_variable(name='encoder_conv_weights_03', shape=[3,3,30,30], initializer=initializer),
            
            'conv_weights_04': tf.get_variable(name='encoder_conv_weights_04',shape=[3,3,30,60],initializer=initializer),
            'conv_weights_05': tf.get_variable(name='encoder_conv_weights_05',shape=[3,3,60,60],initializer=initializer),
            'conv_weights_06': tf.get_variable(name='encoder_conv_weights_06',shape=[3,3,60,60],initializer=initializer),
            
            'conv_weights_07': tf.get_variable(name='encoder_conv_weights_07',shape=[3,3,60,120],initializer=initializer),
            'conv_weights_08': tf.get_variable(name='encoder_conv_weights_08',shape=[3,3,120,120],initializer=initializer),
            'conv_weights_09': tf.get_variable(name='encoder_conv_weights_09',shape=[3,3,120,120],initializer=initializer),
            
            'conv_bias_01': tf.get_variable(name='conv_bias_01', shape=[30], initializer=initializer),
            'conv_bias_02': tf.get_variable(name='conv_bias_02', shape=[30], initializer=initializer),
            'conv_bias_03': tf.get_variable(name='conv_bias_03', shape=[30], initializer=initializer),
            
            'conv_bias_04': tf.get_variable(name='conv_bias_04', shape=[60], initializer=initializer),
            'conv_bias_05': tf.get_variable(name='conv_bias_05', shape=[60], initializer=initializer),
            'conv_bias_06': tf.get_variable(name='conv_bias_06', shape=[60], initializer=initializer),
            
            'conv_bias_07': tf.get_variable(name='conv_bias_07', shape=[120], initializer=initializer),
            'conv_bias_08': tf.get_variable(name='conv_bias_08', shape=[120], initializer=initializer),
            'conv_bias_09': tf.get_variable(name='conv_bias_09', shape=[120], initializer=initializer),
            
            }
        
        self.decoder_weights = {
            
            'conv_weights_09': tf.get_variable(name='decoder_conv_weights_09',shape=[3,3,self._input_channel_num,30],initializer=initializer),
            'conv_weights_08': tf.get_variable(name='decoder_conv_weights_08', shape=[3,3,30,30], initializer=initializer),
            'conv_weights_07': tf.get_variable(name='decoder_conv_weights_07', shape=[3,3,30,30], initializer=initializer),
            
            'conv_weights_06': tf.get_variable(name='decoder_conv_weights_06',shape=[3,3,30,60],initializer=initializer),
            'conv_weights_05': tf.get_variable(name='decoder_conv_weights_05',shape=[3,3,60,60],initializer=initializer),
            'conv_weights_04': tf.get_variable(name='decoder_conv_weights_04',shape=[3,3,60,60],initializer=initializer),
            
            'conv_weights_03': tf.get_variable(name='decoder_conv_weights_03',shape=[3,3,60,120],initializer=initializer),
            'conv_weights_02': tf.get_variable(name='decoder_conv_weights_02',shape=[3,3,120,120],initializer=initializer),
            'conv_weights_01': tf.get_variable(name='decoder_conv_weights_01',shape=[3,3,120,120],initializer=initializer),
            
            'conv_bias_09': tf.get_variable(name='deconv_bias_09', shape=[self._input_channel_num], initializer=initializer),
            'conv_bias_08': tf.get_variable(name='deconv_bias_08', shape=[30], initializer=initializer),
            'conv_bias_07': tf.get_variable(name='deconv_bias_07', shape=[30], initializer=initializer),
            
            'conv_bias_06': tf.get_variable(name='deconv_bias_06', shape=[30], initializer=initializer),
            'conv_bias_05': tf.get_variable(name='deconv_bias_05', shape=[60], initializer=initializer),
            'conv_bias_04': tf.get_variable(name='deconv_bias_04', shape=[60], initializer=initializer),
            
            'conv_bias_03': tf.get_variable(name='deconv_bias_03', shape=[60], initializer=initializer),
            'conv_bias_02': tf.get_variable(name='deconv_bias_02', shape=[120], initializer=initializer),
            'conv_bias_01': tf.get_variable(name='deconv_bias_01', shape=[120], initializer=initializer),
            
            }
        
        
        self.self_expressive_weights = {
            'fc_weights_01': tf.get_variable(name='self_expressive_weights',shape=[self._sample_size_phase_02,self._sample_size_phase_02])
            }
            
    def encoder(self, input_batch):
        '''
        Convolutional encoder part to learn the latent representation.
        '''
            
        conv_hidden_01 = tf.nn.conv2d(input_batch, self.encoder_weights['conv_weights_01'], strides=[1,2,2,1],padding='SAME')
        conv_hidden_01 = tf.nn.relu( tf.nn.bias_add(conv_hidden_01,self.encoder_weights['conv_bias_01']) )
        conv_hidden_02 = tf.nn.conv2d(conv_hidden_01, self.encoder_weights['conv_weights_02'], strides=[1,1,1,1],padding='SAME')
        conv_hidden_02 = tf.nn.relu( tf.nn.bias_add(conv_hidden_02,self.encoder_weights['conv_bias_02']) )
        conv_hidden_03 = tf.nn.conv2d(conv_hidden_02, self.encoder_weights['conv_weights_03'], strides=[1,1,1,1],padding='SAME')
        conv_hidden_03 = tf.nn.relu( tf.nn.bias_add(conv_hidden_03,self.encoder_weights['conv_bias_03']) )
        
        conv_hidden_04 = tf.nn.conv2d(conv_hidden_03, self.encoder_weights['conv_weights_04'], strides=[1,2,2,1],padding='SAME')
        conv_hidden_04 = tf.nn.relu( tf.nn.bias_add(conv_hidden_04,self.encoder_weights['conv_bias_04']) )
        conv_hidden_05 = tf.nn.conv2d(conv_hidden_04, self.encoder_weights['conv_weights_05'], strides=[1,1,1,1],padding='SAME')
        conv_hidden_05 = tf.nn.relu( tf.nn.bias_add(conv_hidden_05,self.encoder_weights['conv_bias_05']) )
        conv_hidden_06 = tf.nn.conv2d(conv_hidden_05, self.encoder_weights['conv_weights_06'], strides=[1,1,1,1],padding='SAME')
        conv_hidden_06 = tf.nn.relu( tf.nn.bias_add(conv_hidden_06,self.encoder_weights['conv_bias_06']) )
        
        conv_hidden_07 = tf.nn.conv2d(conv_hidden_06, self.encoder_weights['conv_weights_07'], strides=[1,2,2,1],padding='SAME')
            
        return conv_hidden_07
        
        
    def decoder(self, input_batch):
        '''
        Convolutional decoder. 
        The input is assumed to be of shape (batch_size X width X heght X channels)
        '''
        
        batch_num = tf.shape(input_batch)
        batch_num = batch_num[0]
        
        conv_hidden_03 = tf.nn.conv2d_transpose(input_batch, self.decoder_weights['conv_weights_03'], output_shape=[batch_num,8,8,60],strides=[1,2,2,1], padding='SAME')
        conv_hidden_03 = tf.nn.relu( tf.nn.bias_add(conv_hidden_03,self.decoder_weights['conv_bias_03']) )
        
        conv_hidden_04 = tf.nn.conv2d_transpose(conv_hidden_03, self.decoder_weights['conv_weights_04'], output_shape=[batch_num,8,8,60],strides=[1,1,1,1], padding='SAME')
        conv_hidden_04 = tf.nn.sigmoid( tf.nn.bias_add(conv_hidden_04,self.decoder_weights['conv_bias_04']) )
        conv_hidden_05 = tf.nn.conv2d_transpose(conv_hidden_04, self.decoder_weights['conv_weights_05'], output_shape=[batch_num,8,8,60],strides=[1,1,1,1], padding='SAME')
        conv_hidden_05 = tf.nn.sigmoid( tf.nn.bias_add(conv_hidden_05,self.decoder_weights['conv_bias_05']) )
        conv_hidden_06 = tf.nn.conv2d_transpose(conv_hidden_05, self.decoder_weights['conv_weights_06'], output_shape=[batch_num,16,16,30],strides=[1,2,2,1], padding='SAME')
        conv_hidden_06 = tf.nn.sigmoid( tf.nn.bias_add(conv_hidden_06,self.decoder_weights['conv_bias_06']) )
        
        conv_hidden_07 = tf.nn.conv2d_transpose(conv_hidden_06, self.decoder_weights['conv_weights_07'], output_shape=[batch_num,16,16,30],strides=[1,1,1,1], padding='SAME')
        conv_hidden_07 = tf.nn.sigmoid( tf.nn.bias_add(conv_hidden_07,self.decoder_weights['conv_bias_07']) )
        conv_hidden_08 = tf.nn.conv2d_transpose(conv_hidden_07, self.decoder_weights['conv_weights_08'], output_shape=[batch_num,16,16,30],strides=[1,1,1,1], padding='SAME')
        conv_hidden_08 = tf.nn.sigmoid( tf.nn.bias_add(conv_hidden_08,self.decoder_weights['conv_bias_08']) )
        conv_hidden_09 = tf.nn.conv2d_transpose(conv_hidden_08, self.decoder_weights['conv_weights_09'], output_shape=[batch_num,self._sample_width,self._sample_height,self._input_channel_num],strides=[1,2,2,1], padding='SAME')
        conv_hidden_09 = tf.nn.sigmoid( tf.nn.bias_add(conv_hidden_09,self.decoder_weights['conv_bias_09']) )
         
        return conv_hidden_09
        
    def autoencoder(self, input_batch):
        '''
        Autoencoder composed of the encoder and decoder.
        Note that this implementation does not contain the self-expressive layer.
        '''
        return self.decoder( self.encoder( input_batch ) )
    
    def self_expressive_layer(self, input_batch):
        '''
        Self-expressive layer implementation: a fully connected, single-layer network without any non-linearities.
        '''
        
        off_diagonal_se_weights = self.self_expressive_weights['fc_weights_01'] - tf.diag( tf.diag_part(self.self_expressive_weights['fc_weights_01']) )
        
        latent_self_expressive_approximation = tf.matmul( tf.transpose(tf.reshape(input_batch,[-1,self._fc_hidden_01_input_len])), off_diagonal_se_weights )
        return tf.reshape( tf.transpose( latent_self_expressive_approximation ),[-1,4,4,120] )

    
    def autoencoder_loss(self, input_batch):
	'''
	LSQ loss for the identity mapping of the autoencoder.
	'''        
	return tf.reduce_mean( tf.square( input_batch - self.autoencoder(input_batch) ) )
    
    def self_expressive_layer_loss(self, input_batch, lambda_reg=5.):
	'''
	Loss function for the self-expressive layer. Latent representations of data points are linearly approximated by linear combinations of other data points.
	'''
        loss = tf.reduce_mean( tf.square(  input_batch - self.self_expressive_layer(input_batch) ) ) +  lambda_reg*tf.reduce_mean( tf.abs( self.self_expressive_weights['fc_weights_01'] ) )
        return loss
    
    
    def optimize(self, model_to_restore=None, learning_rate_phase_01=None, learning_rate_phase_02 = None,
                 num_of_epochs_phase_01 = None, num_of_epochs_phase_02=None,
                 phase_01_batch_size = None, phase_02_batch_size=None):
        '''
        Optimization procedure composed of two different parts:
        1st (initialization) phase is to train a simple convolutional autoencoder
        2nd phase is to train the autoencoder equipped with the self-expressive layer.
        Note that the weights of phase-2 are initialized by the respective weights of the 1st phase.
        '''
        
        if learning_rate_phase_01 is not None:
            self._learning_rate_phase_01 = learning_rate_phase_01
            
        if learning_rate_phase_02 is not None:
            self._learning_rate_phase_02 = learning_rate_phase_02
            
        if num_of_epochs_phase_01 is not None:
            self._num_of_epochs_phase_01 = num_of_epochs_phase_01
            
        if num_of_epochs_phase_02 is not None:
            self._num_of_epochs_phase_02 = num_of_epochs_phase_02
            
        if phase_01_batch_size is not None:
            self._phase_01_batch_size = phase_01_batch_size
            
        if phase_02_batch_size is not None:
            self._phase_02_batch_size = phase_02_batch_size 
        
        encoder_op = self.encoder( self._X )
        
        autoencoder_loss = self.autoencoder_loss(self._X)
        self_expressive_loss = self.self_expressive_layer_loss( encoder_op )
        
        optimization_phase_01 = tf.train.AdamOptimizer( learning_rate = self._learning_rate_phase_01 )
        optimization_phase_02 = tf.train.AdamOptimizer( learning_rate = self._learning_rate_phase_02 )
        
        var_list_phase_01 = [ self.encoder_weights[key] for key in self.encoder_weights ] + [ self.decoder_weights[key] for key in self.decoder_weights ]
        var_list_phase_02 = [ self.self_expressive_weights[key] for key in self.self_expressive_weights ]
        
        training_op_phase_01 = optimization_phase_01.minimize( loss = autoencoder_loss, var_list=var_list_phase_01 )
        training_op_phase_02 = optimization_phase_02.minimize( loss = autoencoder_loss + self_expressive_loss, var_list=var_list_phase_01+var_list_phase_02 )
        
        saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run( init )
            
            if model_to_restore is not None:
                self.restore_weights(session=sess, model_to_restore=model_to_restore)
            
            
            # 1st training phase for autoencoder initialization
            for epoch in range(self._num_of_epochs_phase_01):
                sess.run( training_op_phase_01, feed_dict = {self._X: self._train_set_01} )
                
                if epoch % 10 == 0:
                    current_loss = sess.run( autoencoder_loss , feed_dict={self._X:self._train_set_01} )
                    print('epoch: ', epoch, ' ,' ,'loss: ', current_loss)  
                    saver.save(sess, self._name + '_ae_temp')
            
            saver.save(sess, self._name + '_ae_final')
            
            # 2nd training phase for complete learning procedure (autoencoder_loss + self_expressive_loss)
            for epoch in range(self._num_of_epochs_phase_02):
                
                sess.run( training_op_phase_02, feed_dict={self._X: self._train_set_02} )
                
                if epoch % 10 == 0:
                    current_loss = sess.run( self_expressive_loss+autoencoder_loss, feed_dict={self._X: self._train_set_02} )
                    print('epoch: ', epoch, ' ,' ,'loss: ', current_loss)   
                    saver.save(sess, self._name + '_ae_se_temp')
            
            saver.save(sess, self._name + '_ae_se_final')
            
            
    def restore_weights(self, session, model_to_restore):
	'''
	Supplementary function for restoring weights between different phases of the training procedure.
	'''
        saver = tf.train.Saver()
        saver.restore(session, model_to_restore)

                
    def augmentation(self, model_to_restore, num_of_samples):
	'''
	Augmenter function for generating synthetic samples by the proposed Guided Deep Interpolation (GDI) algorithm.
	@model_to_restore: the (tensorflow) model of the traines DCASE network
	'''
        
        init = tf.global_variables_initializer()
        
        latent_input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,4,4,120])
        
        encoder_op = self.encoder(input_batch=self._X)
        decoder_op = self.decoder(input_batch=latent_input_placeholder)
        
        with tf.Session() as sess:
            
            sess.run( init )
            self.restore_weights(session=sess, model_to_restore=model_to_restore)
            
            latent_images = sess.run(tf.reshape(encoder_op,[-1,4*4*120]), feed_dict={self._X:self._train_set_01})
            latent_generated_images = np.zeros( shape=(num_of_samples, 120*4*4) )
            generated_image_labels = np.zeros( shape=(num_of_samples,1) )
            
            adjacency_matrix = sess.run( self.self_expressive_weights['fc_weights_01'] )
            adjacency_matrix = adjacency_matrix - np.diag( np.diag(adjacency_matrix) )
            adjacency_matrix = np.abs( adjacency_matrix )
            adjacency_matrix = 0.5*(adjacency_matrix + adjacency_matrix.T)
            adjacency_matrix = np.divide(adjacency_matrix,np.max(adjacency_matrix,axis=0))
            
            weighted_degrees = [ np.sum(adjacency_matrix[:,i])/self._sample_size_phase_01 for i in range( self._sample_size_phase_01 ) ]
            inverted_weighted_degrees = [ (1/degree) for degree in weighted_degrees ]
            sum_inv_weighted_degree = np.sum( inverted_weighted_degrees )
            
            prob_01 = [ degree/sum_inv_weighted_degree for degree in inverted_weighted_degrees ]
            
            sample_indices = np.random.choice( self._sample_size_phase_01, num_of_samples, p=prob_01 )
            
            for indx, i in np.ndenumerate(sample_indices):
                
                sum_weighted_degree = np.sum( [ degree*adjacency_matrix[ind[0],i] for ind, degree in np.ndenumerate(np.array(weighted_degrees)) ] )
                prob_02 = [ ( degree*adjacency_matrix[ind[0],i] )/sum_weighted_degree for ind, degree in np.ndenumerate(np.array(weighted_degrees)) ]
                
                sample_neighbor = np.random.choice( self._sample_size_phase_01, 1, p=prob_02 )[0]
               
                u = np.random.uniform(low=0.0,high=0.25)
                
                node = latent_images[i] + 0.001*np.random.randn(120*4*4)
                node_neighbor = latent_images[sample_neighbor] + 0.001*np.random.randn(120*4*4)
                
                interpolated_latent_representation = (1-u)*node + u*node_neighbor
                
                latent_generated_images[indx[0]] = interpolated_latent_representation
                generated_image_labels[indx[0]] = self._train_set_02_labels[i]
                
            generated_images = sess.run( decoder_op, feed_dict={latent_input_placeholder:np.reshape(latent_generated_images,(num_of_samples,4,4,120))} )
            
            return generated_images, generated_image_labels[:,0]
