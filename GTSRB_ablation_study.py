# Copyright 2021 Nokia
# Licensed under the Creative Commons Attribution 
# Non Commercial 4.0 International License
# SPDX-License-Identifier: CC-BY-NC-4.0


import numpy as np
import test

name = 'GTSRB'

train_images = np.load('GTSRB_training_images.npz')['arr_0']
train_labels = np.load('GTSRB_training_labels.npz')['arr_0']

test_images = np.load('GTSRB_test_images.npz')['arr_0']
test_labels = np.load('GTSRB_test_labels.npz')['arr_0']

for _ in range(3):

    for N in [2000, 4000]:
        
        train_images_current, train_labels_current = test.train_data_format(train_images,train_labels,N)
        
        base_accuracy_plain, augmented_accuracy_plain, base_accuracy_resnet50, augmented_accuracy_resnet50 = test.test( N, train_images_current, train_labels_current, test_images, test_labels )
            
        with open(name + '_plain_cnn_aug_sample_' + str(N) + '.txt', 'a') as f:
            f.write("%s\n" % augmented_accuracy_plain)
    
        with open(name + '_resnet50_aug_sample_' + str(N) + '.txt', 'a') as f:
            f.write("%s\n" % augmented_accuracy_resnet50)
            
