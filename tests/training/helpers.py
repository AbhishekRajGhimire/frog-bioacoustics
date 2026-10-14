from __future__ import annotations


TRAINING_CONFIG = b'''schema_version = 1
seed = 7

[model]
pretrained = false

[optimization]
batch_size = 4
max_epochs = 2
learning_rate = 0.001
weight_decay = 0.0
early_stopping_patience = 1

[augmentation]
time_mask_max_frames = 4
frequency_mask_max_bins = 4
max_attenuation_db = 3.0

[baseline]
image_size = 8
inverse_regularization = 1.0
'''
