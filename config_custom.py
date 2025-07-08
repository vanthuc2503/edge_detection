"""
Configuration file for custom TEED training
Modify these settings according to your data and requirements
"""

# Data paths - MODIFY THESE FOR YOUR DATA
DATA_CONFIG = {
    'train_dir': 'ANIMATED/train',  # Your training data directory
    'val_dir': 'ANIMATED/test',     # Your validation/test data directory
    'train_list': 'ANIMATED/train_pair.lst',  # Your training data list file
    'test_list': 'ANIMATED/test_pair.lst',    # Your test data list file
}

# Pretrained weights path - MODIFY THIS
PRETRAINED_WEIGHTS = 'checkpoints/BIPED/5/5_model.pth'  # Path to your pretrained weights

# Training parameters - ADJUST AS NEEDED
TRAINING_CONFIG = {
    'epochs': 10,           # Number of training epochs
    'batch_size': 8,        # Batch size
    'lr': 1e-4,            # Initial learning rate (lower for fine-tuning)
    'lrs': [5e-5],         # Learning rate schedule
    'adjust_lr': [5],      # Epochs when to adjust learning rate
    'wd': 2e-4,            # Weight decay
    'workers': 8,          # Number of data loader workers
}

# Image dimensions - ADJUST BASED ON YOUR DATA
IMAGE_CONFIG = {
    'img_width': 512,      # Training image width
    'img_height': 512,     # Training image height
    'test_img_width': 512, # Test image width
    'test_img_height': 512, # Test image height
}

# Output settings
OUTPUT_CONFIG = {
    'output_dir': 'ANIMATED_RESULTS', 
    'res_dir': 'ANIMATED_RESULTS',         
    'version_notes': 'TEED fine-tuned on custom data from pretrained weights'
}

# GPU settings
GPU_CONFIG = {
    'use_gpu': 0,          # GPU device ID (0 for first GPU)
    'tensorboard': True,   # Enable tensorboard logging
} 