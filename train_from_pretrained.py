import os
import sys
import argparse
from main import parse_args, main

def train_from_pretrained():
    
    is_testing = False

    args, train_info = parse_args(is_testing=is_testing)
    
    args.is_testing = False 
    args.resume = True 
    args.checkpoint_data = '5/5_model.pth'
    
    args.input_dir = 'ANIMATED' 
    args.input_val_dir = 'ANIMATED' 
    args.output_dir = 'ANIMATED_RESULTS'
    args.train_data = 'ANIMATED'
    args.train_list = 'train_pair.lst' 
    args.test_list = 'test_pair.lst'  
    
    args.epochs = 10
    args.batch_size = 4 
    args.lr = 1e-4
    args.lrs = [5e-5]
    args.adjust_lr = [5]

    args.img_width = 1024
    args.img_height = 1024
    args.test_img_width = 1024
    args.test_img_height = 1024
    
    args.output_dir = 'checkpoints/custom_training'
    args.res_dir = 'result/custom_training'
    
    args.version_notes = 'TEED fine-tuned on custom data from pretrained weights'
    args.tensorboard = True
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)
    
    print("============================================================")
    print("TEED Custom Training from Pretrained Weights")
    print("============================================================")
    print(f"Training data: {args.train_data}")
    print(f"Validation data: {args.input_val_dir}")
    print(f"Pretrained weights: {args.checkpoint_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("============================================================")
    
    print("Starting main training function...")
    try:
        main(args, train_info)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    train_from_pretrained() 