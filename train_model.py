#!/usr/bin/env python3
"""
E-Waste Classification Model Training Script

This script trains a CNN model to classify different types of electronic waste
and provides recycling suggestions for each category.

Usage:
    python train_model.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--patience PATIENCE]

Example:
    python train_model.py --epochs 50 --batch-size 32 --patience 10
"""

import argparse
import os
import sys
from ewaste_predictor import EWastePredictor
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, save_path="training_history.png"):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")

def print_training_summary(history, report, class_names):
    """Print a comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\nFINAL METRICS:")
    print(f"   Training Accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"   Training Loss:       {final_train_loss:.4f}")
    print(f"   Validation Loss:     {final_val_loss:.4f}")
    
    # Best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    print(f"\nBEST PERFORMANCE:")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Achieved at Epoch:        {best_epoch}")
    
    # Test results
    if report:
        print(f"\nTEST RESULTS:")
        print(f"   Test Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
        
        print(f"\nPER-CLASS PERFORMANCE:")
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1_score = report[class_name]['f1-score']
                print(f"   {class_name:15} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
    
    print(f"\nMODEL SAVED:")
    print(f"   File: ewaste_model.h5")
    print(f"   Classes: {len(class_names)}")
    print(f"   Class Names: {', '.join(class_names)}")
    
    print("\nNEXT STEPS:")
    print("   1. Run 'streamlit run app.py' to start the web application")
    print("   2. Upload images to test the trained model")
    print("   3. Check the recycling suggestions for each e-waste type")
    
    print("\n" + "="*60)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train E-Waste Classification Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--dataset', type=str, default='ewaste_dataset', help='Dataset directory path')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224], help='Image size (width height)')
    parser.add_argument('--save-plot', action='store_true', help='Save training history plot')
    
    args = parser.parse_args()
    
    print("Starting E-Waste Classification Model Training")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Image Size: {args.image_size[0]}x{args.image_size[1]}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset directory '{args.dataset}' not found!")
        print("Please ensure the dataset is in the correct location.")
        sys.exit(1)
    
    # Check dataset structure
    train_dir = os.path.join(args.dataset, 'train')
    val_dir = os.path.join(args.dataset, 'val')
    test_dir = os.path.join(args.dataset, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        print("ERROR: Dataset structure is incorrect!")
        print("Expected structure:")
        print("  dataset/")
        print("  ├── train/")
        print("  ├── val/")
        print("  └── test/")
        sys.exit(1)
    
    try:
        # Initialize predictor
        print("\nInitializing E-Waste Predictor...")
        predictor = EWastePredictor(
            dataset_dir=args.dataset,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size
        )
        
        # Create model
        print("Creating model architecture...")
        model = predictor.create_model()
        print(f"Model created with {len(predictor.class_names)} classes")
        print(f"Classes: {', '.join(predictor.class_names)}")
        
        # Train model
        print(f"\nStarting training for {args.epochs} epochs...")
        print("This may take a while. Please be patient...")
        
        history, test_gen = predictor.train_model(epochs=args.epochs, patience=args.patience)
        
        # Evaluate model
        print("\nEvaluating model on test data...")
        report, cm, predictions = predictor.evaluate_model(test_gen)
        
        # Save model
        print("\nSaving trained model...")
        predictor.save_model()
        
        # Plot training history
        if args.save_plot:
            print("\nCreating training history plot...")
            plot_training_history(history)
        
        # Print summary
        print_training_summary(history, report, predictor.class_names)
        
        print("\nTraining completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial model may be saved. You can resume training later.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        print("Please check your dataset and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
