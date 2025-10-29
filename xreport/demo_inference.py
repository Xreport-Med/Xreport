#!/usr/bin/env python3
"""
Demo script for Xreport zero-shot inference
Simple example showing how to use the XrayInference class
"""

import os
import json
from zero_shot_inference import XrayInference

def demo_inference():
    """Demo function showing basic usage"""
    
    # Configuration
    config_path = 'configs/Res_train_test.yaml'
    checkpoint_path = 'path/to/your/checkpoint.pth'  # Update this path
    bert_model_name = 'bert-base-uncased'
    image_encoder_name = 'resnet'  # or 'densenet', 'vit'
    
    # Initialize the inference class
    print("Initializing Xreport inference...")
    try:
        inferencer = XrayInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            bert_model_name=bert_model_name,
            image_encoder_name=image_encoder_name
        )
        print("✓ Models loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    # Example 1: Basic prediction
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Prediction")
    print("="*50)
    
    image_path = "path/to/your/xray_image.jpg"  # Update this path
    
    if os.path.exists(image_path):
        results = inferencer.predict(
            image_path=image_path,
            dataset_type='chestxray14',  # or 'chexpert', 'general'
            threshold=0.3,
            top_k=5
        )
        
        print(f"Image: {image_path}")
        print(f"Top {len(results['predictions'])} predictions:")
        for i, pred in enumerate(results['predictions'], 1):
            print(f"  {i}. {pred['disease']}: {pred['probability']:.3f} ({pred['confidence']})")
    else:
        print(f"Image not found: {image_path}")
    
    # Example 2: Generate medical report
    print("\n" + "="*50)
    print("EXAMPLE 2: Medical Report Generation")
    print("="*50)
    
    if os.path.exists(image_path):
        results = inferencer.predict_with_report(
            image_path=image_path,
            dataset_type='general',
            threshold=0.2
        )
        
        print(f"Image: {image_path}")
        print(f"\nGenerated Report:")
        print(f"  {results['report']}")
        
        print(f"\nDetailed findings:")
        for pred in results['predictions']:
            print(f"  - {pred['disease']}: {pred['probability']:.3f} ({pred['confidence']})")
    else:
        print(f"Image not found: {image_path}")
    
    # Example 3: Batch processing
    print("\n" + "="*50)
    print("EXAMPLE 3: Batch Processing")
    print("="*50)
    
    image_folder = "path/to/xray_images/"  # Update this path
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            print(f"Found {len(image_files)} images in {image_folder}")
            
            batch_results = []
            for img_file in image_files[:3]:  # Process first 3 images
                img_path = os.path.join(image_folder, img_file)
                try:
                    results = inferencer.predict_with_report(
                        image_path=img_path,
                        dataset_type='chestxray14',
                        threshold=0.3
                    )
                    batch_results.append({
                        'image': img_file,
                        'report': results['report'],
                        'top_findings': results['predictions'][:3]
                    })
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            
            # Save batch results
            with open('batch_results.json', 'w') as f:
                json.dump(batch_results, f, indent=2)
            print("Batch results saved to batch_results.json")
        else:
            print(f"No image files found in {image_folder}")
    else:
        print(f"Image folder not found: {image_folder}")

def quick_test():
    """Quick test with minimal setup"""
    print("Quick test mode - using random weights")
    
    # This will work even without a real checkpoint
    try:
        inferencer = XrayInference(
            config_path='configs/Res_train_test.yaml',
            checkpoint_path='dummy_checkpoint.pth',  # This won't exist
            bert_model_name='bert-base-uncased',
            image_encoder_name='resnet'
        )
        print("✓ Inference class initialized (with random weights)")
        print("Note: For real predictions, you need a trained checkpoint")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == '__main__':
    print("Xreport Zero-shot Inference Demo")
    print("="*40)
    
    # Check if we have a real checkpoint
    checkpoint_path = 'path/to/your/checkpoint.pth'  # Update this
    if os.path.exists(checkpoint_path):
        demo_inference()
    else:
        print("No checkpoint found. Running quick test...")
        quick_test()
        
        print("\nTo run the full demo:")
        print("1. Download a trained Xreport checkpoint")
        print("2. Update the checkpoint_path in this script")
        print("3. Provide a path to an X-ray image")
        print("4. Run: python demo_inference.py")
