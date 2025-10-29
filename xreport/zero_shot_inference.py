#!/usr/bin/env python3
"""
Zero-shot inference script for Xreport model
Given an input X-ray image, outputs corresponding disease predictions
"""

import argparse
import os
import logging
import ruamel_yaml as yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from models.clip_tqn import CLP_clinical, ModelRes, ModelDense, TQN_Model, ModelRes512
from models.vit import ModelViT

class XrayInference:
    def __init__(self, config_path, checkpoint_path, bert_model_name, image_encoder_name='resnet', device=None):
        """
        Initialize XrayInference class
        
        Args:
            config_path: Path to config YAML file
            checkpoint_path: Path to model checkpoint
            bert_model_name: BERT model name for text encoder
            image_encoder_name: Image encoder type ('resnet', 'densenet', 'vit')
            device: Device to run inference on
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        self.image_encoder_name = image_encoder_name
        
        # Load models
        self._load_models(bert_model_name, checkpoint_path)
        
        # Define disease categories for different datasets
        self.disease_categories = {
            'chestxray14': [
                "atelectasis", "cardiomegaly", "pleural effusion", "infiltration", 
                "lung mass", "lung nodule", "pneumonia", "pneumothorax", 
                "consolidation", "edema", "emphysema", "fibrosis", 
                "pleural thicken", "hernia"
            ],
            'chexpert': [
                "atelectasis", "cardiomegaly", "consolidation", "edema", "pleural effusion"
            ],
            'general': [
                'normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 
                'atelectasis', 'tube', 'consolidation', 'enlarged cardiomediastinum',
                'tip', 'pneumonia', 'line', 'cardiomegaly', 'fracture', 'calcification',
                'device', 'engorgement', 'nodule', 'wire', 'pacemaker', 'pleural thicken', 
                'marking', 'scar', 'hyperinflate', 'blunt', 'collapse', 'emphysema', 
                'aerate', 'mass', 'infiltration', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 
                'dilation', 'aspiration'
            ]
        }
        
        # Image preprocessing
        self._setup_image_transform()
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_models(self, bert_model_name, checkpoint_path):
        """Load image encoder, text encoder, and main model"""
        print("Loading models...")
        
        # Load image encoder
        if self.image_encoder_name == 'resnet':
            if self.config['img_res'] == 224:
                self.image_encoder = ModelRes(res_base_model='resnet50').to(self.device)
            else:
                self.image_encoder = ModelRes512(res_base_model='resnet50').to(self.device)
        elif self.image_encoder_name == 'densenet':
            self.image_encoder = ModelDense(dense_base_model='densenet121').to(self.device)
        elif self.image_encoder_name == 'vit':
            self.image_encoder = ModelViT().to(self.device)
        else:
            raise ValueError(f"Unsupported image encoder: {self.image_encoder_name}")
        
        # Load text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, do_lower_case=True, local_files_only=True)
        self.text_encoder = CLP_clinical(bert_model_name=bert_model_name).to(self.device)
        
        # Load main model
        self.model = TQN_Model().to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load image encoder weights
            if 'image_encoder' in checkpoint:
                self.image_encoder.load_state_dict(checkpoint['image_encoder'])
                print("Loaded image encoder weights")
            
            # Load text encoder weights
            if 'text_encoder' in checkpoint:
                self.text_encoder.load_state_dict(checkpoint['text_encoder'])
                print("Loaded text encoder weights")
            
            # Load main model weights
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("Loaded main model weights")
            
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
        
        # Set models to eval mode
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.model.eval()
    
    def _setup_image_transform(self):
        """Setup image preprocessing pipeline"""
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([
            transforms.Resize(self.config['img_res'], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
    
    def _get_text_features(self, text_list, max_length=77):
        """Get text features for given text list"""
        text_tokens = self.tokenizer(
            text_list, 
            add_special_tokens=True, 
            max_length=max_length, 
            padding=True, 
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.text_encoder.encode_text(text_tokens)
        return text_features
    
    def preprocess_image(self, image_path):
        """Preprocess input image"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # Assume it's already a PIL Image
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def predict(self, image_path, dataset_type='general', threshold=0.5, top_k=5):
        """
        Predict diseases from X-ray image
        
        Args:
            image_path: Path to X-ray image or PIL Image object
            dataset_type: Type of dataset ('chestxray14', 'chexpert', 'general')
            threshold: Confidence threshold for predictions
            top_k: Number of top predictions to return
            
        Returns:
            dict: Prediction results with disease names, probabilities, and confidence scores
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Get disease categories
        disease_list = self.disease_categories.get(dataset_type, self.disease_categories['general'])
        
        # Get text features
        text_features = self._get_text_features(disease_list)
        
        # Run inference
        with torch.no_grad():
            # Get image features
            image_features, _ = self.image_encoder(image_tensor)
            
            # Get predictions
            predictions = self.model(image_features, text_features)  # Shape: [1, num_diseases, 2]
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(predictions, dim=-1)  # Shape: [1, num_diseases, 2]
            
            # Get positive class probabilities (disease present)
            disease_probs = probabilities[:, :, 1].squeeze(0)  # Shape: [num_diseases]
        
        # Convert to numpy for easier handling
        disease_probs = disease_probs.cpu().numpy()
        
        # Create results
        results = []
        for i, (disease, prob) in enumerate(zip(disease_list, disease_probs)):
            results.append({
                'disease': disease,
                'probability': float(prob),
                'confidence': 'high' if prob > 0.7 else 'medium' if prob > 0.3 else 'low'
            })
        
        # Filter by threshold and sort by probability
        filtered_results = [r for r in results if r['probability'] >= threshold]
        filtered_results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get top-k results
        top_results = filtered_results[:top_k]
        
        return {
            'predictions': top_results,
            'total_diseases_checked': len(disease_list),
            'diseases_above_threshold': len(filtered_results),
            'dataset_type': dataset_type,
            'threshold': threshold
        }
    
    def predict_with_report(self, image_path, dataset_type='general', threshold=0.5):
        """
        Predict diseases and generate a simple report
        
        Args:
            image_path: Path to X-ray image or PIL Image object
            dataset_type: Type of dataset ('chestxray14', 'chexpert', 'general')
            threshold: Confidence threshold for predictions
            
        Returns:
            dict: Prediction results with generated report
        """
        # Get predictions
        results = self.predict(image_path, dataset_type, threshold, top_k=10)
        
        # Generate report
        predictions = results['predictions']
        
        if not predictions:
            report = "No significant abnormalities detected above the confidence threshold."
        else:
            high_conf = [p for p in predictions if p['confidence'] == 'high']
            medium_conf = [p for p in predictions if p['confidence'] == 'medium']
            
            report_parts = []
            
            if high_conf:
                diseases = [p['disease'] for p in high_conf]
                report_parts.append(f"High confidence findings: {', '.join(diseases)}")
            
            if medium_conf:
                diseases = [p['disease'] for p in medium_conf]
                report_parts.append(f"Moderate confidence findings: {', '.join(diseases)}")
            
            report = ". ".join(report_parts) + "."
        
        results['report'] = report
        return results


def main():
    parser = argparse.ArgumentParser(description='Xreport Zero-shot Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to X-ray image')
    parser.add_argument('--config', type=str, default='configs/Res_train_test.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--image_encoder', type=str, default='resnet', choices=['resnet', 'densenet', 'vit'], help='Image encoder type')
    parser.add_argument('--dataset_type', type=str, default='general', choices=['chestxray14', 'chexpert', 'general'], help='Dataset type for disease categories')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold for predictions')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--output', type=str, help='Path to save results (JSON format)')
    
    args = parser.parse_args()
    
    # Initialize inference
    print("Initializing Xreport inference...")
    inferencer = XrayInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        bert_model_name=args.bert_model,
        image_encoder_name=args.image_encoder
    )
    
    # Run inference
    print(f"Processing image: {args.image_path}")
    results = inferencer.predict_with_report(
        image_path=args.image_path,
        dataset_type=args.dataset_type,
        threshold=args.threshold
    )
    
    # Print results
    print("\n" + "="*50)
    print("XR REPORT INFERENCE RESULTS")
    print("="*50)
    print(f"Image: {args.image_path}")
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Threshold: {args.threshold}")
    print(f"Total diseases checked: {results['total_diseases_checked']}")
    print(f"Diseases above threshold: {results['diseases_above_threshold']}")
    print("\nGenerated Report:")
    print(f"  {results['report']}")
    
    print("\nDetailed Predictions:")
    for i, pred in enumerate(results['predictions'], 1):
        print(f"  {i}. {pred['disease']}: {pred['probability']:.3f} ({pred['confidence']} confidence)")
    
    # Save results if output path specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
