#!/usr/bin/env python3
"""
Expert System Runner for Medical Report Classification

This script processes medical reports using the expert system classification.
"""

import pandas as pd
import os
import sys
from utils import ZJYY

def process_yhy5_data():
    """Process YHY5_result_label.csv data"""
    
    # File paths
    data_dir = "data/ZJYY"
    report_path = "YHY5_result_label.csv"
    label_path = os.path.join(data_dir, "tag.xlsx")
    result_path = "YHY5_result_classified.csv"
    
    # Check if files exist
    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found!")
        return False
        
    if not os.path.exists(label_path):
        print(f"Error: {label_path} not found!")
        return False
    
    print("Starting expert system classification...")
    print(f"Input file: {report_path}")
    print(f"Label file: {label_path}")
    print(f"Output file: {result_path}")
    
    try:
        # Initialize ZJYY classifier
        classifier = ZJYY(
            report_path=report_path,
            label_path=label_path,
            result_path=result_path,
            diagnosis_col='REPORTSCONCLUSION',  # 诊断结论列
            description_col='REPORTSEVIDENCES'  # 征象描述列
        )
        
        # Run classification
        classifier.classify()
        
        print(f"Classification completed! Results saved to: {result_path}")
        return True
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return False

def main():
    """Main function"""
    print("=== Expert System for Medical Report Classification ===")
    print()
    
    # Process YHY5 data
    success = process_yhy5_data()
    
    if success:
        print("\n✅ Expert system processing completed successfully!")
    else:
        print("\n❌ Expert system processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
