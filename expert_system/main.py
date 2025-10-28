import pandas as pd
import re

def get_report():
    report_dict = {
       '北医三院':'E:/Desktop/research/AOreal/算法预实验/sentence_cluster/北医三院/报告.xlsx',
        '兰州手足外科医院':'E:/Desktop/research/AOreal/算法预实验/sentence_cluster/兰州手足外科医院/检查级报告.xlsx'
    }
    return report_dict

def main():
    report_dict = get_report()
    return

if __name__ == '__main__':
    main()