"""
in this file loading Telco Customer Churn Dataset

Contains customer demographics, account details, subscribed services, billing info,
and a churn label (Yes/No). Commonly used for classification tasks to predict
customer churn in telecom industries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
import random
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
from visualization import visualization
from sklearn.model_selection import train_test_split

class customer_rentention_prediction_system:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            # debug | info | critical | warning | error

            random.seed(42) #set seed fix the data constantly
            partners = ['jio', 'airtel', 'bsnl', 'vodafone']  #adding telecom_partner to the data
            random_values = [random.choice(partners) for _ in range(len(self.df))]
            self.df.insert(0, 'telecom_partner', random_values) # insert at index 0 (first column)


            logger.info(f'Data loaded succesfully')
            logger.info(f'total rows in the data : {self.df.shape[0]}')
            logger.info(f'total columns in the data : {self.df.shape[1]}')
            logger.info(f' {self.df.isnull().sum()}')
            logger.info(f'Missing Values in Total Charges:{self.df['TotalCharges'].isnull().sum()}')
            self.df.drop('customerID', axis=1, inplace=True)
            logger.info(f'total rows in the data : {self.df.shape[0]}')
            logger.info(f'total columns in the data : {self.df.shape[1]}')
            logger.info(f"Sample data with new column:\n{self.df.head().to_string()}") #sample data

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')


    def visualize(self):
        try:
            viz_logger = setup_logging('visualization')  # Initialize a separate logger for visualization
            logger.info("start visualize")
            visualization(self.df, viz_logger) # This ensures everything inside the function logs to visualization.log
            logger.info("visualize completed successfully")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

if __name__ == "__main__":
    try:
        obj = customer_rentention_prediction_system('D:\\Customer_Retention_Prediction_System\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.visualize()

    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

