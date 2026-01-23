import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from log_code import setup_logging
logger = setup_logging('missing_value')

class IMPUTE_DATA:
    def mean_impute(X_train, X_test):
        try:
            imp = SimpleImputer(strategy='mean')
            cols = X_train.select_dtypes(exclude='object').columns
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[cols] = imp.fit_transform(X_train[cols])
            X_test[cols] = imp.transform(X_test[cols])
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def median_impute(X_train, X_test):
        try:
            imp = SimpleImputer(strategy='median')
            cols = X_train.select_dtypes(exclude='object').columns
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[cols] = imp.fit_transform(X_train[cols])
            X_test[cols] = imp.transform(X_test[cols])
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def mode_impute(X_train, X_test):
        try:
            imp = SimpleImputer(strategy='most_frequent')
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[:] = imp.fit_transform(X_train)
            X_test[:] = imp.transform(X_test)
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def backward_fill(X_train, X_test):
        try:
            X_train = X_train.copy().bfill()
            X_test = X_test.copy().bfill()
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def forward_fill(X_train, X_test):
        try:
            X_train = X_train.copy().ffill()
            X_test = X_test.copy().ffill()
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def random_sample_impute(X_train, X_test):
        try:
            X_train = X_train.copy()
            X_test = X_test.copy()
            for col in X_train.columns:
                if X_train[col].isnull().sum() > 0:
                    random_samples = X_train[col].dropna()
                    if len(random_samples) == 0:
                        continue
                    X_train[col] = X_train[col].apply(lambda x: np.random.choice(random_samples) if pd.isnull(x) else x)
                    X_test[col] = X_test[col].apply(lambda x: np.random.choice(random_samples) if pd.isnull(x) else x)
            return X_train, X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')