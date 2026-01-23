import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('cat_to_num')

from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

class CAT_TO_NUM:
    def label_encoding(X_train, X_test):
        try:
            logger.debug('Label Encoding started')
            logger.info(f'Before X_train shape: {X_train.shape}')

            X_train = X_train.copy()
            X_test = X_test.copy()
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            if not cat_cols:
                logger.info("No categorical columns found for Label Encoding.")
                return X_train, X_test

            logger.info(f'Encoding columns: {cat_cols}')

            for col in cat_cols:
                le = LabelEncoder()
                le.fit(pd.concat([X_train[col], X_test[col]]).astype(str))

                X_train[col] = le.transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

            logger.debug('After Label Encoding')
            logger.info(f'After X_train (first 5 rows):\n{X_train.head()}')

            return X_train, X_test
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def one_hot_encoding(X_train, X_test):
        try:
            logger.debug('One-Hot Encoding started')
            logger.info(f'Before X_train shape: {X_train.shape}')

            nominal_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            if not nominal_cols:
                logger.info("No categorical columns found for One-Hot Encoding.")
                return X_train, X_test

            logger.info(f'Encoding columns: {nominal_cols}')

            X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

            logger.debug('One-Hot Encoding complete')
            logger.info(f'After X_train shape: {X_train.shape}')

            return X_train, X_test
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def frequency_encoding(X_train, X_test):
        try:
            logger.debug('Frequency Encoding started')
            logger.info(f'Before X_train shape: {X_train.shape}')

            X_train = X_train.copy()
            X_test = X_test.copy()

            # 1. Identify only categorical columns to avoid corrupting numeric data
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            if not cat_cols:
                logger.info("No categorical columns found for Frequency Encoding.")
                return X_train, X_test

            logger.info(f'Encoding columns: {cat_cols}')

            for col in cat_cols:
                # 2. Calculate frequency (value counts) from the training set only
                freq = X_train[col].value_counts()

                # 3. Map frequencies to both sets
                X_train[col] = X_train[col].map(freq)

                # 4. For the test set, fill new/unseen categories with 0
                X_test[col] = X_test[col].map(freq).fillna(0)

            logger.debug('Frequency Encoding complete')
            logger.info(f'After X_train (first 5 rows):\n{X_train.head()}')

            return X_train, X_test

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def binary_encoding(X_train, X_test):
        try:
            logger.debug('Binary Encoding started')
            logger.info(f'Before X_train shape: {X_train.shape}')

            nominal_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            if not nominal_cols:
                logger.info("No categorical columns found for Binary Encoding.")
                return X_train, X_test

            logger.info(f'Columns to be Binary Encoded: {nominal_cols}')
            encoder = ce.BinaryEncoder(cols=nominal_cols)

            X_train = encoder.fit_transform(X_train)
            X_test = encoder.transform(X_test)

            logger.debug('Binary Encoding complete')
            logger.info(f'After X_train shape: {X_train.head()}')

            return X_train, X_test

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def ordinal_encoding(X_train, X_test):
        try:
            logger.info(f'Ordinal Encoding started')
            X_train, X_test = X_train.copy(), X_test.copy()

            # Define specific hierarchical order
            ordinal_cols = {
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'InternetService': ['No', 'DSL', 'Fiber optic'],
                'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                  'Credit card (automatic)']
            }
            for col, order in ordinal_cols.items():
                if col in X_train.columns:
                    mapping = {v: i for i, v in enumerate(order)}
                    X_train[col] = X_train[col].map(mapping)
                    # Use .get() or fillna to handle values not in the mapping
                    X_test[col] = X_test[col].map(mapping).fillna(-1)

            logger.info(f'Ordinal Encoding complete. Shape: {X_train.shape}')
            return X_train, X_test
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')