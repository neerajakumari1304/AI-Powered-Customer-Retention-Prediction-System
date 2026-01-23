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
logger = setup_logging('data_balance')

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

class DATA_BALANCE:
    def smote(X_train, y_train):
        try:
            logger.info(f"Applying SMOTE. Before: {y_train.value_counts().to_dict()}")
            if X_train.select_dtypes(include=['object']).columns.any():
                return X_train, y_train
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

            logger.info(f"SMOTE Complete. After: {y_res.value_counts().to_dict()}")
            return X_res, y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def random_over_sampling(X_train, y_train):
        try:
            logger.info(f"Random Over Sampling started. Before: {y_train.value_counts().to_dict()}")

            ros = RandomOverSampler(sampling_strategy='auto', random_state=42)

            X_res, y_res = ros.fit_resample(X_train, y_train)

            logger.info(f"Random Over Sampling complete. After: {y_res.value_counts().to_dict()}")

            return X_res, y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def random_under_sampling(X_train, y_train):
        try:
            rus = RandomUnderSampler(random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)
            return X_res, y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def hybrid_balancing(X_train, y_train):
        try:
            if X_train.select_dtypes(include=['object']).shape[1] > 0:
                logger.warning("Strings detected! Skipping hybrid_balancing.")
                return X_train, y_train
            smt = SMOTETomek(random_state=42)
            X_res, y_res = smt.fit_resample(X_train, y_train)
            return X_res, y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return X_train, y_train

    def adasyn(X_train, y_train):
        try:
            if X_train.select_dtypes(include=['object']).shape[1] > 0:
                logger.warning("Strings detected! Skipping ADASYN.")
                return X_train, y_train
            logger.info(f"ADASYN Balancing. Original: {y_train.value_counts().to_dict()}")

            adasyn = ADASYN(random_state=42)
            X_res, y_res = adasyn.fit_resample(X_train, y_train)

            logger.info(f"ADASYN Complete. New count: {y_res.value_counts().to_dict()}")
            return X_res, y_res

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def smote_enn(X_train, y_train):
        try:
            if X_train.select_dtypes(include=['object']).shape[1] > 0:
                logger.warning("Strings detected in data. SMOTE-ENN requires numeric input.")
                return X_train, y_train
            logger.info(" SMOTE-ENN (Hybrid) Balancing")

            sme = SMOTEENN(random_state=42)
            X_res, y_res = sme.fit_resample(X_train, y_train)

            return X_res, y_res
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

