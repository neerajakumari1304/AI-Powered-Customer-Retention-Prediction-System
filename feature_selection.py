import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('feature_selection')

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

class FEATURE_SELECTION:

    def drop_low_variance(X_train, X_test,y=None, threshold=0.01):
        try:
            # FIX: Only use numeric columns for math
            X_train = X_train.select_dtypes(include=[np.number])
            X_test = X_test.select_dtypes(include=[np.number])

            selector = VarianceThreshold(threshold=threshold).fit(X_train)
            cols = X_train.columns[selector.get_support()]
            return X_train[cols], X_test[cols]
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def drop_redundant_features(X_train, X_test,y=None, threshold=0.9):
        try:
            X_tr = X_train.select_dtypes(include=[np.number])

            corr_matrix = X_tr.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            return X_train.drop(columns=to_drop), X_test.drop(columns=to_drop)
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def select_top_features(X_train, X_test, y, k=10):
        try:
            X_tr = X_train.select_dtypes(include=[np.number])
            X_te = X_test.select_dtypes(include=[np.number])

            k = min(k, X_tr.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k).fit(X_tr, y)
            cols = X_tr.columns[selector.get_support()]
            return X_train[cols], X_test[cols]
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def select_by_importance(X_train, X_test, y):
        try:
            X_tr = X_train.select_dtypes(include=[np.number])

            model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y)
            importance = model.feature_importances_
            cols = X_tr.columns[importance >= np.mean(importance)]
            return X_train[cols], X_test[cols]
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')



