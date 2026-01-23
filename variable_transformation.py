import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from scipy.stats import boxcox

from sklearn.impute import SimpleImputer
from log_code import setup_logging
logger = setup_logging('variable_transformation')

class SCALER:
    def standard_scaling(X_train, X_test):
        sc = StandardScaler()
        return pd.DataFrame(sc.fit_transform(X_train), X_train.index, X_train.columns), \
               pd.DataFrame(sc.transform(X_test), X_test.index, X_test.columns)


    def minmax_scaling(X_train, X_test):
        sc = MinMaxScaler()
        return pd.DataFrame(sc.fit_transform(X_train), X_train.index, X_train.columns), \
               pd.DataFrame(sc.transform(X_test), X_test.index, X_test.columns)


    def robust_scaling(X_train, X_test):
        sc = RobustScaler()
        return pd.DataFrame(sc.fit_transform(X_train), X_train.index, X_train.columns), \
               pd.DataFrame(sc.transform(X_test), X_test.index, X_test.columns)


    def log_transform(X_train, X_test):
        return X_train.apply(lambda col: np.log1p(col) if (col >= 0).all() else col), \
               X_test.apply(lambda col: np.log1p(col) if (col >= 0).all() else col)


    def power_transform(X_train, X_test):
        pt = PowerTransformer()
        return pd.DataFrame(pt.fit_transform(X_train), X_train.index, X_train.columns), \
               pd.DataFrame(pt.transform(X_test), X_test.index, X_test.columns)


    def boxcox_transform(X_train, X_test):
        X_tr, X_te = X_train.copy(), X_test.copy()
        for col in X_tr.columns:
            if (X_tr[col] > 0).all():
                X_tr[col], lam = boxcox(X_tr[col])
                X_te[col] = boxcox(X_te[col], lmbda=lam)
        return X_tr, X_te


    def yeojohnson_transform(X_train, X_test):
        pt = PowerTransformer(method='yeo-johnson')
        return pd.DataFrame(pt.fit_transform(X_train), X_train.index, X_train.columns), \
               pd.DataFrame(pt.transform(X_test), X_test.index, X_test.columns)


    def quantile_transform(X_train, X_test):
        qt = QuantileTransformer(output_distribution='normal')
        return pd.DataFrame(qt.fit_transform(X_train), X_train.index, X_train.columns), \
               pd.DataFrame(qt.transform(X_test), X_test.index, X_test.columns)