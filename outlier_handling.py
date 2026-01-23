import numpy as np
import pandas as pd
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import RobustScaler
from tabulate import tabulate

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('outlier_handling')

class Outlier_handling:
    plot_dir = "plot_outliers"
    os.makedirs(plot_dir, exist_ok=True)

    @staticmethod
    def iqr_method(X_train, X_test):
        try:
            Q1, Q3 = X_train.quantile(0.25), X_train.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            return X_train.clip(lower=lower, upper=upper, axis=1), X_test.clip(lower=lower, upper=upper, axis=1)

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def zscore_method(X_train, X_test):
        try:
            mean, std = X_train.mean(), X_train.std()
            lower, upper = mean - 3 * std, mean + 3 * std
            return X_train.clip(lower=lower, upper=upper, axis=1), X_test.clip(lower=lower, upper=upper, axis=1)
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def winsorization(X_train, X_test):
        try:
            lower, upper = X_train.quantile(0.05), X_train.quantile(0.95)
            return X_train.clip(lower=lower, upper=upper, axis=1), X_test.clip(lower=lower, upper=upper, axis=1)
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def clipping(X_train, X_test):
        try:
            lower, upper = X_train.quantile(0.01), X_train.quantile(0.99)
            return X_train.clip(lower=lower, upper=upper, axis=1), X_test.clip(lower=lower, upper=upper, axis=1)
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def log_outlier(X_train, X_test):
        shift = abs(min(X_train.min().min(), 0)) + 1
        return np.log1p(X_train + shift), np.log1p(X_test + shift)

    def robust_scaling(X_train, X_test):
        try:
            scaler = RobustScaler()
            return pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns), \
                pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def no_outlier(X_train, X_test):
        try:
            return X_train, X_test
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    @staticmethod
    def plot_comparison(original_df, transformed_df, col_name, method_name, logger):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=original_df[col_name])
        plt.title("Before")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=transformed_df[col_name])
        plt.title(f"After ({method_name})")

        file_path = f"{Outlier_handling.plot_dir}/{col_name}_{method_name}.png"
        plt.savefig(file_path)
        plt.close()

        # This is the line that generates your missing output!
        logger.info(f"Visual check saved: {file_path}")

    @staticmethod
    def apply_all_techniques(X_train, X_test, logger):
        # List of available methods in the class
        methods = ['iqr_method', 'zscore_method', 'winsorization', 'clipping', 'log_outlier', 'robust_scaling',
                   'no_outlier']
        results = {}
        summary = []

        # Store a copy of original data so methods don't overwrite each other
        X_train_orig = X_train.copy()
        X_test_orig = X_test.copy()

        for m in methods:
            try:
                # Dynamically call the method
                func = getattr(Outlier_handling, m)
                X_tr_mod, X_te_mod = func(X_train_orig.copy(), X_test_orig.copy())

                # Save plots for this specific method
                # Note: Ensure save_outlier_plot is defined in your class
                Outlier_handling.save_outlier_plot(X_tr_mod, X_te_mod, m)

                results[m] = (X_tr_mod, X_te_mod)

                # Calculate average skewness to see the impact
                avg_skew = X_tr_mod.select_dtypes(include='number').skew().mean()

                summary.append([m, X_tr_mod.shape[0], X_te_mod.shape[0], round(avg_skew, 4)])
                logger.info(f"Method '{m}' processed successfully.")

            except Exception as e:
                logger.error(f"Failed to apply {m}: {str(e)}")

        # Create the summary table
        table_str = tabulate(summary, headers=["Method", "Train Rows", "Test Rows", "Avg Skew"], tablefmt="grid")

        # Log the full summary
        logger.info("\nALL TECHNIQUES COMPARISON:\n" + table_str)
        print(table_str)