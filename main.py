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
import pickle
from scipy.stats import skew

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler,Normalizer

from visualization import visualization
from missing_value import IMPUTE_DATA
from variable_transformation import SCALER
from outlier_handling import Outlier_handling
from tabulate import tabulate
from feature_selection import FEATURE_SELECTION
from cat_to_num import CAT_TO_NUM
from data_balance import DATA_BALANCE
from all_model import common

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
            logger.info(f' {self.df.info()}')
            logger.info(f'Missing Values:{self.df.isnull().sum()}')
            self.df.drop('customerID', axis=1, inplace=True)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            logger.info(f' {self.df.info()}')
            logger.info(f'total rows in the data : {self.df.shape[0]}')
            logger.info(f'total columns in the data : {self.df.shape[1]}')
            logger.info(f"Sample data with new column:\n{self.df.head().to_string()}") #sample data

            self.y = self.df['Churn'].map({'Yes': 1, 'No': 0})
            self.X = self.df.drop(columns=['Churn'])

            logger.info(f'Independent Column(X): {self.X.shape}')
            logger.info(f'Dependent Column(y): {self.y.shape}')

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=42)

            logger.info(f'X_train Columns:{self.X_train.columns}')
            logger.info(f'X_test Columns:{self.X_test.columns}')

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'training data size:{self.X_train.shape}')
            logger.info(f'testing data size:{self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')


    def visualize(self):
        try:
            viz_logger = setup_logging('visualization')  # Initialize a separate logger for visualization
            viz_logger.info("start visualize")
            visualization(self.df, viz_logger) # This ensures everything inside the function logs to visualization.log
            viz_logger.info("visualize completed successfully")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

    def missing_value(self):
        try:
            missing_value = setup_logging('missing_value')

            missing_value.info('Selecting Missing Value Technique')
            missing_value.info(f'Before X_train:\n{self.X_train.isnull().sum()}')
            missing_value.info(f'Before X_test:\n{self.X_test.isnull().sum()}')

            techniques = {
                'mean': IMPUTE_DATA.mean_impute,
                'median': IMPUTE_DATA.median_impute,
                'mode': IMPUTE_DATA.mode_impute,
                'ffill': IMPUTE_DATA.forward_fill,
                'bfill': IMPUTE_DATA.backward_fill,
                'random': IMPUTE_DATA.random_sample_impute
            }

            X_train_filled = self.X_train.copy()
            X_test_filled = self.X_test.copy()
            best_technique_per_column = {}

            missing_cols = self.X_train.columns[self.X_train.isnull().any()]

            for col in missing_cols:
                scores = {}
                for name, func in techniques.items():
                    try:
                        X_train_col, _ = func(self.X_train[[col]].copy(), self.X_test[[col]].copy())
                        if pd.api.types.is_numeric_dtype(self.X_train[col]):
                            original_var = self.X_train[col].var()
                            new_var = X_train_col[col].var()
                            var_diff = abs(original_var - new_var)
                        else:
                            var_diff = abs(self.X_train[col].isnull().sum() - X_train_col[col].isnull().sum())
                        scores[name] = var_diff
                    except Exception:
                        continue

                best_tech = min(scores, key=scores.get)
                best_technique_per_column[col] = best_tech

                #  Correct assignment: extract the column
                train_imp, test_imp = techniques[best_tech](self.X_train[[col]].copy(),
                                                            self.X_test[[col]].copy())
                X_train_filled[col] = train_imp[col]
                X_test_filled[col] = test_imp[col]

                missing_value.info(f'Best impute for {col}: {best_tech}')

            self.X_train = X_train_filled
            self.X_test = X_test_filled

            missing_value.info(f'After X_train missing:\n{self.X_train.isnull().sum()}')
            missing_value.info(f'After X_test missing:\n{self.X_test.isnull().sum()}')
            missing_value.info(f'Summary of chosen techniques: {best_technique_per_column}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def variable_transform(self):
        try:
            variable_transformation = setup_logging('variable_transformation')
            variable_transformation.info('Selecting Variable Transformation Technique')
            variable_transformation.info(f'X_train Columns : {self.X_train.columns}')
            variable_transformation.info(f'X_test Columns: {self.X_test.columns}')

            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            variable_transformation.info(f'X_train Numerical columns : {self.X_train_num.columns}')
            variable_transformation.info(f'X_train Categorical columns: {self.X_train_cat.columns}')
            variable_transformation.info(f'X_test Numerical columns: {self.X_test_num.columns}')
            variable_transformation.info(f'X_test Categorical columns: {self.X_test_cat.columns}')

            variable_transformation.info(f'X_train Numerical Shape: {self.X_train_num.shape}')
            variable_transformation.info(f'X_train Categorical Shape: {self.X_train_cat.shape}')
            variable_transformation.info(f'X_test Numerical Shape: {self.X_test_num.shape}')
            variable_transformation.info(f'X_test Categorical Shape: {self.X_test_cat.shape}')

            techniques = {
                'standard': SCALER.standard_scaling,
                'minmax': SCALER.minmax_scaling,
                'robust': SCALER.robust_scaling,
                'log': SCALER.log_transform,
                'power': SCALER.power_transform,
                'boxcox': SCALER.boxcox_transform,
                'yeojohnson': SCALER.yeojohnson_transform,
                'quantile': SCALER.quantile_transform
            }

            best_technique_per_feature = {}
            X_train_transformed, X_test_transformed = self.X_train_num.copy(), self.X_test_num.copy()

            for col in self.X_train_num.columns:
                # compute skewness for each technique
                skew_scores = {
                    name: abs(skew(func(self.X_train_num[[col]], self.X_test_num[[col]])[0][col], nan_policy='omit'))
                    for name, func in techniques.items()
                    if func(self.X_train_num[[col]], self.X_test_num[[col]])[0][col].notnull().any()
                }
                if not skew_scores:
                    continue

                # pick best technique
                best_tech = min(skew_scores, key=skew_scores.get)
                best_technique_per_feature[col] = best_tech

                # apply best technique
                train_imp, test_imp = techniques[best_tech](self.X_train_num[[col]], self.X_test_num[[col]])
                X_train_transformed[col], X_test_transformed[col] = train_imp[col], test_imp[col]

                variable_transformation.info(f'{col}: {best_tech} | Skew={skew_scores[best_tech]:.4f}')

            # replace numeric data
            self.X_train_num, self.X_test_num = X_train_transformed, X_test_transformed
            variable_transformation.info(f'Summary: {best_technique_per_feature}')


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def outlier_handling(self):
        outlier_logger = setup_logging('outlier_handling')
        try:
            # --- 1. Log Initial Metadata (Dtypes & Memory) ---
            counts = self.X_train_num.dtypes.value_counts()
            dtype_str = ", ".join([f"{dtype}({count})" for dtype, count in counts.items()])
            mem_mb = self.X_train_num.memory_usage(deep=True).sum() / (1024 ** 2)

            outlier_logger.info(f"dtypes: {dtype_str}")
            outlier_logger.info(f"memory usage: {mem_mb:.1f}+ MB")

            X_train_out = self.X_train_num.copy()
            X_test_out = self.X_test_num.copy()

            # Library of techniques to test
            techniques = {
                'iqr': Outlier_handling.iqr_method,
                'zscore': Outlier_handling.zscore_method,
                'winsor': Outlier_handling.winsorization,
                'clip': Outlier_handling.clipping,
                'log': Outlier_handling.log_outlier,
                'none': Outlier_handling.no_outlier
            }

            summary = []

            # --- 2. Process Column by Column ---
            for col in self.X_train_num.columns:
                skews = {}
                for name, func in techniques.items():
                    try:
                        # Test each function on a single column
                        train_temp, _ = func(self.X_train_num[[col]], self.X_test_num[[col]])
                        skews[name] = abs(train_temp[col].skew())
                    except:
                        continue

                # Identify the best technique (lowest absolute skewness)
                best_tech_name = min(skews, key=skews.get)

                # Apply the winner to the column
                train_fixed, test_fixed = techniques[best_tech_name](self.X_train_num[[col]], self.X_test_num[[col]])
                X_train_out[col] = train_fixed[col]
                X_test_out[col] = test_fixed[col]

                # --- 3. Save Plot & Log the Path ---
                # This triggers the "Visual check saved" line in your log
                Outlier_handling.plot_comparison(self.X_train_num, X_train_out, col, best_tech_name, outlier_logger)

                summary.append([col, best_tech_name, X_train_out.shape[0]])

            # Update final attributes
            self.X_train_num, self.X_test_num = X_train_out, X_test_out

            # --- 4. Log the Summary Table ---
            table_str = tabulate(summary, headers=["Column", "Best Method", "Rows"], tablefmt="grid")
            outlier_logger.info("\n" + table_str)

            return self.X_train_num, self.X_test_num


        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def cat_to_num(self):
        cat = setup_logging('cat_to_num')
        try:
            X_train_enc = pd.DataFrame()
            X_test_enc = pd.DataFrame()
            logger.info('Starting Column-wise Encoding Competition')

            # Separate Numeric and Categorical
            cat_cols = self.X_train.select_dtypes(include='object').columns.tolist()

            self.X_train_num = self.X_train.drop(columns=cat_cols)
            self.X_test_num = self.X_test.drop(columns=cat_cols)

            self.X_train_cat = self.X_train[cat_cols]
            self.X_test_cat = self.X_test[cat_cols]

            self.X_train_enc = pd.DataFrame(index=self.X_train_cat.index)
            self.X_test_enc = pd.DataFrame(index=self.X_test_cat.index)

            techniques = {
                'label': CAT_TO_NUM.label_encoding,
                'onehot': CAT_TO_NUM.one_hot_encoding,
                'frequency': CAT_TO_NUM.frequency_encoding,
                'binary': CAT_TO_NUM.binary_encoding,
                'ordinal': CAT_TO_NUM.ordinal_encoding
            }

            # Iterate through each categorical column individually
            for col in cat_cols:
                scores = {}
                temp_results = {}

                for name, func in techniques.items():
                    try:
                        # Test the technique on this single column
                        # We wrap it in a list [[col]] to keep it as a DataFrame
                        res = func(self.X_train_cat[[col]], self.X_test_cat[[col]])

                        if res is not None:
                            train_col, test_col = res
                            scores[name] = train_col.shape[1]
                            temp_results[name] = (train_col, test_col)
                    except Exception as e:
                        logger.debug(f'Technique {name} failed for column {col}: {e}')
                        continue

                if not scores:
                    logger.warning(f'No encoding succeeded for {col}. Skipping.')
                    continue

                #  Pick the winner (Method that created the fewest features)
                best_tech = min(scores, key=scores.get)
                logger.info(f'Best encoding for "{col}": {best_tech} (Features: {scores[best_tech]})')

                # Concatenate the best columns
                train_best, test_best = temp_results[best_tech]
                X_train_enc = pd.concat([X_train_enc, train_best], axis=1)
                X_test_enc = pd.concat([X_test_enc, test_best], axis=1)

            # Combine everything back together
            self.X_train_final = pd.concat([self.X_train, self.X_train_enc], axis=1)
            self.X_test_final = pd.concat([self.X_test, self.X_test_enc], axis=1)

            #  Ensure indices are aligned
            self.y_train = self.y_train.loc[self.X_train_final.index]
            self.y_test = self.y_test.loc[self.X_test_final.index]

            logger.info(f'Encoding completed. Final shape: {self.X_train_final.shape}')
            return self.X_train_final, self.X_test_final, self.y_train, self.y_test
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def fs(self):
        fs = setup_logging('feature_selection')
        try:
            fs.info(f'Initial State: {self.X_train.shape[1]} features')

            self.X_train_numeric = self.X_train.select_dtypes(include=['number'])
            self.X_test_numeric = self.X_test.select_dtypes(include=['number'])

            #  Clean Data (Handle Inf and NaN for sklearn compatibility)
            self.X_train = self.X_train.replace([np.inf, -np.inf], np.nan)
            self.X_test = self.X_test.replace([np.inf, -np.inf], np.nan)

            if self.X_train.isnull().sum().sum() > 0:
                fs.info('Imputing missing values with median')
                filler = X_train.median()
                self.X_train = self.X_train.fillna(filler)
                self.X_test = self.X_test.fillna(filler)

            techniques = {
                'variance': FEATURE_SELECTION.drop_low_variance,
                'correlation': FEATURE_SELECTION.drop_redundant_features,
                'kbest': FEATURE_SELECTION.select_top_features,
                'rfe': FEATURE_SELECTION.select_by_importance
            }
            #  Competitive Selection
            feature_counts = {}

            for name, func in techniques.items():
                try:
                    # Run the static method from your FEATURE_SELECTION class
                    result = func(self.X_train, self.X_test, self.y_train)

                    if result is not None:
                        self.X_train_temp, _ = result
                        count = self.X_train_temp.shape[1]

                        # Ensure we don't pick a method that dropped EVERYTHING
                        if count > 0:
                            feature_counts[name] = count
                            fs.info(f'Technique "{name}" kept {count} features')
                        else:
                            fs.warning(f'Technique "{name}" dropped all features. Skipping.')

                except Exception as e:
                    fs.error(f'{name} failed: {e}')
                    continue

            if not feature_counts:
                fs.error('All techniques failed or returned 0 features.')
                return self.X_train, self.X_test

            #  Choose the "Winner" (Minimum features)
            best_name = min(feature_counts, key=feature_counts.get)
            fs.info(f'>>>best: {best_name} ({feature_counts[best_name]} features)')

            # Run the winning technique one last time to get final data
            return techniques[best_name](self.X_train, self.X_test, self.y_train)
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

    def data_balance(self):
        try:
            logger.info('Selecting Data Balancing Technique (ROW-WISE)')

            # 1. Reset indices
            X = self.X_train.reset_index(drop=True)
            y = self.y_train.reset_index(drop=True)

            assert len(X) == len(y), "X and y are misaligned before balancing"

            # 2. Prepare a Numeric-Only version for the Evaluator
            # This prevents 'airtel' from crashing the LogisticRegression test
            X_eval = X.select_dtypes(include=[np.number])

            if X_eval.empty:
                logger.error("No numeric columns found for evaluation! Ensure Encoding happens before Balancing.")
                return X, y

            techniques = {
                'smote': DATA_BALANCE.smote,
                'ros': DATA_BALANCE.random_over_sampling,
                'rus': DATA_BALANCE.random_under_sampling,
                'smote_tomek': DATA_BALANCE.hybrid_balancing,
                'smote_enn': DATA_BALANCE.smote_enn,
                'adasyn': DATA_BALANCE.adasyn
            }

            scores = {}

            for name, func in techniques.items():
                try:
                    # Apply balancing to the numeric data for testing
                    X_res, y_res = func(X_eval, y)

                    if len(X_res) != len(y_res):
                        continue

                    # Test the technique using Logistic Regression
                    model = LogisticRegression(max_iter=1000)
                    f1 = cross_val_score(model, X_res, y_res, scoring='f1', cv=3).mean()

                    scores[name] = f1
                    logger.info(f'{name} F1 score: {round(f1, 4)}')
                except Exception as e:
                    logger.info(f'{name} failed during evaluation: {e}')
                    continue

            # 3. Safety Check: If all failed, return original data
            if not scores:
                logger.warning("CRITICAL: All balancing techniques failed! Returning original data.")
                return X, y

            # 4. Identify the winner
            best = max(scores, key=scores.get)
            logger.info(f'Best Balancing Method Selected: {best}')

            # 5. Apply the WINNING method to the FULL data (X, not X_eval)
            # Note: This will still fail if the winning function (like SMOTE)
            # hasn't been updated to handle strings.
            self.X_final, self.y_final = techniques[best](X, y)

            logger.info(f'After Balancing:\n{self.y_final.value_counts()}')
            return self.X_final, self.y_final

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return self.X_train, self.y_train

    def feature_scaling(self):
        try:
            logger.info('Selecting Best Feature Scaling Technique')

            # 1. DEFINE X_numeric IMMEDIATELY
            # This filters out 'airtel' so the scalers don't crash
            X_numeric = self.X_train.select_dtypes(include=[np.number])

            if X_numeric.empty:
                logger.error("No numeric columns found! Scalers cannot process strings like 'airtel'.")
                return self.X_train_bal, self.X_test

            scalers_dict = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'maxabs': MaxAbsScaler(),
                'normalize': Normalizer()
            }

            scores = {}

            # 2. Loop through the scalers using the numeric-only data
            for name, scalers in scalers_dict.items():
                try:
                    # Use X_numeric here, NOT self.X_train_bal
                    X_scaled = scalers.fit_transform(X_numeric)

                    model = LogisticRegression(max_iter=1000)
                    f1 = cross_val_score(model, X_scaled, self.y_train, scoring='f1', cv=3).mean()

                    scores[name] = f1
                    # Note: Using '->' instead of special arrows to avoid logging errors
                    logger.info(f'{name} scaler -> F1 score: {round(f1, 4)}')

                except Exception as e:
                    logger.info(f'{name} scaler failed -> skipped | {e}')

            if not scores:
                logger.warning("All scalers failed. Returning original data.")
                return self.X_train, self.X_test

            best = max(scores, key=scores.get)
            logger.info(f'Best Scaling Technique Selected: {best}')

            best_scaler= scalers_dict[best]

            # 3. Final transformation (also needs to be numeric only)
            X_test_numeric = self.X_test.select_dtypes(include=[np.number])

            self.X_train_scaled = pd.DataFrame(best_scaler.fit_transform(X_numeric), columns=X_numeric.columns)
            self.X_test_scaled = pd.DataFrame(best_scaler.transform(X_test_numeric), columns=X_test_numeric.columns)

            with open('scaler_path.pkl', 'wb') as f:
                pickle.dump(best_scaler, f)

            common(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test)

            return self.X_train_scaled, self.X_test_scaled

        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')
            return self.X_train, self.y_train

if __name__ == "__main__":
    try:
        obj = customer_rentention_prediction_system('D:\\Customer_Retention_Prediction_System\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
        obj.visualize()
        obj.missing_value()
        obj.variable_transform()
        obj.outlier_handling()
        obj.cat_to_num()
        obj.fs()
        obj.data_balance()
        obj.feature_scaling()

    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')

