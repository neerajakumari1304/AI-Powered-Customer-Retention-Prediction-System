import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('visualization')

def visualization(df,logger):
    try:
        sns.set_theme(style="whitegrid")

        # 1.Distribution of Churn
        plt.figure(figsize=(6, 4))
        sns.countplot(data=df, x='Churn', hue='Churn', palette='Greens')
        plt.title('Distribution of Churn')
        plt.savefig('churn_distribution.png')
        plt.close()
        logger.info("Saved:churn_distribution.png")
        logger.info(f"Churn Distribution:\n{df.groupby('Churn').size().to_string()}")

        # 2.telecom_partner vs gender
        plt.figure(figsize=(10,6))
        sns.countplot(data=df, x='telecom_partner', hue='gender', palette='YlGn')
        plt.title('telecom partner')
        plt.xlabel('Count')
        plt.legend(title='gender')
        plt.savefig('telecom_partner_vs_gender.png')
        plt.close()
        logger.info("saved:telecom_partner_vs_gender.png")
        logger.info(f"telecom partner vs gender:\n{df.groupby(['telecom_partner','gender']).size().unstack()}")

        # 3.SeniorCitizen distribution in Telecom Partners
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x='telecom_partner', hue='SeniorCitizen', palette='BuGn')
        plt.title('Distribution of Senior Citizens across Telecom Partners')
        plt.xlabel('Telecom Partner')
        plt.ylabel('Number of Customers')
        plt.legend(title='Is Senior Citizen')
        plt.savefig('senior_citizen_in_telecom_partner.png')
        plt.close()
        logger.info("saved:senior_citizen_in_telecom_partner.png")
        logger.info(f"senior citizen in telecom partner:\n{df.groupby(['SeniorCitizen','telecom_partner']).size().unstack()}")

        # 4.gender distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='gender', palette='Blues')
        plt.title('Gender Distribution of Customers')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.savefig('gender_distribution.png')
        plt.close()
        logger.info("saved: gender_distribution.png")
        logger.info(f"Gender Distribution:\n{df['gender'].value_counts().to_string()}")

        # 5.Gender vs Churn (To see if gender affects retention)
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='gender', hue='Churn', palette='Oranges')
        plt.title('Churn by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.savefig('gender_churn.png')
        plt.close()
        logger.info("saved: Gender vs Churn.png")
        logger.info(f"Gender vs Churn:\n{df.groupby(['gender', 'Churn']).size().reset_index(name='Count').to_string(index=False)}")

        # 6.Gender Breakdown for Customers with No Internet Service
        plt.figure(figsize=(8, 6))
        sns.countplot(x='gender', data=df.query("InternetService == 'No'"), palette=['skyblue', 'salmon'], edgecolor='black')
        plt.title('Gender Breakdown for Customers with No Internet Service')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.savefig('gender_no_internet.png')
        plt.close()
        logger.info("saved: Gender Breakdown for Customers with No Internet Service.png")
        logger.info(f"Gender breakdown for 'No Internet Service':\n{df[df['InternetService'] == 'No'].groupby('gender').size().to_string()}")

        # 7.Tenure vs Churn
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette='BuGn', element='step')
        plt.title('Tenure Distribution by Churn')
        plt.savefig('tenure_vs_churn.png')
        plt.close()
        logger.info("Saved: tenure_vs_churn.png")
        logger.info(f"Average Tenure by Churn Status:\n{df.groupby('Churn')['tenure'].mean().to_string()}")

        #  8.Churn by Contract
        plt.figure(figsize=(8, 5))
        contract_churn = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        sns.barplot(data=contract_churn, x='Contract', y='Count', hue='Churn', palette='YlGn')
        plt.title('Churn by Contract Type')
        plt.savefig('contract_vs_churn.png')
        plt.close()
        logger.info("Saved: contract_vs_churn.png")
        logger.info(f"Contract Type Churn Counts:\n{df.groupby(['Contract', 'Churn']).size().reset_index(name='Count').to_string(index=False)}")

        # 9.Churn by Internet Service
        plt.figure(figsize=(8, 5))
        internet_churn = df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count')
        sns.barplot(data=internet_churn, x='InternetService', y='Count', hue='Churn', palette='Greys')
        plt.title('Churn by Internet Service Type')
        plt.savefig('internet_service_vs_churn.png')
        plt.close()
        logger.info("Saved: Churn by Internet Service.png")
        logger.info(f"Internet Service Churn Counts:\n{df.groupby(['InternetService', 'Churn']).size().reset_index(name='Count').to_string(index=False)}")

        # 10.Churn by Payment Method
        plt.figure(figsize=(12, 6))
        payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
        sns.barplot(data=payment_churn, x='PaymentMethod', y='Count', hue='Churn', palette='Greens')
        plt.xticks(rotation=15)
        plt.title('Churn by Payment Method')
        plt.tight_layout()
        plt.savefig('payment_method_vs_churn.png')
        plt.close()
        logger.info("Saved: Churn by Payment Method.png")
        logger.info(f"Payment Method Churn Counts:\n{df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count').to_string(index=False)}")

        # 11.SeniorCitizen vs Churn
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='SeniorCitizen', hue='Churn', palette='Greys_r')
        plt.title('Churn Count by Senior Citizen Status')
        plt.xlabel('Senior Citizen (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.savefig('senior_citizen_vs_churn.png')
        plt.close()
        logger.info("Saved: Churn by senior citizen.png")
        logger.info(f'{df.groupby(['SeniorCitizen', 'Churn']).size()}')

        # 12.Senior Citizens within each Gender
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x='gender', hue='SeniorCitizen', palette=['skyblue', 'orange'])
        plt.title('Senior Citizen Distribution by Gender')
        plt.legend(title='Senior Citizen', labels=['No', 'Yes'])
        plt.ylabel('Count')
        plt.savefig('senior_citizen_gender_direct.png')
        plt.close()
        logger.info("Saved: Senior Citizen Distribution by Gender.png")
        logger.info(f"Senior Citizen vs Gender Breakdown:\n{df.groupby(['gender', 'SeniorCitizen']).size().unstack().to_string()}")

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no : {error_line.tb_lineno}:due to {error_msg}')