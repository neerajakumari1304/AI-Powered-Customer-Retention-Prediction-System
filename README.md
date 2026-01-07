# AI-Powered-Customer-Retention-Prediction-System
The goal of this project is to analyze customer data from a telecommunications company to predict whether a customer will churn (leave the service) or stay.

---
##  Step 1: Exploratory Data Analysis (EDA)
### Objective
The purpose of EDA is to analyze the telecom dataset and identify key drivers of customer churn. Using Matplotlib and Seaborn, we visualize customer demographics, service usage, and billing patterns to uncover why customers leave. These insights form the foundation for predictive modeling and retention strategies.

---

###  Dataset Description
- **Rows**: 7,043  
- **Columns**: 21  
- **Source**: [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
---
**Features:**
- **Demographics**: Gender, Senior Citizen, Partner, Dependents  
- **Services**: Phone, Internet, Online Security, Tech Support, Streaming  
- **Account & Billing**: Tenure, Contract type, Payment method, Monthly charges, Total charges  
- **Target Variable**: `Churn` (Yes/No)  
---
### Tools & Technologies
- **Python**  
- **Pandas** – Data loading & cleaning  
- **Matplotlib & Seaborn** – Visualization  
- **Logging Module** – Workflow tracking & error handling
- ---
- ### Feature Augmentation
- Added a synthetic variable: **Telecom Partner (Jio, Airtel, BSNL, Vodafone)**.  
- Random assignment ensures balanced demographics across partners.  
- Simulates competitive market conditions for deeper analysis.  
---
###  Key Visualizations & Insights
## 1️ Churn Distribution
- **Graph**: Bar Chart (countplot)  
- **Values**: No = 5,174 | Yes = 1,869  
- **Observation**: Majority of customers did not churn.  
- **Insight**: Dataset shows **class imbalance** → need metrics beyond accuracy (Precision, Recall, F1).  

---

## 2️ Telecom Partner vs Gender
![Telecom Partner vs Gender](images/telecom_partner_vs_gender.png)
- **Graph**: Bar Chart (countplot)  
- **Values**: Even distribution across partners.  
- **Observation**: Male/Female counts are balanced for each partner.  
- **Insight**: Confirms synthetic partner assignment worked → fair test environment.  

---

## 3️ Senior Citizen in Telecom Partner
- **Graph**: Bar Chart (countplot)  
- **Values**: Non-Seniors ≈ 1,450 per partner | Seniors ≈ 280 per partner.  
- **Observation**: Ratio of young to old consistent across partners.  
- **Insight**: If churn differs later, it’s due to service quality, not demographics.  

---

## 4️ Gender Distribution
- **Graph**: Bar Chart (countplot)  
- **Values**: Male = 3,555 | Female = 3,488  
- **Observation**: Customer base is nearly 50/50.  
- **Insight**: Balanced dataset → no gender bias in churn analysis.  

---

## 5️ Gender vs Churn
- **Graph**: Bar Chart (countplot)  
- **Values**: Female (No = 2,549, Yes = 939) | Male (No = 2,625, Yes = 930)  
- **Observation**: Churn is nearly identical for both genders.  
- **Insight**: Gender is **not a churn driver**.  

---

## 6️ Gender Breakdown (No Internet Service)
- **Graph**: Bar Chart (countplot with query)  
- **Values**: Female = 747 | Male = 779  
- **Observation**: Equal split in “Basic Phone Only” segment.  
- **Insight**: Legacy users’ retention depends on call reliability & basic plan costs.  

---

## 7️ Tenure vs Churn
- **Graph**: Histogram with KDE (histplot)  
- **Values**: Stayed avg. = 37.5 months | Churned avg. = 17.9 months  
- **Observation**: Customers churn early (within 1.5 years).  
- **Insight**: First year is the **danger zone** → onboarding & loyalty rewards critical.  

---

## 8️ Contract Type vs Churn
- **Graph**: Bar Chart (barplot)  
- **Values**:  
  - Month-to-month: No = 2,220 | Yes = 1,655  
  - One year: No = 1,307 | Yes = 166  
  - Two year: No = 1,647 | Yes = 48  
- **Observation**: Month-to-month contracts show highest churn.  
- **Insight**: **Contract length is a key retention driver**.  

---

## 9️ Internet Service vs Churn
- **Graph**: Bar Chart (barplot)  
- **Values**:  
  - DSL: No = 1,962 | Yes = 459  
  - Fiber optic: No = 1,799 | Yes = 1,297  
  - No Internet: No = 1,413 | Yes = 113  
- **Observation**: Fiber optic users churn more.  
- **Insight**: Premium service dissatisfaction → needs audit.  

---

## 10 Payment Method vs Churn
- **Graph**: Bar Chart (barplot)  
- **Values**:  
  - Bank transfer (auto): No = 1,286 | Yes = 258  
  - Credit card (auto): No = 1,290 | Yes = 232  
  - Electronic check: No = 1,294 | Yes = 1,071  
  - Mailed check: No = 1,304 | Yes = 308  
- **Observation**: Electronic check users churn most.  
- **Insight**: Auto-pay reduces churn → encourage adoption.  

---

## 1️1️ Senior Citizen vs Churn
- **Graph**: Bar Chart (countplot)  
- **Values**:  
  - Non-Seniors: No = 4,508 | Yes = 1,393  
  - Seniors: No = 666 | Yes = 476  
- **Observation**: Seniors churn at higher rates.  
- **Insight**: Senior-friendly support & billing could improve retention.

- ---

## EDA Conclusion
- Month-to-month plans → massive churn.  
- High monthly bills → churn trigger.  
- Electronic checks → highest churn risk.  
- Senior citizens → vulnerable group.  
- Gender → not a churn driver.  
- Fiber optic users → churn more than DSL.  

✅ Step 1 completed: dataset is now ready for **feature engineering & model building**.
