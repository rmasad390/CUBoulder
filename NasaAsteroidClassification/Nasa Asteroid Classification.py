#!/usr/bin/env python
# coding: utf-8

# # NASA Asteroid Classification 

# ## Introduction
# Nasa regularly monitors asteroids that are near earth including their distance from earth, speed, and other important data points. This data is then used to predict whether the asteroid is a potential risk to us. In this project we will attempt to build a classification model that aims to predict whether an asteroid is potentially hazardous and needs further analysis. In the upcoming analysis, I will take you through the following phases to reach the projects conclusion.
# 
# 1. Data Preprocessing
#     - Data Cleaning
#     - Exploratory Analysis
# 2. Data Modeling
#     - Logistics Regression
#     - Random Forest Classifier
#     - AdaBoost Classifier
#     - XGBoost Classifier
# 3. Data Evaluation
#     - Cross Validation
#     - ROC AUC Curves
# 4. Conclusion

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(37)

from sklearn.model_selection import train_test_split

# Simple Models
from sklearn.linear_model import LogisticRegression

# Better Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler


# ## Load Data and Exploratory Analysis

# This dataset was obtained via a Kaggle Dataset which was further pulled from a NASA dataset via their API.  
# - **Kaggle Dataset:** https://www.kaggle.com/datasets/lovishbansal123/nasa-asteroids-classification
# - **NASA Dataset:** https://cneos.jpl.nasa.gov/

# In[2]:


df = pd.read_csv('nasa.csv')
pd.set_option('display.max_columns', None)
df.head()


# ## Data
# This dataset containts 4,687 data entries with a total of 40 columns. Of these, there are no blanks, but some of the columns do not appear to be necessary for a Classification Model. 
# 
# The dataset consists of approximately 84% Not Hazardous Occurrences and 16% Hazardous occurrences. This does lead to some challenges due to the model having a higher chance of predicting correctly by just predicting False. This is inversely related with the magnitude of the decision. A False Negative in this model will have a drastically worse impact then a False Positive, so this will need to be taken into account when developing our Model.

# In[3]:


df.info()


# In[4]:


hazardous_counts = df['Hazardous'].value_counts()
plt.pie(hazardous_counts, labels=hazardous_counts.index, autopct='%1.1f%%')
plt.title('Breakdown of Hazardous Asteroids')
plt.show() 


# ## Data Cleaning and Exploratory Data Analysis
# 1. columns not useful for classification (IDs, names, Dates)
# 2. columns that have same metric in various formats (collinearity)
# 3. dataset is not evenly distributed between false and true

# Lets start by checking how many unique values there are per columns.

# In[5]:


for col in df.columns:
    print(f'{col}: {df[col].nunique()}')


# Some things stand out. Orbiting Body and Equinox both only have 1 option and as such are not necessary for this dataset. The bigger call out is that Neo Reference/Name only have 3692 values of the total dataset. This is repeated in a good number of rows indicating potential duplicates. Further analysis is necessary to determine if these duplicates should be left because they may be significantly different or not. (for example: `Relative Velocity` and `Miss Dist` are unique, and may be important to this model, if they are significantly important then they should be left in the dataset).

# In[6]:


df = df.drop(columns=['Orbit ID', 'Close Approach Date',  'Orbit Determination Date', 'Equinox', 'Orbiting Body'])


# ### Correlation Matrix
# The below correlation matrix shows that a number of variables are perfectly correlated with each other. Many of these make sense, Name and ID and the various distance formats being perfectly correlated makes sense. As these are a good sign of collinearity, we will need to remove these.

# In[7]:


correlation = df.corr()

plt.figure(figsize=(20, 12))
sns.heatmap(correlation, annot=True, fmt = ".1f", cmap = 'coolwarm')


# ### Checking Attributes not in Dist
# We still see a few values with high correlation, namely `Orbital Period`, `Aphelion Distance`, and `Semi Major Axis`, as well as a few other variables. After checking the data for these rows, I  determined that they are significantly different enough and should all be included in the analysis.

# In[8]:


fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].scatter(df['Orbital Period'], df['Semi Major Axis'], color='blue', alpha=0.5)
axes[0, 0].set_title('Orbital Period vs Semi Major Axis')

axes[0, 1].scatter(df['Aphelion Dist'], df['Semi Major Axis'], color='blue', alpha=0.5)
axes[0, 1].set_title('Aphelion Dist vs Semi Major Axis')

axes[1, 0].scatter(df['Jupiter Tisserand Invariant'], df['Mean Motion'], color='blue', alpha=0.5)
axes[1, 0].set_title('Jupiter Tisserand Invariant vs Mean Motion')

axes[1, 1].scatter(df['Epoch Osculation'], df['Perihelion Time'], color='blue', alpha=0.5)
axes[1, 1].set_title('Epoch Osculation vs Perihelion Time')

plt.tight_layout()
plt.show()


# ### Checking Names
# After manually checking some of the name values, I think we can leave them in the dataset. They are the same asteroid being recorded at different time periods with different distances from the earth and speeds. This seems amply different enough to leave them in the dataset. We will still be dropping `Neo Reference ID` and `Name` from the dataset as they are not needed for classification.

# In[9]:


duplicates = df[df.duplicated('Name', keep=False)]
duplicates_sorted = duplicates.sort_values('Name')
duplicates_sorted.head(20)


# Before dropping Diameter columns, we will average 1 of them to use the midpoint instead of the min or Max. 

# In[10]:


df['Est Dia in M'] = (df['Est Dia in M(max)'] + df['Est Dia in M(min)'])/2


# In[11]:


df = df.drop(columns=['Neo Reference ID', 'Name', 'Est Dia in KM(min)', 'Est Dia in KM(max)', 'Est Dia in M(min)' ,'Est Dia in M(max)', 'Est Dia in Miles(min)',
                      'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)', 'Relative Velocity km per hr', 
                      'Miles per hour', 'Miss Dist.(lunar)', 'Miss Dist.(kilometers)', 'Miss Dist.(miles)', 'Epoch Date Close Approach'])

# Reorder Columns
columns_except_specific = [col for col in df.columns if col != 'Hazardous']
new_column_order = columns_except_specific + ['Hazardous']
df = df[new_column_order]


# ### Correlation Matrix pt 2
# After rerunning the Correlation Matrix, the dataset consisting of 19 remaining attribute columns appears good to continue on to further analysis.

# In[12]:


correlation = df.corr()

plt.figure(figsize=(15, 12))
sns.heatmap(correlation, annot=True, fmt = ".2f", cmap = 'coolwarm')


# ### Data Distribution
# When checking the data Distributions below, we see that the Data is spread out similarly between Hazardous and Not Hazardous. However, Many of the attributes do not appear to be normally distributed. It will be good to check for outliers and potentially remove them. More data collection would likely be necessary to have accurate predictions for major outliers.

# In[13]:


plt.figure(figsize = (15, 15))
for i, col in enumerate(df.columns[1:-1], 1):
    plt.subplot(5, 4, i)
    sns.histplot(x = df[col], hue = df["Hazardous"], multiple = "dodge")
    plt.title(col)

plt.tight_layout()
plt.show()


# ## Outliers

# There are currently 4,687 entries in the dataset prior to outlier removal.

# In[14]:


attribute_columns = df.columns[:-1]

plt.figure(figsize=(15, 20))

for i, column in enumerate(attribute_columns, 1):
    plt.subplot(5, 4, i)
    sns.boxplot(y=df[column])
    plt.title(column)

plt.tight_layout()
plt.show()


# For now lets check on how many outliers would be removed if we remove based off Standard IQR method.

# In[15]:


robust_scaler = RobustScaler()
df[attribute_columns] = robust_scaler.fit_transform(df[attribute_columns])

def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((series < lower_bound) | (series > upper_bound)).sum()

for column in attribute_columns :
    outlier_count = count_outliers(df[column])
    print(f'Outliers in {column}: {outlier_count}')


# Removing all of these outliers may be a problem especially in the case of the following 3 columns: `Epoch Osculation`, `Est Diameter`, and `Perihelion Time`. After Further Analysis, I will not be removing any outliers as many of these should be predicted for. 

# In[16]:


def clean_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (IQR * 1.5)
    upper_limit = Q3 + (IQR * 1.5)
    data_out = data[((data[column] > upper_limit) | (data[column] < lower_limit))]
    data = data[~((data[column] > upper_limit) | (data[column] < lower_limit))]
    return data, data_out

df_cleaned = df.copy()
df_outliers = pd.DataFrame()
cols = []

'''cols = ['Absolute Magnitude', 'Relative Velocity km per sec', 'Miss Dist.(Astronomical)', 'Orbit Uncertainity',
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg', 'Aphelion Dist',  
        'Mean Anomaly', 'Mean Motion']'''

for column in cols:
    df_cleaned, temp_outliers = clean_outliers(df_cleaned, column)
    
    df_outliers = pd.concat([df_outliers, temp_outliers]).reset_index(drop=True)
    
print(df_cleaned.shape)
print(df_outliers.shape)


# In[17]:


df_outliers
df = df_cleaned.copy()


# In[18]:


df.describe()


# In[19]:


plt.figure(figsize=(15, 20))

for i, column in enumerate(attribute_columns, 1):
    plt.subplot(5, 4, i)
    sns.boxplot(y=df[column])
    plt.title(column)

plt.tight_layout()
plt.show()


# ## Model Preprocessing

# ### Split Data
# 
# With the Exploratory Analysis and Data Cleaning Complete, it is time to start splitting the dataset. I am checking y_train and y_test to ensure that the data was split including both True and False Data.

# In[20]:


X = df.iloc[:, :-1]
y = df['Hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f'X_Train Shape: {X_train.shape}')
print(f'X_Test Shape: {X_test.shape}')
print(f'y_Train Shape: {y_train.shape}')
print(f'y_Test Shape: {y_test.shape}')


# In[21]:


print(y_train.value_counts())
print(y_test.value_counts())


# ### Evaluation Functions
# 
# This step will create common function for creating evaluation metrics for all future models.

# In[22]:


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_probs) * 100
    
    metrics_table = f"""
    | Metric    | Score   |
    |-----------|---------|
    | Accuracy  | {accuracy:.2f}% |
    | Precision | {precision:.2f}% |
    | Recall    | {recall:.2f}% |
    | F1 Score  | {f1:.2f}% |
    | AUC       | {auc:.2f}% |
    """
    print(metrics_table)
    
    # Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred) 
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
       
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr
    }

def plot_roc_curves(model_evals):
    plt.figure(figsize=(10, 7))
    for eval in model_evals:
        plt.plot(eval['fpr'], eval['tpr'], label=f"{eval['model_name']} (AUC = {eval['auc']:.2f}%)")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()

def compare_models(model_evals):
    data = []

    for eval in model_evals:
        data.append([
            eval['model_name'],
            f"{eval['accuracy']:.2f}%",
            f"{eval['precision']:.2f}%",
            f"{eval['recall']:.2f}%",
            f"{eval['f1']:.2f}%",
            f"{eval['auc']:.2f}%"
        ])
    
    df = pd.DataFrame(data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])
    
    return df


# ## Modeling

# ### Logistics Regression
# Logistics Regression was chosen for it's simple and easy to implement nature. This also allowed easy hyperparameter tuning and balancing. With Balancing and tuning we reduced false negatives from 36 total misclassifications to only 7. This is especially important due to a False Negative could lead to catastrophic results.

# In[23]:


logr = LogisticRegression(max_iter = 1000)
logr.fit(X_train, y_train)
logr_eval = evaluate_model(logr, X_test, y_test, 'Logistics Regression')


# In[ ]:


logr = LogisticRegression(class_weight='balanced', max_iter = 10000)

# hyperparameter tuning
param_grid = {
    'C': np.logspace(-4, 4, 20), 
    'solver': ['liblinear', 'lbfgs'] 
}

grid_search = GridSearchCV(logr, param_grid, cv=5, scoring='recall', verbose=1)
grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
print(best_estimator)

logr_eval_tuned = evaluate_model(best_estimator, X_test, y_test, 'Logistic Regression Tuned')


# ### Random Forest
# Random Forest was chosen for it's robustness and ability to quickly and accurately classify. Interestingly, parameter tuning had little to no effect. 

# In[ ]:


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_eval = evaluate_model(rf_clf, X_test, y_test, 'Random Forest')


# In[ ]:


rf_clf = RandomForestClassifier(class_weight='balanced')

# hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3 , 4] 
}
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='recall', verbose=1) 


rf_clf.fit(X_train, y_train)
rf_eval_tuned = evaluate_model(rf_clf, X_test, y_test, 'Random Forest Tuned')


# In[ ]:


def plot_feature_importance_random_forest(model, feature_names):
    # Extract feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df = df.sort_values(by='Importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(df['Feature'], df['Importance'], color='forestgreen')
    plt.xlabel('Importance')
    plt.title('Feature Importance in Random Forest')
    plt.gca().invert_yaxis()
    plt.show()

# Usage
plot_feature_importance_random_forest(rf_clf, attribute_columns)


# ### XG Boost
# XG Boost was chosen for its superior accuracy in classification problems and due to its implementation of gradient boosting.

# In[ ]:


xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xg_eval = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')


# In[ ]:


class_counts = np.bincount(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

xgb_model = XGBClassifier(eval_metric='mlogloss', scale_pos_weight=scale_pos_weight)

# hyperparameter tuning
param_grid = {
    'max_depth': [3, 6, 10],         
    'learning_rate': [0.01, 0.1, 0.2], 
    'n_estimators': [100, 200, 300],  
    'subsample': [0.7, 0.9, 1.0],   
    'colsample_bytree': [0.7, 0.9, 1.0] 
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='recall', verbose=1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
xg_eval_tuned = evaluate_model(best_xgb, X_test, y_test, 'XGBoost Tuned')


# In[ ]:


def plot_feature_importance_xgboost(model, feature_names):
    # Extract feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df = df.sort_values(by='Importance', ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(df['Feature'], df['Importance'], color='darkred')
    plt.xlabel('Importance')
    plt.title('Feature Importance in XGBoost')
    plt.gca().invert_yaxis()
    plt.show()


plot_feature_importance_xgboost(best_xgb, attribute_columns)


# ## Evaluation
# I am prioritizing both AUC and Recall as the primary evaluation metrics. Recall is chosen specifically because I want to minimize False Negatives due to the nature of the dataset. From this we can see that Random Forest and XGBoost have the highest performance. Even without hyperparameter tuning these models performed exceptionally well. Logistics Regression took a small hit when balanced and tuned, but it shows massive improvements in Recall specifically. 

# In[ ]:


compare_models([logr_eval, logr_eval_tuned, rf_eval, rf_eval_tuned, xg_eval, xg_eval_tuned])


# In[ ]:


plot_roc_curves([logr_eval, logr_eval_tuned, rf_eval, rf_eval_tuned, xg_eval, xg_eval_tuned])


# ## Conclusion
# 
# When we started this project, our goal was to categorize asteroids based off simple features. This allows us to determine where we should focus our more expensive analysis. In the end we achieved a model with nearly 100% accuracy by utilizing the incredibly powerful Random Forest and XGBoost models. This was done through a few important steps:
# 1. Data Cleaning
#     - In this step we removed unnecessary features that would have skewed our results including collinearity and bad values.
# 2. Outliers Analysis
#     - This was another important step. It allowed us to check on what were the outliers in our dataset and gain a better overall understanding of the data. While we ultimately decided to leave the outliers this was highly important for the ML process.
# 3. ML Modeling
#     - In this phase we implemented Logistics Regression, XGBoost, and Random Forest models along with tuned variants of the model.
#     - The implementation of the tuning showed how powerful these models can be even without major tuning.
# 4. Evaluation
#     - Evaluation was done using Recall and AUC Curves as the primary form of evaluation. Recall was chosen to help minimize False Negatives with AUC curves helping with overall performance.
# 

# ### Future Improvements
# 
# This project can be improved in the future. Some area of improvements include further data gather and more in depth outlier analysis to see if removing outliers could help improve the model. Looking further into feature importance and potentially removing or adding features would also aid in improving the model.
