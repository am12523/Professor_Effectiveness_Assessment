#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:58:30 2024

@author: ananya
"""
#Importing the necessary libraries

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( 
    accuracy_score, precision_score,
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import statsmodels.api as sm

#Seeding the RNG with my N-number
np.random.seed(15070423)

#Importing the datasets
num_df = pd.read_csv("rmpCapstoneNum.csv", header = None)
qual_df = pd.read_csv("rmpCapstoneQual.csv", header = None)

#Adding column headers
num_df.columns = ['avg_rating', 'avg_difficulty', 'num_ratings', 'pepper', 'retake_prop', 'online_ratings', 'male', 'female']
qual_df.columns = ['major', 'university', 'state']

#Original num rows = 89893
#Setting a threshold to accept data with more than 1 ratings as valid, so that results are not skewed by a single person
num_df = num_df[num_df['num_ratings'] >1]
#after using ratings with more than one rating, num rows = 52371
#i.e. we lost 41.7% of the data (large chunk, but since we have more than 50% of data that is more representative, we accept this)

#Imputing with mean, median, and mode as appropriate to deal with remaining NaN values
num_df['avg_rating'] = num_df['avg_rating'].fillna(num_df['avg_rating'].mean())
num_df['avg_difficulty'] = num_df['avg_difficulty'].fillna(num_df['avg_difficulty'].mean())
num_df['num_ratings'] = num_df['num_ratings'].fillna(num_df['num_ratings'].median())
num_df['pepper'] = num_df['pepper'].fillna(num_df['pepper'].mode())
num_df['retake_prop'] = num_df['retake_prop'].fillna(num_df['retake_prop'].median())
num_df['online_ratings'] = num_df['online_ratings'].fillna(num_df['online_ratings'].median())
num_df['male'] = num_df['male'].fillna(num_df['male'].mode())
num_df['female']= num_df['female'].fillna(num_df['female'].mode())


#QUESTION 1: Is there evidence of a pro-male gender bias in the dataset?
print("Question 1:")
male_profs = num_df[num_df['male'] == 1]
female_profs = num_df[num_df['female'] == 1]

#Choosing metrics to test for gender bias
metrics = [
    'avg_rating', 
    'avg_difficulty',  
    'retake_prop', 
    'pepper'
]

# Performing Welch's t-test
results = {}
for metric in metrics:
    t_statistic, p_value = stats.ttest_ind(
        male_profs[metric], 
        female_profs[metric], 
        equal_var=False
    )
    
    results[metric] = {
        't_statistic': t_statistic,
        'p_value': p_value,
    }

# Printing results
print("Gender Bias Results:")
for metric, stat in results.items():
    print(metric+":")
    print("T-Statistic:", stat['t_statistic'])
    print("P-Value:", stat['p_value'])
    
# Visualization - Boxplot
#creating a singular column for gender
num_df['gender'] = num_df.apply(lambda row: 'Male' if row['male'] == 1 else 'Female', axis=1)

plt.figure(figsize=(8, 6))
sns.boxplot(data=num_df, x="gender", y="avg_rating")
plt.title("Average Ratings by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Rating")
plt.show()
    
print()

#QUESTION 2: Is there an effect of experience on the quality of teaching?
print("Question 2:")

#defining predictor and and outcome variables
X = num_df['num_ratings'].values.reshape(-1, 1)
y = num_df['avg_rating'].values

# Fitting the regression model
model = LinearRegression().fit(X, y)

# Getting the regression line parameters
slope = model.coef_[0]
intercept = model.intercept_

# Predicting values
yHat = model.predict(X)

# R^2 value
rSq = model.score(X, y)

# Pearson correlation and p-value
correlation, p_value = stats.pearsonr(num_df['num_ratings'], num_df['avg_rating'])

# Visualization - scatterplot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(num_df['num_ratings'], num_df['avg_rating'], alpha=0.5)
plt.xlabel('Number of Ratings (Proxy for Experience)')
plt.ylabel('Average Rating (Teaching Quality)')
plt.title(f'Experience vs Teaching Quality\nR^2 = {rSq:.3f}, p-value = {p_value:.3e}')

# Adding regression line
plt.plot(num_df['num_ratings'], yHat, color='red')

plt.show()

#Printing results
print("Effect of experience on quality of teaching results")
print("r:", correlation)
print("r^2:", rSq)
print("p-value:", p_value)

print()

#QUESTION 3:What is the relationship between average rating and average difficulty
print("Question 3:")
    
correlation, p_value = stats.pearsonr(num_df['avg_rating'], num_df['avg_difficulty'])

#Printing results
print("Relationship between average rating and average difficulty")
print("Correlation (r):", correlation)
print("r^2:", correlation**2)
print("p-value:", p_value)

# Visualization - scatterplot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(num_df['avg_difficulty'], num_df['avg_rating'], alpha=0.5)
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating (Teaching Quality)')
plt.title(f'Average Rating vs Average Difficulty\nCorrelation (r) = {correlation:.3f}, p-value = {p_value:.3e}')

# Adding regression line
m, b = np.polyfit(num_df['avg_difficulty'], num_df['avg_rating'], 1)
plt.plot(num_df['avg_difficulty'], m*num_df['avg_difficulty'] + b, color='red')

plt.show()
print()

#QUESTION 4: Do professors who teach a lot of classes in the online modality receive higher or lower ratings than those who don’t?
print("Question 4:")

#Checking the basic distribution of online ratings
print("Distribution of online ratings:\n", num_df['online_ratings'].describe())
print()

# Creating binary categorization of no online ratings and online ratings (justified in the document)
num_df['online_teaching_category'] = num_df['online_ratings'].apply(lambda x: 'No Online Ratings' if x == 0 else 'Online Ratings')

# Splitting into no online and online ratings categories
no_online = num_df[num_df['online_teaching_category'] == 'No Online Ratings']
online = num_df[num_df['online_teaching_category'] == 'Online Ratings']

# Plotting histograms
plt.figure(figsize=(12, 6))
sns.histplot(no_online['online_ratings'], bins=30, alpha=0.5, label='No Online Ratings', color='blue')
sns.histplot(online['online_ratings'], bins=30, alpha=0.5, label='Online Ratings', color='orange')
plt.title('Distribution of Online Ratings')
plt.xlabel('Online Ratings')
plt.ylabel('Frequency')
plt.ylim(0, 5000)
plt.legend()
plt.show()

# Performing t-test for comparing means
t_statistic, p_value = stats.ttest_ind(no_online['avg_rating'], online['avg_rating'])

#Printing results
print("Effect of online teaching modality on average ratings")
print("T-Test Results:")
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")

#Visualization - boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='online_teaching_category', y='avg_rating', data=num_df)
plt.title('Average Rating by Online Teaching Category')
plt.xlabel('Online Teaching Category')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()
print()

#QUESTION 5 : What is the relationship between the average rating and the proportion of people who would take the class the professor teaches again?
print("Question 5:")

#Defining independent and dependent variables
ratings = num_df['avg_rating']
retake_prop = num_df['retake_prop']

# Calculate Pearson correlation coefficient
correlation, p_value = stats.pearsonr(ratings, retake_prop)

# Print correlation and p-value
print("Relationship between average rating and retake proportion")
print("Pearson Correlation Coefficient:", correlation)
print("rSq:",correlation**2 )
print("P-Value:", p_value)

# Visualization: Scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(ratings, retake_prop, alpha=0.5)
plt.xlabel('Average Rating')
plt.ylabel('Proportion of Students Who Would Take Again')
plt.title('Relationship Between Average Rating and Proportion of Students Willing to Take Again')

# Fitting a linear model
m, b = np.polyfit(ratings,retake_prop, 1)

# Plotting the regression line
plt.plot(ratings, m * ratings + b, color='red')

plt.show()

print()

#QUESTION 6: Do professors who are “hot” receive higher ratings than those who are not
print("Question 6:")

#Identifying the different categories - "hot" and "not hot"
hot_profs = num_df[num_df['pepper'] == 1]
not_hot_profs = num_df[num_df['pepper'] == 0]

# Performing t-test
t_statistic, p_value = stats.ttest_ind(hot_profs['avg_rating'], not_hot_profs['avg_rating'])

#: Plotting the results
plt.figure(figsize=(10, 6))
plt.boxplot([hot_profs['avg_rating'], not_hot_profs['avg_rating']], labels=['Hot Professors', 'Not Hot Professors'])
plt.title('Average Ratings by Hotness')
plt.ylabel('Average Rating')
plt.show()

# Printing results
print("Effect of 'hotness' of professor on average ratings")
print("T-Test Results:")
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")
print()

#QUESTION 7: Build a regression model predicting average rating from difficulty
print("Question 7:")

# Initializing the predicto and outcome
X = num_df['avg_difficulty'].values.reshape(-1, 1)
y = num_df['avg_rating'].values

# Splitting the data into training and testing sets
# Already seeded rng globally, so random state is not included here
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Creating and fitting the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared and RMSE
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Average Rating vs Difficulty')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.legend()
plt.show()

# Print model details
print("Linear Regression Results:")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient (Slope): {model.coef_[0]:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE: {rmse:.4f}")

# Residual analysis
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Average Rating')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

print()

#QUESTION 8: Build a regression model predicting average rating from all available factors
print("Question 8:")

# Prepare features (gender has been taken as a singular column - from Q1 - to avoid multicollinearity)
num_df['gender'] = num_df['gender'].map({'Male': 0, 'Female': 1}) 
features = ['avg_difficulty', 'num_ratings', 'pepper', 'retake_prop', 'online_ratings', 'gender']
X = num_df[features]
y = num_df['avg_rating']
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, center=0)
plt.title('Correlation Heatmap of Predictors')
plt.show()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared and RMSE
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print(f"R-squared: {r_squared:.4f}")
print(f"RMSE: {rmse:.4f}")

# Residual analysis
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Average Rating')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Step 5: Analyze Individual Betas
coefficients = pd.DataFrame({'Predictor': X.columns, 'Beta': model.coef_})
print("\nRegression Coefficients:")
print(coefficients.sort_values(by='Beta', key=abs, ascending=False))

print()
  
  
# QUESTION 9: Build a classification model that predicts whether a professor receives a “pepper” from average rating only.
print("Question 9:")


# Prepare data
X_rating = num_df['avg_rating'].values.reshape(-1, 1)
y = num_df['pepper'].values

# Split the data
X_rating_train, X_rating_test, y_train, y_test = train_test_split(
    X_rating, y, test_size=0.3, stratify=y)

# Logistic Regression with Average Rating
lr_rating = LogisticRegression(
    class_weight='balanced')
lr_rating.fit(X_rating_train, y_train)

# Predictions and probabilities
y_pred_rating = lr_rating.predict(X_rating_test)
y_prob_rating = lr_rating.predict_proba(X_rating_test)[:, 1]

# Calculate AUROC for Rating-Only Model
accuracy = accuracy_score(y_test, y_pred_rating)
precision = precision_score(y_test, y_pred_rating)
auroc_rating = roc_auc_score(y_test, y_prob_rating)

# Confusion Matrix and Classification Report for Rating-Only Model
print("\nRating-Only Model Performance:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rating))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rating))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"AUROC: {auroc_rating:.4f}")

# ROC Curve for Rating-Only Model
plt.figure(figsize=(8, 6))
fpr_rating, tpr_rating, _ = roc_curve(y_test, y_prob_rating)
plt.plot(fpr_rating, tpr_rating, label=f'Rating-Only Model (AUROC = {auroc_rating:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.title('ROC Curve - Rating-Only Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
print()

#Quesiton 10: Build a classification model that predicts whether a professor receives a “pepper” from all available factors.

print("Quesiton 10:")

# Prepare features
features = ['avg_rating', 'avg_difficulty', 'num_ratings', 'retake_prop', 'online_ratings', 'gender']
X_all = num_df[features]

# Standardize features
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Split the data
X_all_train, X_all_test, y_train, y_test = train_test_split(
    X_all_scaled, y, test_size=0.3, stratify=y)

# Model 2: Logistic Regression with All Features
lr_all = LogisticRegression(
    class_weight='balanced')
lr_all.fit(X_all_train, y_train)

# Predictions and probabilities
y_pred_all = lr_all.predict(X_all_test)
y_prob_all = lr_all.predict_proba(X_all_test)[:, 1]

# Calculate AUROC for All-Features Model
auroc_all = roc_auc_score(y_test, y_prob_all)

# Confusion Matrix and Classification Report for All-Features Model
print("\nAll-Features Model Performance:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_all))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_all))
print(f"AUROC: {auroc_all:.4f}")

# ROC Curve for All-Features Model
plt.figure(figsize=(8, 6))
fpr_all, tpr_all, _ = roc_curve(y_test, y_prob_all)
plt.plot(fpr_rating, tpr_rating, label=f'Rating-Only Model (AUROC = {auroc_rating:.4f})')
plt.plot(fpr_all, tpr_all, label=f'All-Features Model (AUROC = {auroc_all:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.title('ROC Curve - Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Feature importance for All-Features Model
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(lr_all.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
print()

#EXTRA CREDIT: "Is there a significant difference in average ratings across majors?"

print("Extra Credit:")
combined_df = pd.concat([num_df, qual_df], axis=1)

# Grouping by major and calculate mean ratings
major_ratings = combined_df.groupby('major')['avg_rating'].agg(['mean', 'count']).reset_index()

# Filtering out majors with very few ratings
major_ratings_filtered = major_ratings[major_ratings['count'] >= 10]

# Preparing data for ANOVA
major_groups = [
    combined_df[combined_df['major'] == major]['avg_rating'].dropna() 
    for major in major_ratings_filtered['major']
]

# Removing empty groups
major_groups = [group for group in major_groups if len(group) > 0]

# Performing one-way ANOVA
f_statistic, p_value = stats.f_oneway(*major_groups)

# Calculating effect size (Eta-squared)
def calculate_eta_squared(f_stat, num_groups, total_observations):
    return (f_stat * (num_groups - 1)) / (f_stat * (num_groups - 1) + (total_observations - num_groups))

total_observations = sum(len(group) for group in major_groups)
eta_squared = calculate_eta_squared(f_statistic, len(major_groups), total_observations)

# Printing results
print("One-way ANOVA Results:")
print("F-statistic:", f_statistic)
print("p-value:", p_value, format(p_value, '.2e'))
print("Eta-squared (effect size):", format(eta_squared, '.4f'))

# Boxplot of average ratings by major
plt.figure(figsize=(12, 8))
sns.boxplot(
    data=combined_df[combined_df['major'].isin(major_ratings_filtered['major'])],
    x='major', 
    y='avg_rating', 
    palette='Set2'
)
plt.title('Distribution of Average Ratings Across Majors', fontsize=16)
plt.xlabel('Major', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
