import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('D:/911.csv/911.csv')

# info about the dataset
print(df.info())

# Top 5 rows of the dataset
print(df.head())

# Descriptive statistics
print(df.describe())

# Check for null values
print(df.isnull().sum())

# remove null values
df = df.dropna()

# top 5 zip codes for 911 calls
top_zip_codes = df['zip'].value_counts().head(5)
print(top_zip_codes)

# top 5 townships for 911 calls
top_townships = df['twp'].value_counts().head(5)
print(top_townships)

# unique reasons for 911 calls
reason = df['title'].apply(lambda x: x.split(':')[0]).value_counts()
print(reason)

df['reason'] = df['title'].apply(lambda x: x.split(':')[0])

# countplot of reasons for 911 calls
sns.countplot(x='reason', data=df)
plt.title('Count of 911 Calls by Reason')
plt.show()

# Time series analysis of 911 calls
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].dt.hour
df['Month'] = df['timeStamp'].dt.month
df['Day of Week'] = df['timeStamp'].dt.day_name()

# Count of 911 calls by day of week
sns.countplot(x='Day of Week',hue='reason', data=df)
plt.title('Count of 911 Calls by Day of Week')
plt.show()

# Count of 911 calls by month
sns.countplot(x='Month', hue='reason', data=df)
plt.title('Count of 911 Calls by Month')
plt.show()

# top 5 month for 911 calls
top_months = df['Month'].value_counts().head(5)
print(top_months)

# Count of 911 calls per month
monthly_calls = df.groupby('Month').count()['reason']
plt.figure(figsize=(10, 6))
monthly_calls.plot()
plt.title('Monthly 911 Calls')
plt.xlabel('Month')
plt.ylabel('Number of Calls')
plt.show()

# Count of 911 calls by date
df['Date'] = df['timeStamp'].dt.date
daily_calls = df.groupby('Date').count()['reason']
plt.figure(figsize=(10, 6))
daily_calls.plot()
plt.title('Daily 911 Calls')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.show()

# Count of 911 calls by date for each reason
daily_calls_reason = df.groupby(['Date', 'reason']).count()['title'].unstack()
plt.figure(figsize=(12, 6))
daily_calls_reason.plot()
plt.title('Daily 911 Calls by Reason')
plt.xlabel('Date')
plt.ylabel('Number of Calls')
plt.legend(title='Reason')
plt.show()

# Heatmap of 911 calls by hour and day of week
heatmap_data = df.groupby(['Day of Week', 'Hour']).count()['reason']
heatmap_data = heatmap_data.unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='viridis')
plt.title('Heatmap of 911 Calls by Hour and Day of Week')
plt.xlabel('Hour')
plt.ylabel('Day of Week')
plt.show()

# Heatmap of 911 calls by month and day of week
heatmap_data_month = df.groupby(['Day of Week', 'Month']).count()['reason']
heatmap_data_month = heatmap_data_month.unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_month, cmap='coolwarm')
plt.title('Heatmap of 911 Calls by Month and Day of Week')
plt.xlabel('Month')
plt.ylabel('Day of Week')
plt.show()

# Correlation matrix of numerical features
correlation_matrix = df[['lat', 'lng', 'zip', 'Hour', 'Month']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Pairplot of 911 calls by reason
sns.pairplot(df, hue='reason', vars=['Hour', 'Month'])
plt.title('Pairplot of 911 Calls by Reason')
plt.show()


#---x-----------------------------------x---------------------------------------x------------------------------------------x------------------------------------------x---

# Machine Learning Model to Predict 911 Call Reasons

#---x-----------------------------------x---------------------------------------x------------------------------------------x------------------------------------------x---
 

# regression line graph
plt.figure(figsize=(10, 6))
sns.regplot(x='Hour', y='Month', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Regression Line of Hour vs Month')
plt.show()

# T-test the model with features and target label
from scipy.stats import ttest_ind

# Perform t-test for each reason
medical_calls = df[df['reason'] == 'EMS']['Hour']
traffic_calls = df[df['reason'] == 'Traffic']['Hour']
fire_calls = df[df['reason'] == 'Fire']['Hour']
t_stat_med_traffic, p_value_med_traffic = ttest_ind(medical_calls, traffic_calls)
t_stat_med_fire, p_value_med_fire = ttest_ind(medical_calls, fire_calls)
t_stat_traffic_fire, p_value_traffic_fire = ttest_ind(traffic_calls, fire_calls)
print(f"T-test between Medical and Traffic calls: t-statistic = {t_stat_med_traffic}, p-value = {p_value_med_traffic}")
print(f"T-test between Medical and Fire calls: t-statistic = {t_stat_med_fire}, p-value = {p_value_med_fire}")
print(f"T-test between Traffic and Fire calls: t-statistic = {t_stat_traffic_fire}, p-value = {p_value_traffic_fire}")

# F-test the model with features and target label
from scipy.stats import f_oneway
# Perform F-test for each reason
f_stat_med_traffic, p_value_med_traffic_f = f_oneway(medical_calls, traffic_calls)
f_stat_med_fire, p_value_med_fire_f = f_oneway(medical_calls, fire_calls)
f_stat_traffic_fire, p_value_traffic_fire_f = f_oneway(traffic_calls, fire_calls)
print(f"F-test between Medical and Traffic calls: F-statistic = {f_stat_med_traffic}, p-value = {p_value_med_traffic_f}")
print(f"F-test between Medical and Fire calls: F-statistic = {f_stat_med_fire}, p-value = {p_value_med_fire_f}")

# chi-squared test for categorical features
from scipy.stats import chi2_contingency
# Create a contingency table for reason and day of week
contingency_table = pd.crosstab(df['reason'], df['Day of Week'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared test: chi2 = {chi2}, p-value = {p}, degrees of freedom = {dof}")




# Select features
X = df[['lat', 'lng', 'zip', 'Hour', 'Month']]

# Target label
y = df['reason']

# Encode the target variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Medical: 0, Traffic: 1, Fire: 2

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)

# Evaluate the models
from sklearn.metrics import classification_report, accuracy_score

models = {'Logistic Regression': lr, 'Random Forest': rf, 'SVC': svc}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("-" * 60)


