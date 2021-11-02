# -*- coding: utf-8 -*-
"""
Spyder Editor

Amritha Subburayan code for STOUT DDA FULL STACK CASE STUDIES

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import preprocessing
import sklearn.metrics as sm

data = pd.read_csv(r'//Users//amrithasubburayan//Downloads//loans_full_schema.csv')

data.info()

data.describe()

#Checking missing values

data.isna().sum()

#removing emp_title, state , num_accounts_120d_past_due , num_accounts_30d_past_due, tax_liens, public_record_bankrupt, 
# paid_late_fees , total_collection_amount_ever , current_accounts_delinq , num_historical_failed_to_pay
# num_collections_last_12m, delinq_2y

# check corr and remove this num_mort_accounts 
#storing data to other temp 
data2 = data

# DATA DESCRIPTION AND ISSUES :
#There are two issues in this dataset :
#1) Missing values 2) Multi-collinearity
#Missing values can be found in the following rows:
#1) emp_title 2) emp_length 3) annual_income_joint 4) verification_income_joint
# 5) debt_to_income_joint 6) months_since_last_delinq 7) months_since_90d_late 
#8) months_since_last_credit_inquiry 9) num_accounts_120d_past_due
#Multicollinearity can be found between these columns :
#1) installment and loan amount - 0.94 2) balance and loan amount - 0.93
# 3) annula income joint and total credit limit - 0.54 
#4) Inquires last 12 m and months since last credit inq - 0.51 
#5) total credit lines and open credit lines - 0.76 6) 
#num satisfactory acc and total credit lines - 0.75 
#7) total credit lines and num total cc accounts - 0.77 8) 
#total credit lines and num open cc accounts - 0.62



#Visualizations

plt.figure(figsize=(40,35))
sns.heatmap(data2.corr(), annot = True, cmap = "RdYlGn")

plt.show()


data2['loan_purpose'].value_counts().plot(kind='bar',color=['gray','red','blue','green','purple','yellow','black']).set_title('Loan Purpose')


data2.groupby('homeownership').verified_income.value_counts().unstack(0).plot.bar()



data2.groupby('homeownership').application_type.value_counts().unstack(0).plot(kind="pie",subplots=True, shadow = True,startangle=90,figsize=(15,10),autopct='%1.1f%%')



plt.scatter(data2['installment'],data2['loan_amount'])

d = data2.groupby('emp_length')
s=[]
for key,item in d:
    if(key!=7.0):
        s.append(d.get_group(key)['interest_rate'].mean())
dict1={"emp_length":[0,1,2,3,4,5,6,8,9,10],"int_rate":s}
plt.plot(dict1['emp_length'],s)

df= data2['application_type']

data2.groupby('application_type').loan_purpose.value_counts()

data2.groupby('application_type').loan_purpose.value_counts().unstack(0).plot(kind="pie",subplots=True, shadow = True,startangle=90,figsize=(25,20),autopct='%1.1f%%')




#Replacing missing rows

d = data2.groupby('application_type').loan_purpose.value_counts()

#data2["verification_income_joint"] = data2['verification_income_joint'].fillna('Not Verified')



for i in range(0, len(data2["verification_income_joint"])):

    if pd.isna(data2['verification_income_joint'][i]):
        
        data2['verification_income_joint'][i] = data2['verified_income'][i]
   
        
   
data2["debt_to_income"] = data2['debt_to_income'].fillna(0)

#combining annual income with joint annual income



for i in range(0, len(data2["annual_income_joint"])):

    if pd.isna(data2['annual_income_joint'][i]):
        
        data2['annual_income_joint'][i] = data2['annual_income'][i]
   
#combining debt income with joint debt income



for i in range(0, len(data2["debt_to_income_joint"])):

    if pd.isna(data2['debt_to_income_joint'][i]):
        
        data2['debt_to_income_joint'][i] = data2['debt_to_income'][i]


## Replacing with mean values


data2["months_since_last_credit_inquiry"] = data2['months_since_last_credit_inquiry'].fillna(np.mean(data2["months_since_last_credit_inquiry"]))

data2["emp_length"] = data2['emp_length'].fillna(np.mean(data2["emp_length"]))



#Removing unwanted columns because it has more 0 values which will not impact on building a model

data2.drop("emp_title", axis = 1, inplace=True)
data2.drop("state", axis = 1, inplace=True)
data2.drop("num_accounts_120d_past_due", axis = 1, inplace=True)
data2.drop("num_accounts_30d_past_due", axis = 1, inplace=True)
data2.drop("tax_liens", axis = 1, inplace=True)
data2.drop("public_record_bankrupt", axis = 1, inplace=True)
data2.drop("paid_late_fees", axis = 1, inplace=True)
data2.drop("total_collection_amount_ever", axis = 1, inplace=True)
data2.drop("current_accounts_delinq", axis = 1, inplace=True)
data2.drop("num_historical_failed_to_pay", axis = 1, inplace=True)
data2.drop("num_collections_last_12m", axis = 1, inplace=True)
data2.drop("delinq_2y", axis = 1, inplace=True)
data2.drop("verified_income", axis = 1, inplace=True)
data2.drop("annual_income", axis = 1, inplace=True)
data2.drop("debt_to_income", axis = 1, inplace=True)
data2.drop("months_since_90d_late", axis = 1, inplace=True)
data2.drop("months_since_last_delinq", axis = 1, inplace=True)
data2.drop("issue_month", axis = 1, inplace=True)
data2.drop("initial_listing_status", axis = 1, inplace=True)
data2.drop("disbursement_method", axis = 1, inplace=True)
data2.drop("grade", axis = 1, inplace=True)

#removing columns based on correlation

data2.drop("total_credit_limit", axis = 1, inplace=True)
data2.drop("current_installment_accounts", axis = 1, inplace=True)
data2.drop("accounts_opened_24m", axis = 1, inplace=True)
data2.drop("open_credit_lines", axis = 1, inplace=True)

data2.drop("loan_amount", axis = 1, inplace=True)
data2.drop("balance", axis = 1, inplace=True)
data2.drop("paid_principal", axis = 1, inplace=True)
data2.drop("num_satisfactory_accounts", axis = 1, inplace=True)
data2.drop("total_credit_lines", axis = 1, inplace=True)
data2.drop("num_active_debit_accounts", axis = 1, inplace=True)
data2.drop("num_open_cc_accounts", axis = 1, inplace=True)
data2.drop("installment", axis = 1, inplace=True)
data2.drop("num_total_cc_accounts", axis = 1, inplace=True)

#Removing Outliers based on its Quartile and Max Value

data5 = data2

sns.boxplot(data5['paid_interest'])
data5 = data5.loc[data5["inquiries_last_12m"] < 15]
data5 = data5.loc[data5["total_credit_utilized"] < 400000]
data5 = data5.loc[data5["months_since_last_credit_inquiry"] < 20]
data5 = data5.loc[data5["total_debit_limit"] < 220000]
data5 = data5.loc[data5["num_cc_carrying_balance"] < 20]
data5 = data5.loc[data5["num_mort_accounts"] < 10]
data5 = data5.loc[data5["paid_total"] < 35000]
data5 = data5.loc[data5["paid_interest"] < 3000]

# Encoding Categorical Data using LabelEncoder



le = preprocessing.LabelEncoder()

data5['sub_grade'] = le.fit_transform(data5['sub_grade'].values)
data5['verification_income_joint'] = le.fit_transform(data5['verification_income_joint'].values)
data5['loan_status'] = le.fit_transform(data5['loan_status'].values)
data5['loan_purpose'] = le.fit_transform(data5['loan_purpose'].values)
data5['application_type'] = le.fit_transform(data5['application_type'].values)
data5['homeownership'] = le.fit_transform(data5['homeownership'].values)

data5 = data5.reindex(columns=['emp_length', 'homeownership', 'annual_income_joint',
       'verification_income_joint', 'debt_to_income_joint',
       'earliest_credit_line', 'inquiries_last_12m', 'total_credit_utilized',
       'months_since_last_credit_inquiry', 'total_debit_limit',
       'num_cc_carrying_balance', 'num_mort_accounts',
       'account_never_delinq_percent', 'loan_purpose', 'application_type',
       'term', 'sub_grade', 'loan_status', 'paid_total',
       'paid_interest', 'interest_rate'])


X = data5.iloc[:, :-1].values
y = data5.iloc[:, -1].values

y = y.reshape(len(y),1)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#Modelling the Data

#Support Vector Regression

from sklearn.svm import SVR
regressor_SVM = SVR(kernel = 'rbf')
regressor_SVM.fit(X_train, y_train)

#For Training Data
SVR_train_pred = regressor_SVM.predict(X_train)

score2=r2_score(y_train,SVR_train_pred)
score2

print("Mean absolute error =", round(sm.mean_absolute_error(y_train, SVR_train_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_train, SVR_train_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_train, SVR_train_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_train, SVR_train_pred), 2)) 


#For Testing data

SVR_test_pred = regressor_SVM.predict(X_test)
score3=r2_score(y_test,SVR_test_pred)
score3
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, SVR_test_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, SVR_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, SVR_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, SVR_test_pred), 2)) 

#Random Forest Model


from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor1.fit(X_train, y_train)

#For Training Data

random_train_pred = regressor1.predict(X_train)
score1=r2_score(y_train,random_train_pred)
score1

print("Mean absolute error =", round(sm.mean_absolute_error(y_train, random_train_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_train, random_train_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_train, random_train_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_train, random_train_pred), 2)) 


#For Testing Data

random_test_pred = regressor1.predict(X_test)

score=r2_score(y_test,random_test_pred)
score

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, random_test_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, random_test_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, random_test_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, random_test_pred), 2))

















