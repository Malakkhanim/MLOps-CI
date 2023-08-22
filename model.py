#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('weather.csv')

col_names = df.columns

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)



cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

df.drop('Date', axis=1, inplace = True)


def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)



IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)



IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)


IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)


# In[12]:


X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']


numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']


numerical = [col for col in df.columns if df[col].dtypes != 'O']

# Calculate information value
def calc_iv(df, feature, target, pr=0):

    lst = []

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()


    data = data.sort_values(by=['Variable', 'Value'], ascending=True)

    if pr == 1:
        print(data)

    return data


num_bins = 5  
binned_data = df.copy() 

for column in numerical:
    
    binned_data[column + '_bin'] = pd.qcut(binned_data[column], num_bins, labels=False, duplicates='drop')

binned_data.RainTomorrow = [0 if i=='No' else 1 for i in binned_data.RainTomorrow]

cols = [col for col in binned_data.columns if col.endswith('_bin')]


for colname in cols:
       try:
           iv = calc_iv(binned_data,feature=colname, target='RainTomorrow')['IV'][0]
           original_col_name = colname[:-4]
           X_train[original_col_name].fillna(iv,inplace=True)
           X_test[original_col_name].fillna(iv,inplace=True)
       except:
           X_train.drop(colname[:-4],axis=1,inplace=True)
           X_test.drop(colname[:-4],axis=1,inplace=True)


for temp in [X_train, X_test]:
    temp['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    temp['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    temp['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    temp['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)


def max_value(temp, variable, top):
    return np.where(temp[variable]>top, top, temp[variable])

for temp in [X_train, X_test]:
    temp['Rainfall'] = max_value(temp, 'Rainfall', 3.2)
    temp['WindSpeed9am'] = max_value(temp, 'WindSpeed9am', 55)
    temp['WindSpeed3pm'] = max_value(temp, 'WindSpeed3pm', 57)


train = pd.get_dummies(X_train,drop_first='True')

test = pd.get_dummies(X_test,drop_first='True')


cols = train.columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)

train = pd.DataFrame(train, columns=[cols])

test = pd.DataFrame(test, columns=[cols])

y_train = [0 if i=='No' else 1 for i in y_train]
y_test = [0 if i=='No' else 1 for i in y_test]

# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear', random_state=0)

logreg.fit(train, y_train)


y_pred_test = logreg.predict(test)



from sklearn.metrics import *
y_pred_test = logreg.predict(test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

with open('metrics.csv','w') as outfile:
    outfile.write('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))
