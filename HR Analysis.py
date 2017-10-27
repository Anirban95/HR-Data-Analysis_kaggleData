
# coding: utf-8

# <h1> Human Resource Analysis

# In[91]:

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.tree import DecisionTreeClassifier as dtclf 
from  sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from sklearn.ensemble import GradientBoostingClassifier as GrdBoost
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 
from sklearn.linear_model import LogisticRegression as Logistic


# In[92]:

path = "HR_comma_sep.csv"
data=pd.read_csv(path)
data.columns


# In[93]:

dept = {'sales' : 1 , 'marketing': 2,'technical' : 3 , 'support': 4,'product_mng' : 5 , 'IT': 6,'hr' : 7 , 'management': 8,'accounting' : 9,'RandD':10 }
salary = {'low':0,'medium':1,'high':2}
df=data.replace({'sales':dept})
df=df.replace({'salary':salary})
data = df


# In[94]:

#Data Describe
data_Info = df.info()
df.head(5)


# <h2>#Distribution of Independent Variables

# In[101]:


plt.figure(figsize=(9, 8))
sns.distplot(df['satisfaction_level'], color='r', bins=10, hist_kws={'alpha': 0.4});
print(df['satisfaction_level'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['last_evaluation'], color='b', bins=10, hist_kws={'alpha': 0.4});
print(df['last_evaluation'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['time_spend_company'], color='y', bins=10, hist_kws={'alpha': 0.4});
print(df['time_spend_company'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['number_project'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['number_project'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['salary'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['salary'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['promotion_last_5years'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['promotion_last_5years'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['average_montly_hours'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['average_montly_hours'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['Work_accident'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['Work_accident'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['sales'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['sales'].describe())

plt.figure(figsize=(9, 8))
sns.distplot(df['left'], color='c', bins=10, hist_kws={'alpha': 0.4});
print(df['left'].describe())


# In[100]:

df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# <h2> Check the Multi-Coliniarity

# In[23]:

g = sns.pairplot(data)
g.savefig("Mult_colinear.png")


# In[5]:

X = data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'sales', 'salary']]
Y = data[['left']]


# <h2># feature extraction</h2>
# <h3>  SelectKBest </h3>

# In[102]:

# feature extraction
test = SelectKBest(score_func=chi2, k=9)
fit = test.fit(X,Y)
np.set_printoptions(precision=3)
print(fit.scores_)
feature_vect = fit.transform(X)
feature_vect


# In[103]:

X_train, X_test, y_train, y_test = train_test_split(feature_vect,Y, test_size=0.40, random_state=42)


# <h2> Decision Tree

# In[9]:

model = dtclf()
model.fit(X_train , y_train)


# In[10]:

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[11]:

accuracy_score(y_test, y_pred)


# In[12]:

f1_score(y_test, y_pred,average='macro')


# <h2> <b> Ada-Boost with decision Tree as base-estimator ( An ensemble Approach)

# In[13]:

Model = adaBoost(n_estimators=100,base_estimator=dtclf(),learning_rate=0.98)
Model.fit(X_train , y_train)
y_pred = model.predict(X_test)


# In[14]:

confusion_matrix(y_test, y_pred)


# In[15]:

accuracy_score(y_test, y_pred)


# In[16]:

f1_score(y_test, y_pred,average='macro')


# <h2> Logistic Regression

# In[17]:

Model = Logistic(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
                 solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
Model.fit(X_train , y_train)
y_pred = model.predict(X_test)


# In[18]:

confusion_matrix(y_test, y_pred)


# In[19]:

accuracy_score(y_test, y_pred)


# In[20]:

f1_score(y_test, y_pred,average='macro')

