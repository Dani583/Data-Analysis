#!/usr/bin/env python
# coding: utf-8

# # Imporing Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import roc_curve


# # Impoting the DataSet

# In[3]:


df = pd.read_csv("climber_df.csv")


# In[4]:


pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 11000)
pd.set_option('display.width', 11000)


# # Performing EDA On DataSets

# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


df.describe()


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df["country"].value_counts()


# In[11]:


df["sex"].value_counts()


# In[12]:


df["height"].value_counts()


# In[13]:


df["Year_Active"] = df["year_last"]- df["year_first"]


# In[14]:


df["Good_meanscore"] =df.grades_mean > 40 


# In[15]:


df["Good_meanscore"] = df["Good_meanscore"].astype(int)


# In[16]:


df.head()


# In[17]:


df["Year_Active"].value_counts()


# In[18]:


df['Year_Active'] = df['Year_Active'].abs()


# In[19]:


mio = math.floor(df['Year_Active'].mean())
mio


# In[20]:


df = df[df.Year_Active < 30]


# In[21]:


df["Good_meanscore"].value_counts()


# In[22]:


df.shape


# In[23]:


df['age'] =  df['age'].astype('int64')
df['grades_mean'] =  df['grades_mean'].astype('int64')


# In[24]:


df['age']


# In[25]:


df['grades_max']


# In[26]:


df["Year_Active"].value_counts()


# In[27]:


df["age"].value_counts()


# In[28]:


df["grades_max"].value_counts()


# In[29]:


df["height"].value_counts()


# In[32]:


tar = df['country'].value_counts()


# In[33]:


plt.figure(figsize=(8, 8))
plt.pie(tar.values, labels=tar.index, autopct = '%0.0f%%')
plt.title('Country proportion')
plt.show()


# In[37]:


fig_dims = (20, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='country',data=df)
plt.xticks(size=10);
plt.show()


# In[38]:


fig_dims = (20, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x='country', hue='grades_max',data=df)
plt.xticks(size=10);
plt.title("Country Grouped by the grades max score")
plt.show()


# In[31]:


plt.figure(figsize = (10,10))
plt.scatter(df['age'],df['grades_max'])
plt.show()


# In[32]:


plt.figure(figsize = (10,10))
plt.scatter(df['height'],df['grades_max'])
plt.show()


# In[33]:


plt.figure(figsize = (10,10))
plt.scatter(df['Year_Active'],df['grades_max'])
plt.show()


# # Preparing TesT, Train Data

# In[39]:


df_data = df[['sex','height','weight','age','Year_Active']]

df_target = df[['Good_meanscore']]


# In[40]:


df_data.head()


# In[41]:


df_target.head()


# In[42]:


X_train,X_test,Y_train,Y_test = train_test_split(df_data, df_target, train_size=0.8)


# In[43]:


X_train


# # Appling Algorithms On Datasets

# # LinearRegression

# In[44]:


from sklearn import linear_model

lin= linear_model.LinearRegression()
lin.fit(X_train, Y_train)


# In[45]:


lin_reg = lin.predict(X_test)

lin_reg


# In[46]:


lin.coef_


# In[47]:


lin.intercept_


# In[48]:


print("mean_absolute_error of Linear_Regression: ",  mean_absolute_error(Y_test , lin_reg))
print("mean_squared_error of Linear_Regression: ",  mean_squared_error(Y_test , lin_reg))
print("Score of Linear_Regression: ",  lin.score(X_test,Y_test))
print("Sqaureroot of Mean_squared_error of Linear_Regression: ",  np.sqrt(mean_squared_error(Y_test , lin_reg)))


# In[49]:


plt.figure(figsize=(10,10))
plt.scatter(Y_test,lin_reg, marker ='.', color = 'Green')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show


# # DecisionTreeClassifier

# In[50]:


DT= tree.DecisionTreeClassifier(max_depth=3, random_state=42)


# In[51]:


DT.fit(X_train, Y_train)


# In[52]:


DT_reg = DT.predict(X_test)

DT_reg


# In[53]:


DT.tree_


# In[54]:


print("mean_absolute_error of DecisionTreeClassifier: ",  mean_absolute_error(Y_test , DT_reg))
print("mean_squared_error of DecisionTreeClassifier: ",  mean_squared_error(Y_test , DT_reg))
print("Score of DecisionTreeClassifier: ",  DT.score(X_test,Y_test))
print("Sqaureroot of Mean_squared_error of DecisionTreeClassifier: ",  np.sqrt(mean_squared_error(Y_test , DT_reg)))


# In[55]:


plt.figure(figsize= (20,12))
tree.plot_tree(DT)


# # KNeighborsClassifier

# In[56]:


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, Y_train)


# In[57]:


knn.ped = knn.predict(X_test)
knn.ped


# In[58]:


print("mean_absolute_error of KNeighborsClassifier: ",  mean_absolute_error(Y_test , knn.ped))
print("mean_squared_error of KNeighborsClassifier: ",  mean_squared_error(Y_test , knn.ped))
print("Score of KNeighborsClassifier: ",  knn.score(X_test,Y_test))
print("Sqaureroot of Mean_squared_error of KNeighborsClassifier: ",  np.sqrt(mean_squared_error(Y_test , knn.ped)))


# In[59]:


accuracy_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_data,df['grades_max'],cv=10)
    accuracy_rate.append(score.mean())


# In[60]:


plt.figure(figsize=(20,10))
plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('KNN')
plt.ylabel('Error Rate')


# # Support Vector Machine

# In[61]:


svm = SVC(kernel='linear',C=1)

svm.fit(X_train, Y_train)


# In[62]:


y= svm.predict(X_test)

y


# In[63]:


print("mean_absolute_error of SupportVectorMachine: ",  mean_absolute_error(Y_test ,y))
print("mean_squared_error of SupportVectorMachine: ",  mean_squared_error(Y_test , y))
print("Score of SupportVectorMachine: ",  svm.score(X_test,Y_test))
print("Sqaureroot of Mean_squared_error of SupportVectorMachine: ",  np.sqrt(mean_squared_error(Y_test , y)))


# In[64]:


svm.coef_


# In[65]:


plt.figure(figsize= (10,10))

xfit = np.linspace(50, 80)
plt.scatter(X_test.iloc[:,2], X_test.iloc[:,3], c=y, s=30, cmap='autumn')
plt.plot([65], [45], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(0.5, 1), (0.6, 1.5), (0.7, 2)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.show()


# # Naive Bayes

# In[66]:


nb = GaussianNB()
nb.fit(X_train, Y_train)

nbpred = pd.DataFrame(nb.predict(X_test))


# In[67]:


print("mean_absolute_error of Naive Bayes: ",  mean_absolute_error(Y_test ,nbpred))
print("mean_squared_error of Naive Bayes: ",  mean_squared_error(Y_test , nbpred))
print("Score of Naive Bayes: ",  nb.score(X_test,Y_test))
print("Sqaureroot of Mean_squared_error of Naive Bayes: ",  np.sqrt(mean_squared_error(Y_test , nbpred)))


# In[ ]:




