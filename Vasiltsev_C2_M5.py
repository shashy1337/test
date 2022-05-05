#!/usr/bin/env python
# coding: utf-8

# # Модуль 2
# ---

# Начнем с подключения библиотек

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# Далее загрузим данные из предыдущей сессии

# In[2]:


df = pd.read_csv("c1_result.csv")


# In[3]:


df.head()


# In[7]:


df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


# # Конструирование признаков(Feature Engineering)

# In[8]:


df["year"] = df["pickup_datetime"].apply(lambda x : x.year)
df["month"] = df["pickup_datetime"].apply(lambda x : x.month) 
df["day"] = df["pickup_datetime"].apply(lambda x : x.day)
df["dayofweek"] = df["pickup_datetime"].apply(lambda x : x.dayofweek)
df["hour"] = df["pickup_datetime"].apply(lambda x : x.hour)


# In[9]:


df = df.drop("pickup_datetime", axis = 1)


# In[10]:


df


# # Визуализация данных

# In[12]:


plt.figure(figsize = (20, 10))
sns.heatmap(df.corr(), annot = True)


# ### Графики зависимостей аатрибутов на целевую переменную

# In[13]:


import numpy as np
plt.figure(figsize = (10, 10))
sns.pointplot(y = np.sort(df["trip_duration"]), x = np.sort(df["passenger_count"]))


# In[14]:


import numpy as np
plt.figure(figsize = (10, 10))
sns.pointplot(y = np.sort(df["trip_duration"]), x = np.sort(df["maximum temperature"]))


# In[15]:


for pr in ["month", "day", "dayofweek", "hour"]:
    sns.scatterplot(y = np.sort(df["trip_duration"]), x = np.sort(df[pr]))
    plt.title(pr)
    plt.show()


# ## Разбиение набора данных

# In[16]:


df.head()


# In[17]:


X =  df.drop("trip_duration", axis = 1)
y = df["trip_duration"].array


# In[18]:


y


# Разбиение данных на тестовую и обучающую выборку

# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# 1. Kneigbours
# 2. RandomForest
# 3. GradientBoosting

# In[20]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[21]:


def score(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print("R^2: " + str(r2))
    print("-" * 100)
    print("MAE: " + str(mae))
    print("-" * 100)


# #### 1. RandomForest

# In[22]:


rfc = RandomForestRegressor(n_estimators=150, n_jobs=-1, verbose=3)
rfc.fit(X_train, y_train)
score(y_test, rfc.predict(X_test))


# #### 2. GradientBostingRegressor

# In[23]:


grb = GradientBoostingRegressor(verbose=3)
grb.fit(X_train, y_train)
score(y_test, grb.predict(X_test))


# ### 3. KNeighbours

# In[ ]:


kn = KNeighborsRegressor(n_neighbors=3,n_jobs=-1)
kn.fit(X_train, y_train)
score(y_test, kn.predict(X_test))


# ## Вывод

# - Был произведен Feature Engineering
# - Были провизуализированы признаки которые наибольше всего влияют на целевую переменную
# - Была проведена регрессия исходных признаков
# - Отчет подготовлен
