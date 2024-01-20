#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno


# In[2]:


titanic_data = pd.read_csv('titanic.csv')


# In[3]:


titanic_data


# In[4]:


titanic_data.info()


# In[5]:


titanic_data.isnull().sum()


# In[6]:


msno.bar(titanic_data)


# In[7]:


msno.matrix(titanic_data)


# In[8]:


titanic_data['Survived'].value_counts()


# In[9]:


sns.countplot(data=titanic_data,x='Survived')


# In[10]:


sns.countplot(data=titanic_data,x='Survived',hue='Gender')


# In[11]:


titanic_data['Pclass'].value_counts()


# In[12]:


titanic_data['Age'].value_counts()


# In[13]:


sns.boxplot(data=titanic_data,x='Pclass',y="Age")


# In[14]:


def fill_age(row):
    pclass = row[0]
    age = row[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 28
        elif pclass==3:
            return 21
    else:
        return age
    


# In[15]:


def add_10(num):
    print(num)


# In[16]:


add_10(22)


# In[17]:


titanic_data[['Pclass','Age']].apply(add_10,axis=1)


# In[18]:


titanic_data['Age']=titanic_data[['Pclass','Age']].apply(fill_age,axis=1)


# In[19]:


titanic_data['Age']


# In[20]:


from sklearn.preprocessing import LabelEncoder
label_enco = LabelEncoder()


# In[21]:


label_enco.fit(titanic_data['Gender'])


# In[22]:


label_enco.transform(titanic_data['Gender'])


# In[23]:


titanic_data['Gender'] = label_enco.transform(titanic_data['Gender'])


# In[24]:


titanic_data


# In[25]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# In[26]:


titanic_data.head()


# In[27]:


titanic_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[28]:


titanic_data.head()


# In[29]:


titanic_data.dropna(inplace=True)


# In[30]:


titanic_data.reset_index(drop=True,inplace=True)


# In[31]:


from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()


# In[32]:


one_hot.fit(titanic_data[['Embarked']])


# In[33]:


temp_arr = one_hot.transform(titanic_data[['Embarked']]).toarray()


# In[34]:


temp_arr


# In[35]:


temp_data = pd.DataFrame(temp_arr,columns=['C','Q','S'],dtype='int')


# In[36]:


temp_data


# In[37]:


titanic_data = pd.concat([titanic_data,temp_data],axis=1)


# In[38]:


titanic_data.drop('Embarked',axis=1,inplace=True)


# In[39]:


titanic_data.head()


# In[40]:


X = titanic_data.drop('Survived',axis=1)
Y = titanic_data['Survived']


# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=101)


# In[42]:


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()


# In[43]:


logistic.fit(X_train,Y_train)


# In[44]:


Y_pred = logistic.predict(X_test)


# In[45]:


Y_test[9]


# In[47]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(Y_pred,Y_test))


# In[48]:


print(classification_report(Y_pred,Y_test))


# In[49]:


sns.heatmap(confusion_matrix(Y_pred,Y_test),cmap='viridis',annot=True)


# In[50]:


from sklearn.metrics import accuracy_score,classification_report


# In[52]:


accuracy = accuracy_score(Y_test,Y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(Y_test, Y_pred))


# In[53]:


from sklearn.ensemble import RandomForestClassifier


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)


# In[55]:


rf = RandomForestClassifier()
rf.fit(X_train, Y_train)


# In[56]:


Y_pred = rf.predict(X_test)


# In[57]:


Y_pred


# In[58]:


accuracy = accuracy_score(Y_test, Y_pred)
print(f'Model Accuracy: {accuracy}')


# In[59]:


print(confusion_matrix(Y_pred,Y_test))


# In[60]:


print(classification_report(Y_pred,Y_test))


# In[61]:


sns.heatmap(confusion_matrix(Y_pred,Y_test),cmap='plasma',annot=True)


# In[62]:


Ypred_train_rf = rf.predict(X_train)
Ypred_test_rf = rf.predict(X_test)


# In[63]:


a=pd.DataFrame({'ACTUAL':Y_train, 'PREDICTED': Ypred_train_rf})


# In[64]:


b=pd.DataFrame({'ACTUAL':Y_test, 'PREDICTED': Ypred_test_rf})


# In[65]:


c=pd.concat([a,b])
c


# In[66]:


final_titanic = titanic_data.join(c)


# In[67]:


final_titanic


# In[68]:


def prediction(row):
    if row['ACTUAL'] == row['PREDICTED']:
        return 'correct prediction'
    else:
        return 'incorrect prediction'

final_titanic['prediction'] = final_titanic.apply(prediction, axis=1)


# In[69]:


final_titanic['prediction'].value_counts()


# In[73]:


Predict = final_titanic['prediction'].value_counts()
plt.figure(figsize=(6,8))
plt.pie(Predict, autopct='%1.1f%%',
        startangle=140,colors=['skyblue','yellow'])
plt.legend(Predict.index)
plt.title('Pie Chart for predictions')
plt.show()


# In[ ]:




