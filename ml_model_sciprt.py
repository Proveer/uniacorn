import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('train.csv')
# handling null values for age
df['Age'].fillna(df['Age'].median(),inplace=True)
# fill empty value for Embarked with most occuring value
df['Embarked'].fillna('S',inplace=True)

#Our job is to apply feature engineering so we can extract 
#the Miss, Mrs, Dr which provide a meaning to our ml model
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'No Title'
    
titles = set([x for x in df.Name.map(lambda x : get_title(x))])

# Still many titles lets combine them to create shorter titles
def shorter_titles(title):
    #title = x['Title']
    if title in ['Capt','Col','Major']:
        return "Officer"
    elif title in ['Don','Jonkheer','Lady','Sir','the Countess']:
        return "Royalty"
    elif title == "Mne":
        return "Mrs"
    elif title in ['Mlle','Ms']:
        return 'Miss'
    else:
        return title
    
    
df["Title"] = df['Name'].map(lambda x:get_title(x))
df["Shorter Title"] = df['Title'].map(lambda x:shorter_titles(x))
df.drop('Name',axis=1,inplace=True)
df.drop('Title',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('Cabin',axis=1,inplace=True)

df.Sex.replace(('male','female'),(0,1),inplace=True)
df.Embarked.replace(('S','C','Q'),(0,1,2),inplace=True)

lt=[]
for i in df['Shorter Title']:
    lt.append(i)
k = list(set(lt))
df['Shorter Title'].replace(k,(0,1,2,3,4,5,6,7,8),inplace=True)
del df['PassengerId']


from sklearn.model_selection import train_test_split
x = df.drop(['Survived'],axis = 1)
y = df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)
acc_rf = round(accuracy_score(y_pred,y_test)*100,2)
print(acc_rf)
pickle.dump(rf,open('titanic_model.sav','wb'))



