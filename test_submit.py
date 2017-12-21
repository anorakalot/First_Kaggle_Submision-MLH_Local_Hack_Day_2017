# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#usese pandas read_csv to create initial dataframes
train_df = pd.read_csv('~/Documents/machine_learning/titanic_script/train.csv')
#train_df = pd.read_csv('home/anorak/Documents/mlh_local_hack_day/train.csv')
test_df = pd.read_csv('~/Documents/machine_learning/titanic_script/test.csv')
#use this to create change datasets in both train_df and test_df concurrently
combined_data = [train_df,test_df]

#print(train_df.head(20))

'''
#USE THIS TO SEE WHICH FEATURES ACTUALLY MATTER
#group the data and analyze the means of survival ratio
#try to see which features affected Survival the most
print(train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False))
print(train_df[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by = 'Survived', ascending=False))
print(train_df[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by = 'Survived', ascending = False))
print(train_df[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by = 'Survived', ascending = False))
'''

#Feature Engineering Stuff
#now create new features in order to have more impactful features to run the ML algorithms through
#TRYING TO CREATE IMPACTFUL FEATURES
#TRYING TO TRANSFORM FEATURES TO NAMES
#GET RID OF FEATURES THAT DONT MATTER


#one for loop for all feature changes


#create FamilySize feature
for dataset in combined_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']
#print (train_df[['FamilySize','Survived']].groupby(['FamilySize']).mean())


#create IsAlone Feature
#this is a good feature because it shows it has a big impact on any individual's survival
for dataset in combined_data:
    #default is that they are not alone
    dataset['IsAlone'] = 0
# if statements dont work because using dataframes dont give exact bool expressions
#    if dataset.loc[dataset['FamilySize']==1]:
#        dataset['IsAlone'] = 1
    #if they are the only one in their family to be on here then they are alone
    dataset.loc[dataset['FamilySize'] == 0, 'IsAlone'] = 1
#print (train_df[['IsAlone', 'Survived']].groupby(['IsAlone']).mean())

#start making Title feature in dataframe
for dataset in combined_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand = False)

#print (train_df[['Title', 'Survived']].groupby(['Title']).mean())

#replace rare titles with more common names that mean the same thing
#for titles that are super rare just replace them with rare
for dataset in combined_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countleas','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

#map the titles to numeric values
title_map= {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5};
for dataset in combined_data:
    dataset['Title'] = dataset['Title'].map(title_map)#.astype(int)
    dataset['Title'] = dataset['Title'].fillna(0)

#print(train_df.head())

sex_map = {'female':1,'male':2}
#map the sex feature to a numerical values
for dataset in combined_data:
    dataset['Sex'] = dataset['Sex'].map(sex_map)
    dataset['Sex'] = dataset['Sex'].fillna(0)

#fill in empty age values with the mean age 25 (FIXME: Do something else with the missing data)
for dataset in combined_data:
    dataset['Age'] = dataset['Age'].fillna(25)

#print(train_df.head(10))

#turn Age into set numeric values with decisive age ranges
for dataset in combined_data:
    dataset.loc[ dataset['Age'] <= 16 , 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64),'Age'] = 3
    dataset.loc[dataset['Age'] > 64,'Age'] = 4

embarked_map = {'C':1,'Q':2,'S':3}
for dataset in combined_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_map)
    dataset['Embarked'] = dataset['Embarked'].fillna(0)

#print(train_df.head(10))

#DROPPING FEATURES THAT DONT MATTER
#these features dont matter
#drop features at the end of in order to not mess up indexes when creating new features
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
train_df = train_df.drop(['Name','PassengerId'],axis = 1)
test_df = test_df.drop(['Name'],axis = 1)

train_df = train_df.drop(['Fare'], axis = 1)
test_df = test_df.drop(['Fare'],axis = 1)
train_df = train_df.drop(['Parch','SibSp','FamilySize'],axis = 1)
test_df = test_df.drop(['Parch','SibSp','FamilySize'],axis = 1)
combine = [train_df,test_df]


#combine = [train_df, test_df]

#USE THIS TO TEST TO SEE IF DATAFRAMES HAVE CORRECT VALUES
#print(train_df.head(10))
#print(test_df.head(10))


#maps to keep acc values and predicted values connected to algorithm name
acc_map = {}
pred_map = {}
#set up dataframes to be ready for training and testing
X_train = train_df.drop("Survived",axis = 1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId",axis = 1).copy()
#print(Y_train.head(10))

#logistic regression ML algorithm
#using sigmoid function to the plot predicted values as 0 or 1
#by their position on that sigmoid function
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100,2)
acc_map['logistic_regression'] = acc_log
pred_map['logistic_regression'] = Y_pred_log
print("Logistic Regression accuracy")
print(acc_log)


#perceptron ML algorithm
#similar t logistic_regression but instead of using sigmoid function to plot predicted values
#as 0 or 1 based on number of training data it changes the boundary for classification
#also it increases number of variables to calculate as training iterations climb
percep = Perceptron()
percep.fit(X_train,Y_train)
Y_pred_percep = percep.predict(X_test)
acc_percep = round(percep.score(X_train,Y_train)*100,2)
acc_map['perceptron'] = acc_percep
pred_map['perceptron'] = Y_pred_percep
print("Perceptron accuracy")
print(acc_percep)

#random_forest ML algorithm
#uses decision trees with random outcomes
#after each tree is evaluated take the mean of all decision tree outcomess
rand_for = RandomForestClassifier()
rand_for.fit(X_train,Y_train)
Y_pred_rand_for = rand_for.predict(X_test)
acc_rand_for = round(rand_for.score(X_train,Y_train)*100,2)
acc_map['random_forest'] = acc_rand_for
pred_map['random_forest'] = Y_pred_percep
print("Random Forest classifier accuracy")
print(acc_rand_for)

#linear support vector machines
#uses similar to perceptron with changing the boundaries for calssification but utilizes
#the extreme cases for each classifircation
lin_svc = LinearSVC()
lin_svc.fit(X_train,Y_train)
y_pred_lin_svc = lin_svc.predict(X_test)
acc_lin_svc = round(lin_svc.score(X_train,Y_train)*100,2)
acc_map['linear_svc'] = acc_lin_svc
pred_map['linear svc'] = acc_lin_svc
print("Linear support vector machines")
print(acc_lin_svc)


#see which one had the best accuracy
maximum = acc_map['logistic_regression']
for x in acc_map:
    if maximum < acc_map[x]:
        maximum = acc_map[x]
        name = x

#outputs which ML algorithm is the best
print("This was the best classifier ML algorithm for this dataset")
print(name)
print("and its accuracy was")
print(maximum)

#this is the Y_pred you use based on the best accuracy
Y_pred = pred_map[name]

#stores a submission csv file
submission = pd.DataFrame({
    "PassengerId":test_df["PassengerId"],
    "Survived":Y_pred
})

#makes the kaggle submission csv file
submission.to_csv('submission.csv',index = False)




















#getting the lay of the land data-wise
'''
#shows the column classification and a preview of the data
print(train_df.columns.values)
print('\n')
print(train_df.head())
print(train_df.tail())

#shows off the type of data the columns are in the csv files
train_df.info()
print('_'*40)
test_df.info()

#describes the data in statistical terms like mean, std-dev,min,max
print(train_df.describe())

#map a histogram of age and how many at that age survived
g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age')
plt.show(g)

#map a histogram of Pclass and how many at each class had Survived 0 and Survived 1
a = sns.FacetGrid(train_df,col = 'Survived')
a.map(plt.hist,'Pclass')
plt.show(a)
'''

'''
#garbage
train_df = train_df.drop(['Ticket','Cabin'],axis = 1)
test_df = test_df.drop(['Ticket','Cabin'],axis = 1)
combine = [train_df,test_df]
print("Before",train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)

#create Is alone feature test
for dataset in combined_data:
    dataset['IsAlone'] = 0
    if (dataset.loc[dataset["FamilySize"] == 1]):
        dataset['IsAlone'] = 1
print (train[['IsAlone','Survived']].groupby(['IsAlone'].mean()))
'''
