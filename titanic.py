import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Explore dataset
titanic = pd.read_csv('train.csv')
#print(titanic.head())
#print(titanic.info())

#Prepare the data for model training
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})

X = titanic.drop('Survived', axis=1)
y = titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Model training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Model evaluation
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Test data accuracy:', "{:.2f}%".format(accuracy * 100))

#Make predictions on full dataset
predictions = logreg.predict(X)
fullaccuracy = accuracy_score(titanic['Survived'], predictions)
print('Full dataset accuracy', '{:.2f}%'.format(fullaccuracy * 100))

lived = 0
for outcome in predictions:
    if outcome == 1:
        lived += 1

survivors = lived / len(predictions)
print("{:.2f}%".format(survivors * 100), 'of passengers survived')