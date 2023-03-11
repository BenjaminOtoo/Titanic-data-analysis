import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Explore dataset
titanic_df = pd.read_csv('train.csv')
#print(titanic_df.head())
#print(titanic_df.info())

#Prepare the data for model training
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Model training
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Model evaluation
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

#Make predictions on new data
new_data = pd.DataFrame({
    'Pclass': [3, 1, 1],
    'Sex': [0, 1, 1],
    'Age': [25, 35, 50],
    'SibSp': [0, 1, 1],
    'Parch': [0, 2, 2],
    'Fare': [7.8958, 120, 120],
})
predictions = logreg.predict(new_data)
print('Predictions:', predictions)
