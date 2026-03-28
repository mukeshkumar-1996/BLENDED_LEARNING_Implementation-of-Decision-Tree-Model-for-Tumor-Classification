# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the dataset to initiate the analysis.
2.Examine the dataset to identify patterns, distributions, and relationships.
3.Determine the most important features to enhance model accuracy and efficiency.
4.Separate the dataset into training and testing sets for effective validation.
5.Use the training data to build and train the model.
6.Measure the model’s performance on the test data with relevant metrics.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: V MUKESHKUMAR
RegisterNumber:  25012063

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data=pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
x=data.drop(['Class'], axis=1)
y=data['Class']
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name: V MUKESHKUMAR")
print("Register number : 25012063")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
*/
```

## Output:
<img width="835" height="345" alt="Screenshot 2026-03-17 201032" src="https://github.com/user-attachments/assets/03917beb-4d6e-48fa-b619-9785f7e010ff" />
<img width="579" height="277" alt="Screenshot 2026-03-17 201040" src="https://github.com/user-attachments/assets/079c9e52-70a4-4880-a6d7-cafa2f4a08de" />

<img width="777" height="570" alt="Screenshot 2026-03-17 201047" src="https://github.com/user-attachments/assets/206d7f70-6ee8-4377-aeb1-860a285c7782" />


## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
