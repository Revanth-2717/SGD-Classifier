# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1:Start

STEP 2:Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

STEP 3:Split the data into training and test sets using train_test_split.

STEP 4:Create and fit a logistic regression model to the training data.

STEP 5:Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

STEP 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

STEP 7:End

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: REVANTH.P
RegisterNumber:  212223040143
*/

import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#load the iris dataset
iris=load_iris()
#create a pandas dataframe
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
#display the first few rows of the dataset
print(df.head())
#split the data into features (x) and target (y)
x=df.drop('target',axis=1)
y=df['target']
#split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#create an SGD classifier with default parameters
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
#train the classifier on the training data
sgd_clf.fit(x_train,y_train)
#make predictions on the testing data
y_pred=sgd_clf.predict(x_test)
#evaluate the classifier's accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
#calculate the confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
dataset :
![dataset](https://github.com/user-attachments/assets/3328ca01-dacf-4acd-8447-51e11b0c80be)

classifier's accuracy:
![classifier](https://github.com/user-attachments/assets/25b8d925-2596-43de-a995-54693d6735cf)

confusion matrix:
![confusion](https://github.com/user-attachments/assets/42a070cb-f472-4d6c-82e2-59daee3a3d52)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
