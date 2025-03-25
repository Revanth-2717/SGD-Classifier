# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1. start

step 2. Import Necessary Libraries and Load Data

step 3. Split Dataset into Training and Testing Sets

step 4. Train the Model Using Stochastic Gradient Descent (SGD)

step 5. Make Predictions and Evaluate Accuracy

step 6. Generate Confusion Matrix

step 7. end

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: RAGALA SAI VIVEK
RegisterNumber: 212223230163 
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

dataset
![dataset](https://github.com/user-attachments/assets/2ff1d906-0d78-47f6-b866-1f9f5dade3f3)

classifier's accuracy
![classifier](https://github.com/user-attachments/assets/770f7a7d-1c73-4c48-b86c-720281309033)

confusion matrix
![CONFUSION MATRIX](https://github.com/user-attachments/assets/4768443a-183e-4593-83a4-a46e1c877bda)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
