Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
Use the standard libraries in python for Gradient Design.

Set variables for assigning dataset values.

Import linear regression from sklearn.

Assign the points for representing the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given data.
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAYAHARI E
RegisterNumber: 212221040065
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
print('Values of MSE')
Output:
1.df.head()
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/6b386d50-228b-4cec-a742-7ef92608ec88)


2.df.tail()

![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/f525640a-053e-4d39-8669-07276105558c)


3.Array value of X

![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/76a437e7-415a-440d-bf8e-7de20e2242b7)


4.Array value of Y
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/ed0e8523-6994-4f3e-81e3-7f6c4db9c7dc)


5.Values of Y prediction
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/56fb0c71-a723-4d58-a37f-e4eb91cfdbdb)


6.Array values of Y test
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/45b88cd0-5107-40c3-a0b7-90671cdcf414)


7.Training set graph
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/9b3ecc6d-b3e6-467d-92f5-faafa5a4a450)


8.Test set graph
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/b41bfaf1-071f-480d-8c7b-aad5dd423a06)


9.Values of MSE,MAE and RMSE
![image](https://github.com/jayahari10001/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/115681467/d51aa1c9-e0be-46da-9456-a1cc06ff1d2c)


Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
