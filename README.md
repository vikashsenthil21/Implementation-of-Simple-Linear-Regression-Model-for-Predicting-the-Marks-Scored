# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```


Developed by: vikash s
RegisterNumber:  212222240115

import pandas as pd
df= pd.read_csv('/content/student_scores.csv')
df.info()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)

plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()


mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)

a=np.array([[10]])
y_pred1=reg.predict(a)
print(y_pred1)
*/
```

## Output:

# 1. df.head():

![2](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/2b3a6c3f-8902-468f-bceb-97bad95c3d21)

# 2. df.tail():

![3](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/b608e811-9e93-41bc-9bd6-b9dd2a6567c5)

# 3. Array value of X:

![4](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/557615a3-e691-4b85-9026-539b11beafa2)

# 4. Array value of Y:


![5](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/78b71082-f716-4bb4-aab7-df5ab0a8081e)

# 5. Values of Y prediction:

![8](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/ea47338b-f958-4898-8fec-55128763d114)


# 6. Array values of Y test:

![7](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/c46388d8-8024-441b-9277-9e752c33356d)


# 7. Training Set Graph:


![9](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/cbbf58d6-11d3-4af2-bab4-4b20237cfba2)



# 8. Test Set Graph:

![10](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/76f43df7-d040-49cd-8e66-068136ed5697)



# 9. Values of MSE, MAE and RMSE:


![11](https://github.com/Pavithraramasaamy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118596964/d98c36d0-4ef9-41a4-87fb-2c314803fe3a)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
