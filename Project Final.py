import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
data=pd.read_csv('boston_housing.csv')

data.info()

x1=data.medv

y1=data.lstat
y2=data.rm
y3=data.ptratio
y4=data.crim

plt.subplot(221)
plt.scatter(x1,y1,c='b')
plt.xlabel('Medv')
plt.ylabel('Lstat')

plt.subplot(222)
plt.scatter(x1,y2,c='r')
plt.xlabel('Medv')
plt.ylabel('Rm')

plt.subplot(223)
plt.scatter(x1,y3,c='y')
plt.xlabel('Medv')
plt.ylabel('Ptratio')

plt.subplot(224)
plt.scatter(x1,y4,c='g')
plt.xlabel('Medv')
plt.ylabel('Crim')

plt.show()

plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True)
plt.show()

#Linear Regression Model:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20 ,random_state = 3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(xtrain,ytrain)
ypred=regressor.predict(xtest)



from sklearn.metrics import mean_squared_error

print(" \n Mean Squared Error:")
print(mean_squared_error(ytest,ypred)) 

print("\n Accuracy:")
print(regressor.score(xtest,ytest))



#the bellow are just some dummy test cases made for the purpose of callculation which need not be considered
print("IGNORE BELLOW")
























print("\n Average medv value = 20.021")

print("Test case 1: 1st row of dataset")
#1st row of dataset
ypred1=regressor.predict([[0.0063,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98]])
print(ypred1)

print("Test case 2: Last row of dataset")
#Last row
ypred2=regressor.predict([[0.04741,0,11.93,0,0.573,6.03,80.8,2.505,1,273,21,396.9,7.88]])
print(ypred2)

print("Test case 3: Average of every columns of dataset")
#Average
ypred3=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,9.549,408.237,18.455,356.674,12.653]])
print(ypred3)

print("Test case 4: Highest Lstat all others are avarage of dataset")
#highest Lstat all others are avarage
ypred4=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,9.549,408.237,18.455,356.674,37.97]])
print(ypred4)

print("Test case 5: Highest black all other are avarage of dataset")
#highest black all other are avarage
ypred5=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,9.549,408.237,18.455,396.6,12.653]])
print(ypred5)

print("Test case 6: Highest ptratio all other are avarage of dataset")
#highest ptratio all other are avarage
ypred6=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,9.549,408.237,22,356.674,12.653]])
print(ypred6)

print("Test case 7: Highest tax all other are avarage of dataset")
#highest tax all other are avarage
ypred7=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,9.549,711,18.455,356.674,12.653]])
print(ypred7)

print("Test case 8: Highest rad all other are avarage of dataset")
#highest rad all other are avarage
ypred8=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,24,408.237,18.455,356.674,12.653]])
print(ypred8)

print("Test case 9: Highest dis all other are avarage of dataset")
#highest dis all other are avarage
ypred9=regressor.predict([[3.613,11.36,11.136,0.0691,0.554,6.284,68.574,12.1265,9.549,408.237,18.455,356.674,12.653]])
print(ypred9)

print("Test case 10: Highest crime all other are avarage of dataset")
#highest crime all other are avarage
ypred10=regressor.predict([[88.9762,11.36,11.136,0.0691,0.554,6.284,68.574,3.795,9.549,408.237,18.455,356.674,12.653]])
print(ypred10)


