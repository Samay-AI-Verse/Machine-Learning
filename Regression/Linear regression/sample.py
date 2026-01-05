import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('C:\Data Science\Machine Learing\Linear Regression\placement.csv')
x = data['cgpa']
y = data['package']

x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

reg = LinearRegression()

reg.fit(x_train,y_train)

input1 = float(input('Enter Your CGPA: '))

result = reg.predict([[input1]])

print('Your Package Will Become: ',result)