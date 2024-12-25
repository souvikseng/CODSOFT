

import pandas as pd




dataset= pd.read_csv('advertising.csv')

df=dataset.copy()

df= df.drop(['Radio','Newspaper'], axis=1)

X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =  \
    train_test_split(X,Y,test_size=0.3, random_state=1234)
    
 
    
 
    
from sklearn.linear_model  import LinearRegression














std_reg= LinearRegression()
  
std_reg.fit(X_train, Y_train)
 
Y_predict= std_reg.predict(X_test)

slr_score= std_reg.score(X_test, Y_test)

slr_coefficient= std_reg.coef_
slr_intercept= std_reg.intercept_

from sklearn.metrics import mean_squared_error
import math
slr_rmse= math.sqrt(mean_squared_error(Y_test,Y_predict)) 
 
import matplotlib.pyplot as plt
plt.scatter(X_test, Y_test)

plt.scatter(X_test, Y_predict)

plt.show()
   
   