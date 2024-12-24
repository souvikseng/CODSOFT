

  



import pandas as pd
 
ship_data= pd.read_csv('Titanic-Dataset.csv')
ship_prep= ship_data.copy()
ship_prep.isnull().sum(axis=0)
ship_prep = ship_prep.dropna()
ship_prep=ship_prep.drop(['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], axis=1)
ship_prep.dtypes
ship_prep=pd.get_dummies(ship_prep, drop_first=True)

X= ship_prep.iloc[:,:-1]
Y= ship_prep.iloc[:,1]
from sklearn.model_selection  import train_test_split

X_train, X_test, Y_train, Y_test= \
    train_test_split(X,Y, test_size=0.3, random_state=1234, stratify=Y)
    
    
    
    
    
    
    
    
    
from sklearn.linear_model import LogisticRegression
   

    
lr= LogisticRegression()

lr.fit(X_train, Y_train)
Y_predict= lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Y_predict)
score= lr.score(X_test, Y_test)

#So we see in the confusion matrix that the no of true negatives and true positives are accurately 
#predicted and their is no false predictions.