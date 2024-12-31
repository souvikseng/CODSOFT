

from sklearn import datasets
iris = datasets.load_iris()

X= iris.data 
Y=iris.target   

#Split the X and Y dataset into training and testing set
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y, test_size=0.3, random_state=1234, stratify=Y)
    
    
    
   
    
   
from sklearn.svm    import SVC
    
from sklearn.metrics import confusion_matrix
   
   
    
   
    
svc=SVC(kernel='rbf', gamma=1.0)
  

  
svc.fit(X_train, Y_train)
    
Y_predict = svc.predict(X_test)
cm_rbf= confusion_matrix(Y_test, Y_predict)

#So we see that the no of correctely or true positives predicted data are more in the confusion matrix. 
