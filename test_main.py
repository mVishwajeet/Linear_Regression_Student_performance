import tensorflow
import  pandas as pd
import  numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("/home/vishwajeet/Machine_Learning_Data_sets/Student_performance/student-mat.csv", sep=";")

print(data.head())

data = data[["G1" ,"G2", "G3", "failures", "studytime" ,"absences"]]  #i dont understand double brackets

print(data.head())

predict = "G3"

X = np.array(data.drop([predict],1) )  #Features #Independent variables
Y = np.array(data[predict])          #Labels  #Dependent variable

x_train , x_test , y_train , y_test =sklearn.model_selection.train_test_split(X,Y,test_size=0.2)

linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)

print(acc)
