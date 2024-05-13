# -*- coding: utf-8 -*-
"""

@author: sksha
"""
import numpy as np
import pandas as pd
df = pd.read_csv("Zoo.csv")
df.head()
df.shape   #(101, 18)
df.info()

#==================================================================================================
#EDA
#BOXPLOT AND OUTLIERS CALCULATION #
df1 = df.iloc[:,1:18]
from scipy import stats
# Define a threshold for Z-score (e.g., Z-score greater than 3 or less than -3 indicates an outlier)
z_threshold = 3
# Calculate the Z-scores for each column in the DataFrame
z_scores = np.abs(stats.zscore(df1))

# Create a mask to identify rows with outliers
outlier_mask = (z_scores > z_threshold).any(axis=1)

# Remove rows with outliers from the DataFrame
df = df[~outlier_mask]
df.shape  #(93, 18)
df.info()
# Now, df contains the data with outliers removed

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()

#=============================================================================================
#Split the variables
X = df.iloc[:,1:16]
X
# Standardization of feature variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X

Y = df["type"]
Y
# Label Encoding for the target variable 'type'
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)

#=============================================================================================
#DATA PARTITION
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75)

#=============================================================================================
#KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5,p=2) #5,7,9,11,13,15
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

#METRICS ACCURACY SCORE
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training accuracy:",ac1.round(3))   #Training accuracy: 0.928
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test accuracy:",ac2.round(3))       #Test accuracy: 0.875

#VALIDTAION APPROACH FOR KNN
l1 = []
l2 = []
training_accuracy=[]
test_accuracy=[]
for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.75,random_state=i)
    KNN = KNeighborsClassifier(n_neighbors=11,p=2) #n_neighbors =5,7,9,11,13,15 #best k value = 11
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
print("Average Trianing accuracy :",np.mean(training_accuracy))  #Average Trianing accuracy : 0.8216952129995607
print("Average Test accuracy :",np.mean(test_accuracy))          #Average Test accuracy : 0.7916666666666669

#==========================================================================================
#Average accuracies are getting stored 
l1.append(np.mean(training_accuracy))
l2.append(np.mean(test_accuracy))
print(l1)
print(l2)

#subtracting two list by converting into arrays
l1 
l2  

array1= np.array(l1)
array1
array2= np.array(l2)
array2
deviation = np.subtract(array1,array2)
deviation
list(deviation.round(3))
#==============================================================================================
# Visualisation
# Visualize the distribution of the target variable 'type'
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=df)
plt.title('Distribution of Animal Types')
plt.xlabel('Animal Type')
plt.ylabel('Count')
plt.show()

# Visualize the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# The initial KNN model with 5 neighbors achieved a training accuracy of approximately 92.8% and a test accuracy of 87.5%.
# When using a validation approach with different random states, the average training accuracy was around 85.6%, and the average test accuracy was approximately 81.6%.
# visualisation for  the distribution of the target variable 'type' hass been done
# also we have obtained the confusion matrix visualisation
#==============================================================================================


