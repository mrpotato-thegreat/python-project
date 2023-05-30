import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

                                ## Data exploration ##
data=pd.read_csv(r"C:\Users\EL RWAD\Downloads\Salary_Data.csv")
print(data)

                                ## Data pre-processin ##
print(data.shape)
print(data.head(0))
print( data.nunique)
data=data.dropna()
print(data.dtypes)
print(data.isnull())
print(data.isnull().sum())
print(data.fillna(2))
print(data.duplicated())
print(data.duplicated().sum())
print(data.nunique())

#############################=====> Data Encoding   >======###########################

from sklearn import preprocessing
Pr_data = preprocessing.LabelEncoder()
d_types=data.dtypes
for i in range(data.shape[1]):
  if d_types[i]=='object':
   data[data.columns[i]]=Pr_data.fit_transform(data[data.columns[i]])
# print(data)

# ############################=====> Data correlation  >======###########################

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
zy = scaler.fit_transform(data)
new_data=pd.DataFrame(data=zy,columns=data.columns)
# print(zy)
Z=new_data.corr()
r=data.corr()
sns.heatmap(r, annot=True)
sns.pairplot(data)
plt.show()
A=data["Age"]
G=data["Gender"]
plt.plot(A,G)
plt.show()

# ###############################======> model 1 >======#########################

from sklearn.tree import DecisionTreeClassifier
data = data.drop(["Gender"], axis = 1)
from sklearn.model_selection import train_test_split
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
model_1= DecisionTreeClassifier()
model_1.fit(x_train, y_train)
score=model_1.score(x_test, y_test)
print(score)

#################################======> model 2 >=======##########################

from sklearn.naive_bayes import GaussianNB
model_2=GaussianNB()
model_2.fit(x_train,y_train)
score_2=model_2.score(x_test,y_test)
print(score_2)

#################################======> model 3 >========###########################

from sklearn.linear_model import LogisticRegression
model_3=LogisticRegression()
model_3.fit(x_train,y_train)
score_3=model_3.score(x_test,y_test)
print(score_3)

