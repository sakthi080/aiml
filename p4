import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=dataset = pd.read_csv(r'C:\Users\lenovo\Desktop\insurance.csv')
df.head()
df['sex'] = df['sex'].apply({'male':0,'female':1}.get)
df['smoker']= df['smoker'].apply({'yes':1,'no':0}.get)
df['region'] = df['region'].apply({'southwest':1,'southeast':2,'northwest':3,'northeast':4}.get)
df.head()
x = df.drop(['charges','sex'], axis=1)
y = df.charges
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state =42)
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
print("y_train shape: ",y_train.shape)
print("y_test shape: ",y_test.shape)
Linreg = LinearRegression()
Linreg.fit(x_train,y_train)
pred = Linreg.predict(x_test)
from sklearn.metrics import r2_score
print("R2 score: ",(r2_score(y_test,pred)))
Data = {'age':50,'bmi':25,'children':12 ,'smoker':0,'region':2}
Index =[0]
Cust_df = pd.DataFrame(Data,Index)
Cust_df
Cost_pred1 = Linreg.predict(Cust_df)
print("The medical insurance cost of the new customer category Non Smoker : ",Cost_pred1)
Data = {'age':50,'bmi':25,'children':12 ,'smoker':1,'region':2}
Index =[0]
Cust_df = pd.DataFrame(Data,Index)
Cust_df
Cost_pred2 = Linreg.predict(Cust_df)
print("The medical insurance cost of the new customer category Smoker : ",Cost_pred2)
