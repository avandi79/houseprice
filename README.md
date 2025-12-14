import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
df = pd.read_csv("housePrice.csv")
df.head()
display(df.dtypes)
print("total number:", len(df))
## cleaning Area columns 
df['Area'] = (
    df['Area']
    .astype(str)
    .str.strip()
    .str.replace(',', '', regex = False)
)
df['Area'] = pd.to_numeric(df['Area'], errors='coerce').astype('Int64')
print("max area in before cleaning :", df['Area'].max())
print("len of before:", len(df))
df = df[(df['Area'] > 0) & (df['Area'] <= 500)].reset_index(drop=True)
print("max area in after cleaning:", df['Area'].max())
print("len of clean:", len(cdf))
## cleaning address column  
addr = df['Address'].astype('string')
suspect = addr[~addr.str.contains(r'[A-Za-z0-9]', na=False)]
print( "rows with empty/missing value:", len(suspect))
print(suspect.map(repr).head(30))    
df = df.dropna(subset=['Address']).reset_index(drop=True)
print(df['Address'].isna().sum())
print(len(df))
## make a new copy
clean_df = df.copy()
clean_df.describe()
## Encoding Address 
len(clean_df['Address'].unique())
vc = clean_df['Address'].value_counts()
print(vc.head(10))
print(vc.tail(10))
threshold = 30
top_addrs = vc[vc >= threshold].index
print(len(top_addrs))
clean_df['Address_g']=np.where(
    clean_df['Address'].isin(top_addrs),
    clean_df['Address'],
    'Others'
)
print(clean_df['Address_g'].value_counts())
print(clean_df['Address_g'].unique())
num_features = ['Area','Room','Parking','Warehouse','Elevator']
X_num = clean_df[features].astype(float)
X_addr = pd.get_dummies(
    clean_df['Address_g'],
    prefix='addr',
    drop_first=True,
)
X_encoded = pd.concat([X_num, X_addr], axis=1)
y = clean_df['Price']
print(X_encoded.shape)
print(X_encoded.dtypes.unique())    
X_model = X_encoded.astype(float).copy()
print(X_model.dtypes.unique()) 
## Drawing histograms 
viz = clean_df[['Room','Price','Area','Address']]
viz.hist(figsize=(8,8))
plt.show()
## Making plot
plt.figure(figsize=(8,6))
plt.scatter(clean_df['Area'], clean_df['Price'], color='purple')
plt.xlabel('Area')
plt.ylabel('Price')
plt.grid(True)
plt.xlim(0,600)
## Spilt dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y,
    test_size = 0.20,
    random_state=42,
)
print(X_train.shape)
print(X_test.shape)
## modelling 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("R2: ", r2_score(y_test,y_predict))
print("RMSE: ", root_mean_squared_error(y_test,y_predict))
