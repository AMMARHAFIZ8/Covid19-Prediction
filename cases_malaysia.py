# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:11:45 2022

@author: ACER
"""

import os
import pandas as pd
import missingno as msno
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.metrics  import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from modules import EDA, ModelDevelopment, ModelEvaluation
import pickle



CSV_PATH = os.path.join(os.getcwd(),'cases_malaysia_train.csv')

MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model.h5')
MMS_PATH=os.path.join(os.getcwd(),'mms.pkl')

log_dir=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH = os.path.join(os.getcwd(),'Logs',log_dir)



#%%  Model Training
# EDA

# Step 1 Data Loading
df = pd.read_csv(CSV_PATH)

# Step 2 Data Inspection

df.info()
df.describe().T
df.head(10)
df.columns

df.duplicated().sum() 
df[df.duplicated()]
msno.matrix(df)

# EDA

eda = EDA()
eda.plot_graph(df) #plot the graph


# Step 3 Data Cleaning

# to change dtype from object to numerical
df['cases_new'] = pd.to_numeric(df['cases_new'],errors='coerce')

df.info()
df.isna().sum()

# there is ? character in column cases_new change to NaNs
df=df.replace('?', np.nan)

temp =df['cases_new'].interpolate(method='polynomial', order=2,inplace=True) # acts like fillna for timeseries
# temp.isna().sum()


# Step 4 Feature  Selection

# we are only selecting Opening data


# Step 5 Preprocessing

mms=MinMaxScaler()
df=mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))


X_train=[]
y_train=[]    # to initialize empty list

win_size=30

for i in range(win_size,np.shape(df)[0]):
    X_train.append(df[i-win_size:i,0])
    y_train.append(df[i,0])

X_train=np.array(X_train)
y_train=np.array(y_train)

#%% Model Development

md = ModelDevelopment()
model = md.simple_lstm_layer(X_train)

model.compile(optimizer='adam',loss='mse',metrics='mape')


plot_model(model,show_layer_names=(True),show_shapes=(True))

X_train = np.expand_dims(X_train,axis=-1)

#Tensorboard

tensorboard_callback=TensorBoard(log_dir=LOG_PATH)

hist=model.fit(X_train,y_train,
              batch_size=128,epochs=100,
              callbacks=[tensorboard_callback])

hist.history.keys()
hist_keys = [i for i in hist.history.keys()]

plt.figure()
plt.plot(hist.history['mape'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.show()

results=model.evaluate(X_train,y_train)
print(results)


#%% Model Deployment

CSV_TEST_PATH = os.path.join(os.getcwd(),'cases_malaysia_Test.csv')

test_df = pd.read_csv(CSV_TEST_PATH)

test_df['cases_new'].interpolate(method='polynomial', order=2,inplace=True)
test_df = mms.transform(np.expand_dims(test_df['cases_new'], axis=-1)) #converts into array and overwrite df


con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-130:]

X_test = []
for i in range(win_size,len(con_test)): 
    X_test.append(con_test[i-win_size:i,0])


X_test = np.array(X_test)

predicted = model.predict(np.expand_dims(X_test, axis=-1))

#Model Evaluation

me = ModelEvaluation()
me.plot_predicted_graph(test_df,predicted,mms)



#%% MSE , MAPE


print('mse :', mean_squared_error(test_df,predicted))
print('mae :',mean_absolute_error(test_df, predicted))
print('mape :',mean_absolute_percentage_error(test_df,predicted))

test_df_inversed = mms.inverse_transform(test_df)
predicted_inversed = mms.inverse_transform(predicted)

print('mse :', mean_squared_error(test_df_inversed,predicted_inversed))
print('mae :', mean_absolute_error(test_df_inversed, predicted_inversed))
print('mape :', mean_absolute_percentage_error(test_df_inversed,predicted_inversed))

print(mean_absolute_error(test_df,predicted)/sum(abs(test_df))*100)


# Model Saving
# saving model development
model.save(MODEL_SAVE_PATH)

# saving mms.pkl
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)
#%% Discussion

# The model is able to predict the trend of the covid casses
# Dispute error is around 7% maeand mape is only 15%
# can include a web scarping algorithm to analyse the latest news
# to improve the model performance
