# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:12:38 2022

@author: ACER
"""

# MODULES FOR THINGS THAT USUALLY USEd SO, coding > Neat and clean 

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Input
import numpy as np

# from sklearn.metrics  import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df):
        '''
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(df['cases_new'])
        plt.plot(df['cases_recovered'])
        plt.plot(df['cases_active'])
        plt.legend(['cases_recovered','cases_active','cases_new'])
        plt.show()
        


#Model Development

class ModelDevelopment():
    def __init__(self):
        pass
    
    def simple_lstm_layer(self,X_train,num_node=64, drop_rate=0.3, output_node=1):
        
        model = Sequential()
        model.add(Input(shape=(np.shape(X_train)[1],1)))
        model.add(LSTM(num_node,return_sequences=True)) # LASTM RNN GRU only accept 3d
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node,return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,activation='relu'))
        model.summary()
        
    
        return model


# Model Evaluation

class ModelEvaluation():
    def plot_predicted_graph(self, test_df,predicted,mms):
        plt.figure()
        plt.plot(test_df,'b',label='actual cases_new')
        plt.plot(predicted,'r',label='predicted cases_new')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(mms.inverse_transform(test_df),'b',label='actual cases_new')
        plt.plot(mms.inverse_transform(predicted),'r',label='predicted cases_new')
        plt.legend()
        plt.show()
