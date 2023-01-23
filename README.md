# Covid19-Prediction
 To predict new cases (cases_new) in Malaysia using the past 30 days of number of cases using RNN.
 
 ## EDA 
### 1) Data Loading

### 2) Data Inspection

Train dataset graph

![Alt text](https://github.com/AMMARHAFIZ8/Covid19-Prediction/blob/main/Graph%20and%20figure/eda.png)


### 3) Data Cleaning

### 4) Features Selection
 Only selecting Opening data

### 5) Preprocessing
  
 - MinMaxScaler

  
## Model Development

1) LSTM Bidirectional Embedding
2) Stopping Callbacks 
3) Tensorboard

4) Plot Visualisation 
Plot predicted graph.

Mape

![Alt text](https://github.com/AMMARHAFIZ8/Covid19-Prediction/blob/main/Graph%20and%20figure/Figure%202022-07-01%20105347%20Mape.png)

Loss

![Alt text](https://github.com/AMMARHAFIZ8/Covid19-Prediction/blob/main/Graph%20and%20figure/loss.png)


## Model Deployment

## Model Evaluation / Analysis

![Alt text](https://github.com/AMMARHAFIZ8/Covid19-Prediction/blob/main/Graph%20and%20figure/Figure%202022-07-01%20104336.png)

![Alt text](https://github.com/AMMARHAFIZ8/Covid19-Prediction/blob/main/Graph%20and%20figure/Figure%202022-07-01%20104313.png)

MSE , MAPE
1) mean absolute error 
2) mean absolute percentage error 

## Model Saving


 ## Discussion/Reporting

The mean absolute error is 7%  and mean absolute percentage error of this model is 16% score
Model is consider great and its learning from the training.  
Training graph shows an overfitting since the training accuracy is higher  than validation accuracy
     
Model

![Alt text](https://github.com/AMMARHAFIZ8/Covid19-Prediction/blob/main/Graph%20and%20figure/model.png)


[Credit](https://github.com/MoH-Malaysia/covid19-public)
