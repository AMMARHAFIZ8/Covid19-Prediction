# Covid19-Prediction
 To predict new cases (cases_new) in Malaysia using the past 30 days of number of cases.
 
 ## EDA 
### 1) Data Loading

### 2) Data Inspection

Train dataset graph

![Alt text]()


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

![Alt text]()

Loss

![Alt text]()


## Model Deployment

## Model Evaluation / Analysis
![Alt text]()

![Alt text]()

MSE , MAPE
1) mean absolute error 
2) mean absolute percentage error 

## Model Saving


 ### Discussion/Reporting

The mean absolute error is 7%  and mean absolute percentage error of this model is 16% score
Model is consider great and its learning from the training.  
Training graph shows an overfitting since the training accuracy is higher  than validation accuracy
     
This model seems not give any effect although Earlystopping with LSTM can overcome overfitting.
With suggestion to overcome overfitting can try other 
architecture like BERT, transformer or GPT3 model.




[Credit](https://github.com/MoH-Malaysia/covid19-public)
