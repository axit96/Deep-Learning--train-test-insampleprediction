# Deep-Learning--train-test-insampleprediction

Hello, 

Here I used both incremental training and insample prediction.

This three algorithms take the daily weather data as input, it will ingore last 7 values of dataset as we want to do insample prediction, firstly model will take forst four year data(4*365) as input then it will do train,test andlast sevendays insample prediction. After this first itteration it will increment data set by 3 months data(91 datasamples) then again repeat steps train, test and sevendays insample prediction. This process will continue till end of dataset.

## To run this program 
First you need to install python. then required libraries are tensorflow, keras, matplotlib,math, numpy, pandas and sklearn.



Thank you and Greetings
