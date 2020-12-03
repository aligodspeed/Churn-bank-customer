# Churn-bank-customer
Here in this project, I will use classification algorithms to make model to predict customer’s behavior and prevent them from abandoning bank's service.

## Directory Structure

### [data](https://github.com/aligodspeed/Churn-bank-customer/tree/main/data)
This folder contains data from kaggle (https://www.kaggle.com/shivan118/churn-modeling-dataset)which has Information of 10000 customers from three countries France, Germany and Spain.

### [img](https://github.com/aligodspeed/Churn-bank-customer/tree/main/img)
This folder contains images and visualizations of our project.

### [notebooks](https://github.com/aligodspeed/Churn-bank-customer/tree/main/notebooks)
This folder contains jupyter notebooks with EDA data analyses and my final report notebook. 

### [Reference](https://github.com/aligodspeed/Churn-bank-customer/tree/main/reference%20)
This folder contains outside reference that we used for extra information during our project and the pdf file of my notebook.

### [reports](https://github.com/aligodspeed/Churn-bank-customer/tree/main/reports)
This folder contains a non-technical presentation pdf file.


### [src](https://github.com/aligodspeed/Churn-bank-customer/tree/main/src)
This folder contains two helper functions to plot Confusion matrix and get cross_val score.

### [README](https://github.com/aligodspeed/Churn-bank-customer/blob/main/README.md)
This folder contains a brief explanation of the project.

### [environment.yml](https://github.com/aligodspeed/Churn-bank-customer/blob/main/environment.yml)
This folder contains all the necessary packages needed to recreate your conda environment.

## Overview and Business Question.
Customer churn, or Exited in this project, occurs when customers stop doing business with a company. The companies are interested in identifying segments of these customers because the price for acquiring a new customer is usually higher than retaining the old one. Banks are a good example of this companies. If banks know how they customers leave their account, they could prevent it by special offers to keep their clients. What will help businesses to keep their customers is making a model using past data to predict the number of churn or exited customers.

I am going to use classification algorithms to make model to predict customer’s behavior and prevent them from abandoning bank's service. What my model does is predicting higher number of class 0 which is Not Exited customers and keep lower the number of class 1 which is Exited customers as well as keep as low as possible the number of False Negative.

Fales Negative is the number of Exited customers who predicted not Exited and False Positive is the number of stayed customers who predicted exited. This would be a misleading for a company which tries to have the best predict of its customers. To do So, I will use f1 score as my metric to make my model. Also, I will use ROC (Receiver Operating Characteristic) curve to show performance over a range of trade-offs between true positive (TP) and false positive (FP) error rates.

I work with dataset from Kaggle which has Information of 10000 bank's customers from three countries France, Germany and Spain. Some of features in this dataset such as Age, Balance, Gender and regions of banks are the most correlated with Exited column which is the target of my prediction. I will use those features for visualization and recommendations.

Here in this project, I will help stakeholders of this bank to find the causes of churned customers and keep their clients.

## Data
I work with dataset from Kaggle(https://www.kaggle.com/shivan118/churn-modeling-dataset) which has Information of 10000 customers from three countries France, Germany and Spain. Some of futures in this dataset such as Age, Balance and Gender are the most correlated with Exited. I will use those futures for my recommendation to the bank.
Download data from Kaggle Churn Modeling.
Upload data into data folder inside the repository.
Data is a 668.81 KB zip file which can be unzipped from terminal. The best way to do is change the directory in terminal to where data will store, use unzip command following by uploaded data folder.
The unzipped data is a single csv file which is ready to use.

## Visualization 
I made some visualizations that I used in my presentation. These plots and bars are the most correlated features with target. These visualizations give me a good vision of dataset and more information about what dataset looks like and what I need to work on. 

- Distribution of Age vs target 
![Age](/img/Age_Contribution.png)

- Distribution of Balance vs target
![Balance](/img/Balance_Contribution.png)

- Number of Male=0 and Female=1
![Gender](/img/Gender-bar.png)

- Number of customers in countries vs exited
![Countries](/img/Geography_count.png)

## EDA

Before making any model, I need to do some preprocessing and preparation on raw data that I have in data folder. 
Although this dataset is clean, I did some work on this dataset:" See my final notebook at (https://github.com/aligodspeed/Churn-bank-customer/blob/main/notebooks/reports/final-notebook.ipynb) for more information".

- Changing Gender column from categorical to binary
- Dropping Useless columns 
- Making Exited column as my target
- Split our dataset to training and test set.
- Dealing with Categorical Columns
- One Hot Encoding 
- Scaling 
The most important part of preprocessing of my dataset is that this dataset is heavily imbalance. This means that the number of values in target (Exited column) which are
0= Stayed or not exited and 1= Exited or churn customers is inequal, and this will affect our result, because the number of stayed are much more than churned customers. To avoid the power of stayed customers, I use the Over Sampling technique which makes the number of two classes equal. 
I used RandomOverSampler technique.

Original dataset shape Counter ({0: 6356, 1: 1644})
Resample dataset shape Counter ({0: 6356, 1: 6356})

![Imbalance_data](/notebooks/exploratory/Imbalance_data.png)

## Model
As first simple model, I used LogisticRegresion, because it's simple, fast and easy to interpretation. The average of f1 score that I got from all LogisticRegresion was about 0.49 which is not acceptable.
The next algorithm was DecisionTreeClassifier and the average f1 score of 0.50
Makes the training time of decision tree faster.
Because of its simplicity, it's easy to code, visualize, interpret, and manipulate simple decision tree.
Decision tree follows a non-parametric method.
Can work on both categorical and numerical data.
For the next model, I used RandomForestClassifier. The f1 score from this model was between 0.59 to 0.62
I use Random Forest, because it creates as many trees on the subset of the data and combines the output of all the trees. In this way it reduces overfitting problem in decision trees and also reduces the variance.
Random Forest works well with both categorical and numerical variables.
The last algorithm was XGBClassifier with the average f1 score of 0.60
Although I got good score out of Random Forest model, it was slow to run train set.
I use XGBoost to see if I get better or the same score in better computational speed.
XGBoost has in-built L1 (Lasso Regression) and L2 (Ridge Regression) regularization which prevents the model from overfitting.
XGBoost allows user to run a cross-validation at each iteration of the boosting process.
Although Random forest does not have the best computational speed and it takes more time to run compare to other models, it gave me the best f1 score.
I use Random Forest model with Grid Search as my best model.

## 
## Next step
- Is there any pattern between mobile customer and bank churn?
- Does my model work with another dataset?

## Recommendation 
# Extra data before churn
- Number of customer service call
- Number of in-person customers in bank
- Make survey frequently
# Extra data after churn
- who they talked to for the last time
- who they visited for the last time
