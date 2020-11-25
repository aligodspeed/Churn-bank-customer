# Churn-bank-customer
Here in this project, I will use classification algorithms to make model to predict customer’s behaviour and prevent them from abandoning bank's service.

## Directory Structure

### [data](https://github.com/aligodspeed/Churn-bank-customer/tree/main/data)
This folder contains data from kaggle(https://www.kaggle.com/shivan118/churn-modeling-dataset)which has Information of 10000 customers from three countries France, Germany and Spain.

### [img](https://github.com/aligodspeed/Churn-bank-customer/tree/main/img)
This folder contains images and visualizations of our project.

### [notebooks](https://github.com/aligodspeed/Churn-bank-customer/tree/main/notebooks)
This folder contains jupyter notebooks with eda data analyses and our final report notebook. 

### [Reference](https://github.com/aligodspeed/Churn-bank-customer/tree/main/reference%20)
This folder contains outside reference that we used for extra information during our project and the pdf file of my notebook.

### [reports](https://github.com/aligodspeed/Churn-bank-customer/tree/main/reports)
This folder contains a non-technical presentation pdf file.


### [src](https://github.com/aligodspeed/Churn-bank-customer/tree/main/src)
This folder contains the source code to created functions in use during this project.

### [README](https://github.com/aligodspeed/Churn-bank-customer/blob/main/README.md)
This folder contains a brief explanation of the project.

### [environment.yml](https://github.com/aligodspeed/Churn-bank-customer/blob/main/environment.yml)
This folder contains all the necessary packages needed to recreate your conda environment.

## Overview and Business Question.
Customer churn,or Exited in our project, occurs when customers stop doing business with a company. The companies are interested in identifying segments of these customers because the price for acquiring a new customer is usually higher than retaining the old one. Banks are a good example for this. If banks know how they customers leav their account, they could prevent it by special offers to keep them. What will help business to keep their customers is making a model from past data to predict the number of churn or exited customers.

Here in this project, I will use classification algorithms to make model to predict customer’s behaviour and prevent them from abandoning bank's service. What my model does is predicting higher number of class 0 which is Not Exited customers and keep lower the number of class 1 which is Exited customers as well as keep as low as possible the number of False Negative.

Fals Negative is the number of Exited customers who predicted not Exited. This would be a misleading for a company which tries to have the best predict of its customers. To do So, I will use f1 score as my metric to make my model. Also, I will use ROC(Receiver Operating Characteristic) curve to show performance over a range of trade-offs between true positive (TP) and false positive (FP) error rates.

## Data
I work with dataset from Kaggle(https://www.kaggle.com/shivan118/churn-modeling-dataset) which has Information of 10000 customers from three countries France, Germany and Spain. Some of futures in this dataset such as Age, Balance and Gender are the most correlated with Exited. I will use those futures for my recommendation to the bank.

## Model
For the first simple model, I used LogisticRegresion, because it's simple, fast and easy to interpretation. The average of f1 score that I got from all LogisticRegresion was about 0.49 which is not acceptable.
Then I tried KNeighborsClassifier with the average f1 score of 0.48
The next algorithm was DecisionTreeClassifier and the average f1 score of 0.50. It was slightly improved, but not good enough to use as my model.
For the next model, I used RandomForestClassifier. The f1 score from this model was about 0.59 and close to 0.62 after I used GridSearch as hyper-tuning technique.
The last algorithm was XGBClassifier with the average f1 score of 0.60
I chose RandomForestClassifier with GridSearch to finalize my model and get the result on test set.

## Next step and recommendations
Based on distribution plots from Age, Balance and the region of banks, I would recommend:

keep tarck of age on customers, because it will show that customers between age of 40 to 50 are mostly leave their accounts especially at age of 40 and 50.

Customers with balance between 100,000 and 150,000 will be at risk of leaving bank. They can have especial offer for customers once they reach that amount.

Also, it is mostly to leave bank account from Germany than other countries. They can have further investigation about customer service and other involvement in this process.
