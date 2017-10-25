# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 01:25:22 2017

@author: Arjun
"""

#Implmenting multiple machine learning models and evaluating the performance
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#We are going to use IOWA real estate data set and perform prediction
#reading the contents from iowa file and printing them
iowa_file = 'C:/Users/Arjun/Desktop/Project december/train.csv';
iowa_contents = pd.read_csv(iowa_file);
print(iowa_contents);
#repeating the same step for IOWA dataset, select one of the columns
iowa_year_sold = iowa_contents.YrSold;
#print first few lines of the selected column using print() and head()
print(iowa_year_sold.head())
#selecting three columns of interest from iowa data
my_interest = ['YrSold','YearBuilt','Neighborhood'];
#laoding the data from selected columns onto a new variable
my_interest_data = iowa_contents[my_interest];
#summary of the selected data
my_interest_data.describe(); 
#First step in building the model,choose the prediction target
iowa_y = iowa_contents.SalePrice;
#choosing the predictors
iowa_predictors = ['LotArea',
'YearBuilt','TotalBsmtSF','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr',
'TotRmsAbvGrd','GarageArea'];
iowa_X= iowa_contents[iowa_predictors];
#performing the test train split
iowa_X_train,iowa_X_test,iowa_y_train,iowa_y_test = train_test_split(iowa_X,iowa_y);
#Building the models and fitting the data
iowa_models = [DecisionTreeRegressor(),LinearRegression(),RandomForestRegressor()];
for model in iowa_models:
    model.fit(iowa_X_train,iowa_y_train)
    model_pred=model.predict(iowa_X_test);
    Error_Measure = mean_absolute_error(iowa_y_test,model_pred);
    print(Error_Measure);
#Error mesaure can be used to identify which model performs well
#in our case it error is highest for Decision tree, followed by linear regression 
#Random Forest regressor performs better when compared to the other two models