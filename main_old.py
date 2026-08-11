import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score 

#1. load data
housing = pd.read_csv("housing.csv")

#2.stratifie test test
# we make a variety of median_income whose name we kept income_cat
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
                               labels = [1,2,3,4,5])
# all the incoem_cat is representative of the training and testing set of the population
split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2,random_state = 42)

for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat",axis=1) # set aside the test data

# 2. copy of training data
housing = strat_train_set.copy()

# 3. sep features and labels we make a copy() with the instance of housing_labels to use later if we want
housing_lables = housing["median_house_value"].copy() # labels of House 
# axis = 1 dropped median house value as its the thing the model has to predict so drop it 
housing = housing.drop("median_house_value",axis = 1) 

# print(housing,housing_lables)

# 4. List the numerical and Categorical columns
num_attribs = housing.drop("ocean_proximity",axis = 1).columns.tolist() # we'll get numerical attribs as list
cat_attribs = ["ocean_proximity"]

# 5. Lets make the pipeline 
# for numerical columns
# pipeline takes list
num_pipeline = Pipeline([  
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. For Categorical Columns
cat_pipeline = Pipeline([
#     # (ordinal,ordinalencoder) 
# '''
# #     Caution: Ordinal encoding implies an order between categories, which may not
# # be true here. For example, it treats INLAND (1) as closer to <1H OCEAN (0) than
# # NEAR OCEAN (4) , which might not make sense.
# # 
# # One-Hot Encoding:
#           For unordered categories, one-hot encoding is a better choice. It creates one
#           binary column per category.
# '''

    ("onehot",OneHotEncoder(handle_unknown = "ignore")), # ignores unknown values 
])
# Full Pipeline
# column transformer merges the values back into one table
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attribs), # take all num attribs(num_pipeline) values and add it to the list num_attribs list 
    ("cat",cat_pipeline,cat_attribs) # cat pipeline takes all cat_pipeline add it to the cat_attribs
])
# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
#print(housing_prepared)

# 7. Train the model
# Linear Regression
lin_reg = LinearRegression()
# the below line will fit the Linear Regre model in housing_prepared data in one line
lin_reg.fit(housing_prepared,housing_lables) 
lin_preds = lin_reg.predict(housing_prepared)
# since rmse predicts the vlaues between actual(housing data) and the predicted values(lin_predicts) read its formula
lin_rmse = root_mean_squared_error(housing_lables,lin_preds) 
#print(f"The root mean squared error for Linear Regression is {lin_rmse}")
lin_rmses = -cross_val_score(lin_reg,housing_prepared,housing_lables,scoring = "neg_root_mean_squared_error", cv = 10)
print(pd.Series(lin_rmses).describe())

# NOTE : IN all the models cv(cross validation) = 10 beacuse all the training will be K-1 = 10-1 = 9 values of taraining and 1 value of Testing

# For Decision Tree Model
# use multi cursor by copying same code from Linear Regression key : Alt + Click
# dont add random state as its seed which will fix the data n_splits = 10 what ever is given 
# if not added each time it will take new / differnt variables in it 
dec_reg = DecisionTreeRegressor() 
dec_reg.fit(housing_prepared,housing_lables) 
dec_preds = lin_reg.predict(housing_prepared)
#dec_rmse = root_mean_squared_error(housing_lables,dec_preds) 
dec_rmses = -cross_val_score(dec_reg,housing_prepared,housing_lables,scoring = "neg_root_mean_squared_error", cv = 10)

#print(f"The root mean squared error for DecisionTreeClassifier is {dec_rmse}")
print(pd.Series(dec_rmses).describe())
# Ctrl + X the DecisionTreeClassifier , Ctrl N This will open a new tab 
# then change all the Occurences and make the Randomforest changes just to reduce the time

# Random Forest Model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared,housing_lables) 
random_forest_preds = random_forest_reg.predict(housing_prepared)
random_forest_rmse = root_mean_squared_error(housing_lables,random_forest_preds) 
#print(f"The root mean squared error for RandomForestRegresion is {random_forest_rmse}")

random_forest_rmses = -cross_val_score(random_forest_reg,housing_prepared,housing_lables,scoring = "neg_root_mean_squared_error", cv = 10)

print(pd.Series(random_forest_rmses).describe())

# NOTE : A Warning About Training RMSE
# Training RMSE only shows how well the model fits the training data. 
# It does not tell us how well it will perform on unseen data. 
# In fact, the Decision Tree and Random Forest may overfit, 
# leading to very low training error but poor generalization.

# if the root mean Squared Error is 0.0 the model is Overfit like all the values given
# Ex: asssume (F(x) = x pow 2 , Q) x = 2 val is 4 , for x = 4 val is 8) all this given inside the DATA
#  like wise there is no guareantee the value will be correct for the input data as well
# Overfiting wont work for Future labels / Generalisation variable  also

# NOTE: After Training the Model and finding the Best fit for it 
# After testing all the Models we can say that RandomForest has the least mean Meaning it will have the 
# Least Error making it Efficient than the DecisionTreeRegressor and the LinearRegressor

