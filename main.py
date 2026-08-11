import os
import pandas as pd
import numpy as np
import joblib
 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# the .plk file analogy PICKLE make and keep till years read it when you want
MODEL_FILE = "model.pkl" # in caps as it will be constant 
PIPELINE_FILE = "pipeline.pkl"
# we do both pickle bcz the incoming data will be raw features in that imputer , scalar,oneHOtENcoder wont be there to add them outself 
# we pickle the pipeline for adding it so that the incoming data can be transformed as we did in the trianing   


def build_pipeline(num_attribs,cat_attribs):
    num_pipeline = Pipeline([  
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 6. For Categorical Columns
    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown = "ignore")), # ignores unknown values 
    ])

    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs), # take all num attribs(num_pipeline) values and add it to the list num_attribs list 
        ("cat",cat_pipeline,cat_attribs) # cat pipeline takes all cat_pipeline add it to the cat_attribs
    ])
    return full_pipeline

# if there is not model file existing! then we will have to train it! so just copy form the previous one 
if not os.path.exists(MODEL_FILE):
    # lets train the model
    housing = pd.read_csv("housing.csv")

    #stratifie test test
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
                                labels = [1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2,random_state = 42)
    
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        # "input.csv" is a string, but the to_csv() method interprets that string as a file path.
        housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False) 
        housing = housing.loc[train_index].drop("income_cat",axis=1)


    housing_labels = housing["median_house_value"].copy() 
    housing_features = housing.drop("median_house_value",axis = 1) 

    # makeing the data to fit directly into the input.csv file
    # housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index = False)

    num_attribs = housing_features.drop("ocean_proximity",axis = 1).columns.tolist() # we'll get numerical attribs as list
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs,cat_attribs)
    print(housing_features)
    housing_prepared = pipeline.fit_transform(housing_features)
    # print(housing_prepared)
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    #
    #joblib.dump(model, MODEL_FILE)      # Saves model.pkl ✅
    #joblib.dump(pipeline, MODEL_FILE)   # Overwrites model.pkl ❌
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is Trained. Congrats")

else:
    # Lets do inference this can be called Inference Piepline 
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv') 
    # we did the input.csv's corresponing variables based on the input size 
    # we can also do the above with the flask app also with an API also in JS API in cmd line 
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv",index = False) # index=False do else index will be 1,2,3,4... which wont look good
    print("Inference is complete, results saved to output.csv Enjoy!")

    # Therefore the now you can compare the median_house_value of the input and the output using which we can say the error 
    # The APproximate values in it 

    # USE imputer column transformer, oneHotENcoder do all this and predict with high level of accuaracy

    # we can integrate in flask app and plug directly into it 

    # H.W : which ever data you do for next Model use Other techniques rather than DecisionTree,LinearReg,RandomFOres  test OTHERS and chack
    # differennt problems have differnt parameters 
    
    # in SKlearn GridSearchCV helps to remove the best HypterParameters to remove an Particular Algo 
    # Ex : in RandomFOrestRegressor go to google in its doc there are many things like n_estimators = 100,..= squared_error
    # TO find out what were its results and the stuff are how it will change noteting down what are the best results for it
    
    
    # noting it down is DIfficult so we use GridSearchCV this is used for HyperParameterTUning this is what we call hyperParameter tuning 
    # i.e what are those values which we can tweek and make this algo perform best 
    # RandomSearchCV also work the same way where we can find the best HyperParameters  




