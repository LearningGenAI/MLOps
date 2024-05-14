import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # mean_sq measures average difference statistical models (predicted valued & actual value) 
from sklearn.linear_model import ElasticNet
import joblib #this is used for saving the model
import json
import mlflow #for orchestration of mlops
from urllib.parse import urlparse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#Step2
def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]#copy from here till random state from split date file.
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dirs"] #Model_dir will be the location where ML model would be saved
#now create 2 json report files mentioned in params file
#step3 after creating the models we need to save parameters inside that location
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"] #params.jason values
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]  #Score values

    target = config["base"]["target_col"]#we need create one more Variable called target
    train = pd.read_csv(train_data_path, sep=",")#post target we need to read our train & test values
    test = pd.read_csv(test_data_path, sep=",")
 #now we need to split this data in terms of target & features. 
    train_x = train.drop(target, axis=1) #drop is to remove to target column
    test_x = test.drop(target, axis=1)

    train_y = train[target]#train & test_y is nothing but target column
    test_y = test[target]
    print("Unique values in train_y:", train_y.unique())

 #######################step4###########################################
  #lr is linear regrestion
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y) #make sure the raw data 100% clean, if not this line will not work.

    predicted_values = lr.predict(test_x)  #we are storing the predictions inside a variable based on test_x

    (rmse, mae, r2) = eval_metrics(test_y, predicted_values) #to check the accuracy of this ML model

    #print("ElasticNet model (alpha = %f, l1_ratio = %f):" %(alpha, l1_ratio))#this will help find alpha, l1_ratio value
#apart from above print, we can also check individually by below prints.
    #print("RMSE:%s" %rmse)
    #print("MAE:%s" %mae)
    #print("R2:%s" %r2)

    score_file = config["reports"]["score"] #instead of print, we can save the results in Json files
    params_file = config["reports"]["params"]

    with open(score_file, "w") as f:
        score = {
            "rmse" : rmse,
            "mae" : mae,
            "r2" : r2
        }
        json.dump(score, f, indent=4)

################Step 5##################################
    with open(score_file, "w") as f:
        score = {
            "alpha" : alpha,
            "l1_ratio" : l1_ratio,
            "r2" : r2
        }
        json.dump(score, f, indent=4)
   
    os.makedirs(model_dir, exist_ok=True) #this is ti save this file with dat.
    model_path = os.path.join(model_dir, "models.joblib")
    joblib.dump(lr, model_path)  #till this step5 will help us create model in the local machine. 
#but our target is to design & orchestrate this model in server/remote machines.

#Now create a new file in SRC "train_and_evaluate_mlops.py" for 




#Step1:
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)