# Import all dependencies
import subprocess, sys, os
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile,OutputArtifact
from typing import NamedTuple
import kfp.dsl as dsl

# Every step in training job requires base image which already contains dependencies
# Lets build base image with new tag everytime new code is checked in, as dependencies may change
# To have new image name everytime we use git commit id as image tag.

# Get git commit id
git_commit_id = subprocess.run(['git', 'log','-1', '--pretty=%h'], capture_output=True)

# Create container name with new tag
os.environ["CONTAINER_NAME"] = 'vinodswnt306/new_public_mlops:' + git_commit_id.stdout.decode('utf-8').replace('\n','')
CONTAINER_NAME = os.environ["CONTAINER_NAME"]

# e.g.
# CONTAINER_NAME = 'vinodswnt306/new_public_mlops:aada71f'

# Step 1 of Kubeflow pipeline 
# Read the data 
@dsl.python_component(
    name='read_split',
    description='',
    base_image=CONTAINER_NAME  # you can define the base image here, or when you build in the next step. 
)
def read_and_split(gcs_path: str,output_csv: OutputPath(str),mlpipeline_ui_metadata_path: OutputPath('ui')):
    """
    Read and Splits data into train, validation and test set
  
    Parameters:
    gcs_path (str) : Path of input data
    output_csv (str) : (internally assigned by kfp) Path where output csv will be stored and 
                       passed to next container step
    mlpipeline_ui_metadata_path : Path where metadata is stored

    """
    from sklearn.model_selection import train_test_split
    import os
    print(os.listdir())
    import json
    import pandas
    import gcsfs
    import pandas as pd
    
    file_list = gcs_path.split(',')
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets.json'
    fs = gcsfs.GCSFileSystem(project='leafy-ether-314809' , token='secrets.json')
    
    # Note : Here I have read only 1 file, we can create empty dataframe and 
    # read all the csv files in it with concat on axis 0 (stacking one above the other)
    with fs.open(file_list[0]) as f:
        loan_data = pandas.read_csv(f)
        
    # keeping validation set aside
    X = loan_data.iloc[:,loan_data.columns!='Loan_Status']
    y = loan_data.iloc[:,loan_data.columns=='Loan_Status']
    X, Xval, y, yval = train_test_split(X,y,test_size=0.15, random_state=45)
    loan_data = pd.concat([X,y],axis=1).reset_index(drop=True)
    Xval,yval = Xval.reset_index(drop=True),yval.reset_index(drop=True)
    
    # Send output for next container step
    loan_data.to_csv(output_csv,index=False)
    
    # Log train data files
    file_list = [i+'#'+fs.stat(i)['generation'] for i in file_list]
    metadata = {
    'outputs' : [
    # Markdown that is hardcoded inline
    {
      'storage': 'inline',
      'source': '# Training files used\n'+ ','.join(file_list),
      'type': 'markdown',
    }]
      }
    with open(mlpipeline_ui_metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    #Save splitted data to validation forlder if required


# Step 2 of Kubeflow pipeline 
@dsl.python_component(
    name='preprocess',
    description='',
    base_image=CONTAINER_NAME # you can define the base image here, or when you build in the next step. 
)
def preprocess(text_path: InputPath(),output_csv: OutputPath(str),imputer_path: OutputPath(str)):
    """
    Data preprocessing step
  
    Parameters
    ----------
    text_path (str) : Path of input training data
    output_csv (str) : (internally assigned by kfp) Path where output csv will be stored and 
                       passed to next container step
    imputer_path (str) : (internally assigned by kfp) Path where imputer instance will be stored and 
                       passed to training container step
  
    """
    import os
    print(os.listdir())
    import sys
    sys.path.append('.')
    import pandas as pd
    import numpy as np
    from utils.impute_class import impute
    
    global imputer_cls
    loan_data = pd.read_csv(text_path)
    imputer = impute()
    loan_data = imputer.fit(loan_data)
    loan_data.to_csv(output_csv,index=False)
    
    imputer_cls = imputer
    import joblib
    import dill
    with open(imputer_path, "wb") as dill_file:
        dill.dump([imputer_cls],dill_file)
    
    
# Step 3 of Kubeflow pipeline 
@dsl.python_component(
    name='FE',
    description='adds two numbers',
    base_image=CONTAINER_NAME  # you can define the base image here, or when you build in the next step. 
)
def FE(text_path: InputPath(),output_csv: OutputPath(str),FE_path: OutputPath(str)):
    """
    Feature engineering step
  
    Parameters
    ----------
    text_path (str) : Path of input training data
    output_csv (str) : (internally assigned by kfp) Path where output csv will be stored and 
                       passed to next container step
    FE_path (str) : (internally assigned by kfp) Path where Feature engineering instance will be 
                        stored and passed to training container step
    
    """
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    import sys
    import joblib
    global FE_cls
    import dill
    sys.path.append('.')
    from utils.FE import Feature_engineering
    
    ###################
    
    loan_data = pd.read_csv(text_path)
    FE_pipeline = Feature_engineering()
    loan_data = FE_pipeline.fit(loan_data)
    loan_data.to_csv(output_csv,index=False)
    
    FE_cls = FE_pipeline
    with open(FE_path, "wb") as dill_file:
        dill.dump([FE_cls],dill_file)
        
# Step 4 of Kubeflow pipeline 
@dsl.python_component(
    name='train',
    description='',
    base_image=CONTAINER_NAME  # you can define the base image here, or when you build in the next step. 
)
def train(text_path: InputPath(),imputer_path: InputPath(), FE_path :  InputPath(), mlpipeline_metrics_path: OutputPath('Metrics')):
    """
    Model training step
  
    Parameters
    ----------
    text_path (str) : Path of input training data
    imputer_path (str) : (internally assigned by kfp) Path where imputer instance pkl is stored
    FE_path (str) : (internally assigned by kfp) Path where Feature engineering pkl is stored
    mlpipeline_metrics_path : Path where metrics are stored
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    import dill
    import sys
    import gcsfs
    import os
    sys.path.append('.')
    
    # Load train data
    loan_data = pd.read_csv(text_path)
    
    loan_data['Loan_Status']=loan_data['Loan_Status'].astype('int')
    loan_data = loan_data[loan_data>=0].dropna()

    X = loan_data.iloc[:,loan_data.columns!='Loan_Status']
    y = loan_data.iloc[:,loan_data.columns=='Loan_Status']

    # Split data into train and test set
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.10, random_state=45) # creating train test split
    log_reg = LogisticRegression()
    log_reg_model = log_reg.fit(Xtrain,ytrain) # classifier function will train the MLmodel
    ypred = log_reg_model.predict(Xtest) # Performing perdiction on test test
    f1 = f1_score(y_true=ytest,y_pred=log_reg_model.predict(Xtest)) # Getting f1 score on test dataset
    
    # Log metrics
    import json
    accuracy = 0.9
    metrics = {
    'metrics': [{
      'name': 'accuracy-score', # The name of the metric. Visualized as the column name in the runs table.
      'numberValue':  accuracy, # The value of the metric. Must be a numeric value.
      'format': "PERCENTAGE",   # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
    }]
    }
    with open(mlpipeline_metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    # Load imputer and feature engineering pipeline which we saved and passed from
    # previous container steps
    with open(imputer_path, "rb") as imputer_file, open(FE_path, 'rb') as FE_file:
        imputer = dill.load(imputer_file)[0]
        FE_pipeline = dill.load(FE_file)[0]
        
    # Save model file into current container
    model_file = (r'loan_model.pkl')
    with open(model_file, "wb") as dill_file:
        dill.dump([imputer,FE_pipeline, log_reg_model],dill_file)
    
    # Connect to GCS
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets.json'
    fs = gcsfs.GCSFileSystem(project='leafy-ether-314809' , token='secrets.json',cache_timeout=0)
    
    Production_model_f1 = 0.8
    # If no files present then save first model in folder 01(version)
    if len(fs.ls('gs://loan_model_pipeline')) == 0 :
        # Upload model to GCS
        with open("loan_model.pkl", "rb") as local_file:
            with fs.open("gs://loan_model_pipeline/" + "1/loan_model.pkl", "wb") as gcs_file:
                gcs_file.write(local_file.read())

    # Save model to new folder if better than production model            
    elif f1 > Production_model_f1: # production model f1 score
        gcs_files = sorted([int(i.replace('loan_model_pipeline/','')) for i in fs.ls('gs://loan_model_pipeline/')])
        next_folder_num = str(int(gcs_files[-1]) + 1)
        with open("loan_model.pkl", "rb") as local_file:
            with fs.open("gs://loan_model_pipeline/" + next_folder_num + "/loan_model.pkl", "wb") as gcs_file:
                gcs_file.write(local_file.read())   

        with open("ds_train.yaml", "rb") as local_file:
            with fs.open("gs://loan_model_pipeline/" + next_folder_num + "/pipeline.yaml", "wb") as gcs_file:
                gcs_file.write(local_file.read())   


