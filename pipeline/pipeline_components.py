import subprocess, sys, os
command = subprocess.run(['git', 'log','-1', '--pretty=%h'], capture_output=True)
os.environ["BASE_IMAGE_TAG"] = command.stdout.decode('utf-8').replace('\n','')
os.environ["CONTAINER_NAME"] = 'vinodswnt306/new_public_mlops:' + command.stdout.decode('utf-8').replace('\n','')

subprocess.run(['export', 'CONTAINER_NAME="$(git log -1 --pretty=%h)"'])

CONTAINER_NAME = os.environ["CONTAINER_NAME"]

from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile,OutputArtifact
from typing import NamedTuple
import kfp.dsl as dsl

@dsl.python_component(
    name='read_split',
    description='',
    base_image=CONTAINER_NAME  # you can define the base image here, or when you build in the next step. 
)
def read_and_split(gcs_path: str,output_csv: OutputPath(str)) -> str:
    '''Calculates sum of two arguments'''
    
    from sklearn.model_selection import train_test_split
    import os
    print(os.listdir())
    import json
    import pandas
    import gcsfs
    import pandas as pd
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets.json'
    
    fs = gcsfs.GCSFileSystem(project='leafy-ether-314809' , token='secrets.json')
    with fs.open(gcs_path) as f:
        loan_data = pandas.read_csv(f)
        
    # keeping validation set aside
    X = loan_data.iloc[:,loan_data.columns!='Loan_Status']
    y = loan_data.iloc[:,loan_data.columns=='Loan_Status']
    X, Xval, y, yval = train_test_split(X,y,test_size=0.15, random_state=45)
    loan_data = pd.concat([X,y],axis=1).reset_index(drop=True)
    Xval,yval = Xval.reset_index(drop=True),yval.reset_index(drop=True)
    
    # Send output for next container step
    loan_data.to_csv(output_csv,index=False)
    
    #Save splitted data to validation forlder if required
    return output_csv

        
@dsl.python_component(
    name='preprocess',
    description='',
    base_image=CONTAINER_NAME # you can define the base image here, or when you build in the next step. 
)
def preprocess(text_path: InputPath(),output_csv: OutputPath(str),imputer_path: OutputPath(str)):
    '''Calculates sum of two arguments'''
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
    
@dsl.python_component(
    name='FE',
    description='adds two numbers',
    base_image=CONTAINER_NAME  # you can define the base image here, or when you build in the next step. 
)
def FE(text_path: InputPath(),output_csv: OutputPath(str),FE_path: OutputPath(str)):
    '''Calculates sum of two arguments'''
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    import sys
    sys.path.append('.')
    from utils.FE import Feature_engineering
    
    ###################
    
    loan_data = pd.read_csv(text_path)
    FE_pipeline = Feature_engineering()
    loan_data = FE_pipeline.fit(loan_data)
    loan_data.to_csv(output_csv,index=False)
    
    global FE_cls
    FE_cls = FE_pipeline
    import joblib
    import dill
    with open(FE_path, "wb") as dill_file:
        dill.dump([FE_cls],dill_file)
        
    
@dsl.python_component(
    name='train',
    description='',
    base_image=CONTAINER_NAME  # you can define the base image here, or when you build in the next step. 
)
def train(text_path: InputPath(),imputer_path: InputPath(), FE_path :  InputPath()):
    '''Calculates sum of two arguments'''
    
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,plot_confusion_matrix
    import sys
    sys.path.append('.')
    
    loan_data = pd.read_csv(text_path)
    
    loan_data['Loan_Status']=loan_data['Loan_Status'].astype('int')
    loan_data = loan_data[loan_data>=0].dropna()

    X = loan_data.iloc[:,loan_data.columns!='Loan_Status']
    y = loan_data.iloc[:,loan_data.columns=='Loan_Status']

    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.10, random_state=45) # creating train test split
    log_reg = LogisticRegression()
    log_reg_model = log_reg.fit(Xtrain,ytrain) # classifier function will train the MLmodel
    ypred = log_reg_model.predict(Xtest) #Performing perdiction on test test

    f1 = f1_score(y_true=ytest,y_pred=log_reg_model.predict(Xtest)) # Getting f1 score on test dataset
    
    import dill
    with open(imputer_path, "rb") as imputer_file, open(FE_path, 'rb') as FE_file:
        imputer = dill.load(imputer_file)[0]
        FE_pipeline = dill.load(FE_file)[0]
        
    
    model_file = (r'model\loan_model.pkl')
    
    #     with open(imputer_path, "rb") as dill_file:
    #         dill.dump([imputer,log_reg_model],dill_file)
    
    # Check with production model and save it if better than it
    
    # Save model as well as its pipeline yaml file
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    