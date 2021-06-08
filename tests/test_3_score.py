import pytest
import sys, os.path
import pandas as pd
import numpy as np
import joblib
from azureml.core.model import Model
from azureml.core import Workspace, Datastore, Dataset

py_scrpt_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(py_scrpt_dir)
from utils.impute_class import impute
from utils.FE import Feature_engineering


# Connect to workspace to get model
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

sp = ServicePrincipalAuthentication(tenant_id="a146078e-7f44-4ce0-91f0-45354d595be9", # tenantID
                                    service_principal_id="468ccf20-612d-4831-a533-62657e1b5ac9", # clientId
                                    service_principal_password="C0gF9ZJItm14TIaqsRIXBP1Ww3Lc.~T.r9") # clientSecret

ws = Workspace.get(name="MLOps_workspace",
           subscription_id='73bea878-9593-4d00-9e9b-56195ada7075',
           resource_group='MLOps_resource_grp',auth=sp)

# Loading a model       
model_path = Model.get_model_path('Loan_model',version=None,_workspace=ws)
imputer,FE_pipeline,model = joblib.load(model_path)

# Loading validation dataset
validation_dataset = Dataset.get_by_name(ws, "Loan_validation_data").to_pandas_dataframe()


# Xval,yval
def predict_for_validation(val_set):
    
    # Transform the data
    data = imputer.transform(val_set)
    
    # FE pipeline
    loan_data_OHE = FE_pipeline.transform(data)

    # Predictions
    predictions = model.predict(loan_data_OHE)
    
    return predictions

def test_func1(): # for run function
    assert (predict_for_validation(validation_dataset.iloc[:,:-1]) == validation_dataset.y_val_pred).all()
