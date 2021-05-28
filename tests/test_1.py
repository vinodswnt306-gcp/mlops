import pytest
import sys, os.path
import joblib

py_scrpt_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
+ '/deployment/')
sys.path.append(py_scrpt_dir)
from score import *
import score

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
           resource_group='MLOps_resource_grp',
           auth=sp)

# Loading a model       
model_path = Model.get_model_path('Loan_model',version=None,_workspace=ws)
score.imputer ,score.FE_pipeline,score.model = joblib.load(model_path)

# Creating dummy collector to be used in scoring script, as we cannot initiate init() function from it.
class idc:
    def __int__(self):
        self.z=0
    def collect(self,arr):
        self.z=arr

score.inputs_dc = idc()

#load all init function materials
def test_func1(): # for run function
    # Create test data
    test_data = {
                     'Loan_ID':['LP001002'],'Gender':['Male'],
                     'Married':['No'],'Dependents':['0'],
                     'Education':['Graduate'],'Self_Employed':['No'],
                     'ApplicantIncome':[5849],'CoapplicantIncome':[0],
                     'LoanAmount':[128],'Loan_Amount_Term':[360],
                     'Credit_History':[1],'Property_Area':['Urban']
                }

    import json
    json_data = json.dumps({"data": test_data})

    # Call service for prediction
    z=run(json_data)
    assert z == '["Reject"]'
