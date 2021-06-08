import pytest
import sys, os.path

py_scrpt_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(py_scrpt_dir)
from utils.impute_class import *

loan_data = pd.read_csv(r'.\data\dataloan.csv')

def test_func1(): # for run function
    global loan_data
    imputer = impute()
    loan_data = imputer.fit(loan_data)
    number_of_columns_with_NA = loan_data.isna().sum()[loan_data.isna().sum() > 0].shape[0]

    assert number_of_columns_with_NA == 0
    
    
