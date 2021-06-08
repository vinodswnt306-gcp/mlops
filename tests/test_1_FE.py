import pytest
import sys, os.path

py_scrpt_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(py_scrpt_dir)
from utils.FE import *

FE_test_inputs = pd.read_csv(r'.\data\FE_test_inputs.csv')

def test_func1(): # for run function
    global FE_test_inputs
    FE_pipeline = Feature_engineering()
    FE_test_output = FE_pipeline.fit(FE_test_inputs)
    
    number_of_non_numeric_columns = FE_test_output.select_dtypes(exclude=['int64','float64']).shape[1]
    assert number_of_non_numeric_columns == 0
    
    
