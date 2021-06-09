#This file compiles all the pipeline steps and saves yaml file
from pipeline_components import *
import kfp
import kfp.components as comp
import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp import components

####################################################################################

# Convert read_and_split function to a pipeline operation.
read_split_op = components.func_to_container_op(
    read_and_split,
    base_image=CONTAINER_NAME,
    packages_to_install=['pandas==1.1.4','gcsfs','scikit-learn'] # optional
)

# Convert preprocessing function to a pipeline operation.
preprocess_op = components.func_to_container_op(
    preprocess,
    base_image=CONTAINER_NAME,
    packages_to_install=['pandas==1.1.4','gcsfs','joblib','dill'] # optional
)

# Convert feature engineering function to a pipeline operation.
FE_op = components.func_to_container_op(
    FE,
    base_image=CONTAINER_NAME,
    packages_to_install=['pandas==1.1.4','gcsfs','scikit-learn','dill'] # optional
)

# Convert training function to a pipeline operation.
train_op = components.func_to_container_op(
    train,
    base_image=CONTAINER_NAME,
    packages_to_install=['pandas==1.1.4','gcsfs','scikit-learn','joblib','dill'] # optional
)

###################################################################################

@dsl.pipeline(
   name='Calculation pipeline',
   description='A toy pipeline that performs arithmetic calculations.'
)
def ds_pipeline(
   gcs_path: str,
                ):
    #Passing pipeline parameter and a constant value as operation arguments
    read_split = read_split_op(gcs_path) #Returns a dsl.ContainerOp class instance. 
    read_split.container.set_image_pull_policy('Always') # Because by default it uses cached image
    
    preprocess=preprocess_op(read_split.outputs['output_csv'])
    preprocess.container.set_image_pull_policy('Always')
    
    FE = FE_op(preprocess.outputs['output_csv'])
    FE.container.set_image_pull_policy('Always') 
    
    train = train_op(FE.outputs['output_csv'],preprocess.outputs['imputer'],FE.outputs['FE'])
    train.container.set_image_pull_policy('Always')

    
# Combine pipeline and save yaml file
kfp.compiler.Compiler().compile(
    pipeline_func=ds_pipeline,
    package_path='pipeline/ds_train.yaml')
