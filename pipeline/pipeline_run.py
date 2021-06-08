# Check gcs storage
import os
import json
import pandas
import gcsfs
import pandas as pd

# Connect to GCS
import gcsfs
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secrets.json'
fs = gcsfs.GCSFileSystem(project='leafy-ether-314809' , token='secrets.json',cache_timeout=0)
initial_length = len(fs.ls('gs://loan_model_pipeline'))

# Run kfp pipeline

import kfp
client = kfp.Client('https://2886795278-31380-shadow05.environments.katacoda.com/pipeline/')
run = client.create_run_from_pipeline_package(
        pipeline_file='pipeline/ds_train.yaml',
        arguments = {'gcs_path': 'gs://bucket-306/data/train/dataloan.csv' },experiment_name='MLOps_prod'

        )


client.wait_for_run_completion(run.run_id, 3600)

# If model was registered then save this pipeline to model's file
new_length = len(fs.ls('gs://loan_model_pipeline'))
if new_length > initial_length:
    gcs_files = [i.replace('loan_model_pipeline/','') for i in fs.ls('gs://loan_model_pipeline/')]
    folder_num = gcs_files[-1]
    with open("pipeline/ds_train.yaml", "rb") as local_file:
        with fs.open("gs://loan_model_pipeline/" + folder_num + "/model_pipeline.yaml", "wb") as gcs_file:
            gcs_file.write(local_file.read())

