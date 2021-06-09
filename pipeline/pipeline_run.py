
# This file runs a saved pipeline and also if new model is registered then it saves 
# its compiled pipeline file yaml with it

# Check gcs storage
import os
import json
import pandas
import gcsfs
import pandas as pd

# Run kfp pipeline
import kfp
# Set up Kubeflow client using its url
client = kfp.Client('https://2886795272-31380-shadow05.environments.katacoda.com/pipeline/')
run = client.create_run_from_pipeline_package(
        pipeline_file='pipeline/ds_train.yaml',
        arguments = {'gcs_path': 'gs://bucket-306/data/train/dataloan.csv' },experiment_name='MLOps_prod'
        )

client.wait_for_run_completion(run.run_id, 3600)
if client.get_run(run.run_id).run.status == 'Succeeded':
    print("completed")
else:
    print("job failed")
