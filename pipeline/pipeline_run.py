import kfp

client = kfp.Client('https://2886795272-31380-shadow05.environments.katacoda.com/pipeline/')

client.create_run_from_pipeline_package(
    pipeline_file='pipeline/ds_train.yaml',
    arguments = {'gcs_path': 'gs://bucket-306/data/train/dataloan.csv' },experiment_name='MLOps_prod'
    
)



import subprocess, sys, os

command = subprocess.run(['git', 'log','-1', '--pretty=%h'], capture_output=True)
print(command.stdout.decode('utf-8')[:-2])


