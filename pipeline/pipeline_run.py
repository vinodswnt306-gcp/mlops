import kfp

client = kfp.Client('https://2886795272-31380-shadow05.environments.katacoda.com/pipeline/')

client.create_run_from_pipeline_package(
    pipeline_file='pipeline/ds_train.yaml',
    arguments = {'gcs_path': 'gs://bucket-306/data/train/dataloan.csv' },experiment_name='MLOps_prod'
    
)