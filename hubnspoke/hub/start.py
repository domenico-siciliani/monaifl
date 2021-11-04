from pathlib import Path
cwd = str(Path.cwd())

import sys
sys.path.append('.')
import os
from hub import Client, Stage
import logging
import concurrent.futures
import shutil
from datetime import datetime
import json
import boto3

FL_CLIENT_ENDPOINTS = json.loads(os.environ.get('FL_CLIENT_ENDPOINTS'))
ENVIRONMENT = os.environ.get('ENVIRONMENT')
MODEL_ID = os.environ.get('MODEL_ID')
GLOBAL_ROUNDS = int(os.environ.get('GLOBAL_ROUNDS'))

modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"
modelFile = os.path.join(modelpath, modelName)
w_loc = list() 

main_logger = logging.getLogger('main')
syslog = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s]-[%(model_id)s]-[%(status)s]-%(message)s')
syslog.setFormatter(formatter)
main_logger.setLevel(logging.INFO)
main_logger.addHandler(syslog)

# to write logs in a file
logpath = os.path.join(modelpath, 'hub.log')
fh = logging.FileHandler(logpath)
fh.setLevel(level=logging.DEBUG)
fh.setFormatter(formatter)
main_logger.addHandler(fh)

main_logger_extra = {'model_id': MODEL_ID, 'status': ''}
main_logger = logging.LoggerAdapter(main_logger, extra=main_logger_extra)


def model_spread_plan(client):
    # spreading model to nodes
    client.bootstrap()

def train_plan(client):
    # initializing training on nodes
    client.train()
    
def aggregate_plan(client):
    # aggregating models from nodes
    client.aggregate(w_loc)

def test_plan(client):
    # testing models on nodes
    client.test()
    
def stop_now(client):
    # asking nodes to stop
    client.stop()

def upload_results_in_s3_bucket(source_path: str, bucket_name: str = 'flip-uploaded-federated-data-bucket'):
    global main_logger
    main_logger_extra['status'] = Stage.UPLOAD_STARTED

    bucket_name += '-' + ENVIRONMENT
    
    main_logger.info('zipping the final model and the reports...')
    zip_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(cwd, "save", zip_name)
    shutil.make_archive(zip_path, 'zip', source_path)

    main_logger.info(f'uploading zip file {zip_path} to S3 bucket {bucket_name} in folder {MODEL_ID}...')
    bucket_zip_path = MODEL_ID + '/' + zip_name
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(zip_path + '.zip', bucket_name, bucket_zip_path + '.zip')
        main_logger_extra['status'] = Stage.UPLOAD_COMPLETED
        main_logger.info('upload completed')
    except Exception as e:
        main_logger_extra['status'] = Stage.UPLOAD_FAILED
        main_logger.error(e)


if __name__ == '__main__':
    main_logger_extra['status'] = Stage.FEDERATION_INITIALIZATION_STARTED

    main_logger.info("fl hub started")
    clients = [Client(fl_client['flclientendpoint'], fl_client['name']) for fl_client in FL_CLIENT_ENDPOINTS]

    for round in range(GLOBAL_ROUNDS):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(model_spread_plan, clients)
        
        main_logger_extra['status'] = Stage.FEDERATION_INITIALIZATION_COMPLETED
        main_logger.info("current model shared with all fl nodes")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(train_plan, clients)

        main_logger_extra['status'] = Stage.TRAINING_COMPLETED
        main_logger.info("model training completed for all fl nodes")    

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(aggregate_plan, clients)

        main_logger_extra['status'] = Stage.AGGREGATION_COMPLETED
        main_logger.info("aggregation completed for all fl nodes models")    

        main_logger.info(f"global round {round+1}/{GLOBAL_ROUNDS} completed")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(test_plan, clients)    
    
    main_logger_extra['status'] = Stage.TESTING_COMPLETED
    main_logger.info("global model tested in all fl nodes")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(stop_now, clients)  
    
    # all processes are excuted
    main_logger_extra['status'] = Stage.FEDERATION_COMPLETED
    main_logger.info(f"model training is completed across all sites and current global model is available at {modelFile}")

    upload_results_in_s3_bucket(modelpath)

    main_logger.info("fl hub terminated")