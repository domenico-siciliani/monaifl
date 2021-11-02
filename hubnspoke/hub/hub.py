from pathlib import Path
cwd = str(Path.cwd())

import os
import sys
sys.path.append('.')

import grpc
from common.monaifl_pb2_grpc import MonaiFLServiceStub
from common.monaifl_pb2 import ParamsRequest
from io import BytesIO
import torch as t
import os
import copy
import logging
from coordinator import FedAvg
import json

class Stage():
    FEDERATION_INITIALIZATION_STARTED = 'FEDERATION_INITIALIZATION_STARTED'
    FEDERATION_INITIALIZATION_COMPLETED = 'FEDERATION_INITIALIZATION_COMPLETED'
    TRAINING_STARTED = 'TRAINING_STARTED'
    TRAINING_COMPLETED = 'TRAINING_COMPLETED'
    AGGREGATION_STARTED = 'AGGREGATION_STARTED'
    AGGREGATION_COMPLETED = 'AGGREGATION_COMPLETED'
    TESTING_STARTED = 'TESTING_STARTED'
    TESTING_COMPLETED = 'TESTING_COMPLETED'
    FEDERATION_COMPLETED = 'FEDERATION_COMPLETED'
    UPLOAD_STARTED = 'UPLOAD_STARTED'
    UPLOAD_COMPLETED = 'UPLOAD_COMPLETED'
    UPLOAD_FAILED = 'UPLOAD_FAILED'


modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"
modelFile = os.path.join(modelpath, modelName)
whitelist = set()

logger = logging.getLogger('federated_process')
syslog = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s]-[%(model_id)s]-[%(status)s]-[%(trust_name)s]-%(message)s')
syslog.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(syslog)

logger_extra = {'model_id': os.environ.get('MODEL_ID'), 'status':'', 'trust_name':''}
logger = logging.LoggerAdapter(logger, extra=logger_extra)

class Client():
    def __init__(self, address, name):
        self.address = address
        whitelist.add(address)
        self.name = name
        self.data = None
        self.model = None
        self.modelFile = os.path.join(modelpath, modelName)
        reportName = self.name.replace(' ','') + '.json'
        self.reportFile = os.path.join(modelpath, reportName)
    

    def bootstrap(self):
        logger_extra['status'] = Stage.FEDERATION_INITIALIZATION_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            try:
                logger.info("starting model sharing...")
                buffer = BytesIO()
                if self.address in whitelist:
                    logger.info("fl node is whitelisted")
                    if os.path.isfile(modelFile):
                        logger.info(f"buffering the current model {modelFile}...") 
                        checkpoint = t.load(modelFile)
                        t.save(checkpoint['weights'], buffer)
                    else:
                        logger.info("initial model does not exist, initializing and buffering a new one...")
                        t.save(self.model.state_dict(), buffer)
                    size = buffer.getbuffer().nbytes
                    
                    logger.info("sending the current model...")
                    opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
                    self.channel = grpc.insecure_channel(self.address, options = opts)
                    client = MonaiFLServiceStub(self.channel)
                    fl_request = ParamsRequest(para_request=buffer.getvalue())
                    fl_response = client.ModelTransfer(fl_request)

                    logger.info("answer received")
                    response_bytes = BytesIO(fl_response.para_response)
                    response_data = t.load(response_bytes, map_location='cpu')

                    logger_extra['status'] = Stage.FEDERATION_INITIALIZATION_COMPLETED
                    logger.info(f"returned status: {response_data}") # Model received OR Error
                else:
                    logger.error("fl node is not whitelisted. Please contact admin for permissions")
            except grpc.RpcError as rpc_error:
                logger.info(rpc_error.code())
                logger.info("returned status: dead")

    def train(self):
        logger_extra['status'] = Stage.TRAINING_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            try:
                self.data = {"id": "server"} # useless
                buffer = BytesIO()
                t.save(self.data, buffer)
                size = buffer.getbuffer().nbytes

                logger.info(f"sending the training request...")
                opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
                self.channel = grpc.insecure_channel(self.address, options = opts)
                client = MonaiFLServiceStub(self.channel)
                fl_request = ParamsRequest(para_request=buffer.getvalue())
                fl_response = client.MessageTransfer(fl_request)

                logger.info("answer received")
                response_bytes = BytesIO(fl_response.para_response)
                response_data = t.load(response_bytes, map_location='cpu')

                logger_extra['status'] = Stage.TRAINING_COMPLETED
                logger.info(f"returned status: {response_data}") # Training completed 
            except grpc.RpcError as rpc_error:
                logger.info(rpc_error.code())
                logger.info("returned status: dead")
    
    def status(self):
        try:
            self.data = {"check": 'check'}
            buffer = BytesIO()
            t.save(self.data, buffer)
            size = buffer.getbuffer().nbytes

            logger.info("checking fl node status...")
            opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
            self.channel = grpc.insecure_channel(self.address, options = opts)
            client = MonaiFLServiceStub(self.channel)
            fl_request = ParamsRequest(para_request=buffer.getvalue())
            fl_response = client.NodeStatus(fl_request)

            response_bytes = BytesIO(fl_response.para_response)
            response_data = t.load(response_bytes, map_location='cpu')
            logger.info(f"returned status: {response_data}")
            return response_data
        except:
            logger.info("returned status: dead")
            return 'dead'
  
    def gather(self):
        self.data = {"id": "server"} # useless
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info("sending the request for the trained model...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.TrainedModel(fl_request)

        logger.info("received the trained model")
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        return response_data

    def aggregate(self, w_loc):
        logger_extra['status'] = Stage.AGGREGATION_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            try:
                checkpoint = self.gather()
                result_file_dict = dict()
                for k in checkpoint.keys():
                    if k == "epoch":
                        logger.info(f"local epochs: {checkpoint[k]}") 
                    elif k == "weights":
                        w = checkpoint['weights']
                        logger.info("copying weights...")
                        w_loc.append(copy.deepcopy(w))
                        logger.info("aggregating weights...")
                        w_glob = FedAvg(w_loc)
                    elif k == "val_mean_dice_scores":
                        logger.info(f"validation mean dice scores: {checkpoint[k]}" )
                        result_file_dict[k] = checkpoint[k]
                    elif k == "train_loss_values":
                        logger.info(f"training loss values: {checkpoint[k]}" )
                        result_file_dict[k] = checkpoint[k]
                    else:
                        logger.info(f"unknown data received from the node (unexpected key found: {k})")
                cpt = {#'epoch': 1, # to be determined
                    'weights': w_glob#,
                    #'metric': 0 # to be aggregated
                    }
                t.save(cpt, modelFile)

                logger.info(f"writing training results in {self.reportFile}...")
                if not Path(self.reportFile).exists():
                    initial_reportFile = dict()
                    with open(self.reportFile, 'w') as f:
                        # each element of a list will be a list containing the values of a single local epoch
                        for key in result_file_dict.keys():
                            initial_reportFile[key] = [result_file_dict[key]]
                        json.dump(initial_reportFile, f)
                else:
                    with open(self.reportFile, 'r+') as f:
                        reportFile_dict = json.load(f)
                        for key in reportFile_dict.keys():
                            reportFile_dict[key].append(result_file_dict[key])
                        f.seek(0)
                        json.dump(reportFile_dict, f)

                logger_extra['status'] = Stage.AGGREGATION_COMPLETED
                logger.info("aggregation completed")
            except grpc.RpcError as rpc_error:
                logger.info(rpc_error.code())
                logger.info("returned status: dead")
        
    def test(self):
        logger_extra['status'] = Stage.TESTING_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            try:
                buffer = BytesIO()
                checkpoint = t.load(modelFile)
                t.save(checkpoint['weights'], buffer)
                size = buffer.getbuffer().nbytes

                logger.info("sending the test request...")
                opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
                self.channel = grpc.insecure_channel(self.address, options = opts)
                client = MonaiFLServiceStub(self.channel)
                fl_request = ParamsRequest(para_request=buffer.getvalue())
                fl_response = client.ReportTransfer(fl_request)

                logger.info("test results received")
                response_bytes = BytesIO(fl_response.para_response)    
                response_data = t.load(response_bytes, map_location='cpu')
                logger.info(f"test dice scores: {response_data['test_dice_scores']}")
                
                logger.info(f"writing test results in {self.reportFile}...")
                with open(self.reportFile, 'r+') as f:
                    result_file_dict = json.load(f)
                    result_file_dict['test_dice_scores'] = response_data['test_dice_scores']
                    f.seek(0)
                    json.dump(result_file_dict, f, indent = 4)

                logger_extra['status'] = Stage.TESTING_COMPLETED
                logger.info('report file created successfully')
            except grpc.RpcError as rpc_error:
                logger.info(rpc_error.code())
                logger.info("returned status: dead")
    
    def stop(self):
        logger_extra['status'] = Stage.FEDERATION_COMPLETED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            try:
                self.data={"stop":"yes"} # useless
                buffer = BytesIO()
                t.save(self.data, buffer)
                size = buffer.getbuffer().nbytes

                logger.info("sending the stop message...")
                opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
                self.channel = grpc.insecure_channel(self.address, options = opts)
                client = MonaiFLServiceStub(self.channel)
                fl_request = ParamsRequest(para_request=buffer.getvalue())
                fl_response = client.StopMessage(fl_request)

                logger.info("received the node status")
                response_bytes = BytesIO(fl_response.para_response)    
                response_data = t.load(response_bytes, map_location='cpu')
                logger.info(f"returned status: {response_data['reply']}")
            except grpc.RpcError as rpc_error:
                logger.info(rpc_error.code())
                logger.info("returned status: dead")