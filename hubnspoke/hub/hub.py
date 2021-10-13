from pathlib import Path
cwd = str(Path.cwd())
print(cwd)

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
        self.client = None
        self.data = None
        self.model = None
        self.optimizer = None
        self.modelFile = os.path.join(modelpath, modelName)
        self.loc_weights = None
    

    def bootstrap(self):
        logger_extra['status'] = Stage.FEDERATION_INITIALIZATION_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            logger.info("bootstrapping...")
            buffer = BytesIO()
            if self.address in whitelist:
                logger.info("fl node is whitelisted")
                if os.path.isfile(modelFile):
                    logger.info(f"buffering the provided initial model {modelFile}...") 
                    checkpoint = t.load(modelFile)
                    t.save(checkpoint['weights'], buffer)
                else:
                    logger.info("initial model does not exist, initializing and buffering a new one...")
                    t.save(self.model.state_dict(), buffer)
                size = buffer.getbuffer().nbytes
                
                logger.info("sending the initial model...")
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

    def train(self, epochs):
        logger_extra['status'] = Stage.TRAINING_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
            self.data = {"epochs": epochs}
            buffer = BytesIO()
            t.save(self.data, buffer)
            size = buffer.getbuffer().nbytes

            logger.info(f"sending the training request for {epochs} local epochs...")
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
            checkpoint = self.gather()
            for k in checkpoint.keys():
                if k == "epoch":
                    #epochs = checkpoint['epoch']
                    logger.info(f"node's best epoch: {checkpoint['epoch']}") 
                elif k == "weights":
                    w = checkpoint['weights']
                    logger.info("copying weights...")
                    w_loc.append(copy.deepcopy(w))
                    logger.info("aggregating weights...")
                    w_glob = FedAvg(w_loc)
                elif k == "metric":
                    logger.info(f"node's best metric: {checkpoint['metric']}" )
                else:
                    logger.info(f"unknown data received from the node (unexpected key found: {k})")
            cpt = {#'epoch': 1, # to be determined
                'weights': w_glob#,
                #'metric': 0 # to be aggregated
                }
            t.save(cpt, modelFile)

            logger_extra['status'] = Stage.AGGREGATION_COMPLETED
            logger.info("aggregation completed")
        
    def test(self):
        logger_extra['status'] = Stage.TESTING_STARTED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
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

            logger.info("test report received")
            response_bytes = BytesIO(fl_response.para_response)    
            response_data = t.load(response_bytes, map_location='cpu')

            reportName = self.name.replace(' ','') + '.txt'
            reportFile = os.path.join(modelpath, reportName)
            logger.info(f"writing the test report in {reportFile}...")
            with open(reportFile, 'w') as f:
                f.write(response_data)

            logger_extra['status'] = Stage.TESTING_COMPLETED
            logger.info('report file created successfully')
    
    def stop(self):
        logger_extra['status'] = Stage.FEDERATION_COMPLETED
        logger_extra['trust_name'] = self.name

        if self.status() == "alive":
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
