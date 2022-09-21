import boto3
import configparser
import datetime

config = configparser.ConfigParser()
config.read('config.ini')

EXECUTION_ROLE = config['ROLE']['EXECUTION_ROLE']
SAGEMAKER_SUBMIT_DIRECTORY = config['PATH']['SAGEMAKER_SUBMIT_DIRECTORY']

now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d%H%M%S%f')
model_name = "IFmodel-" + now

client = boto3.client('sagemaker')

params = {
   "EnableNetworkIsolation": False,
   "ExecutionRoleArn": EXECUTION_ROLE,
   "ModelName": model_name,
   "PrimaryContainer": { 
      "ContainerHostname": "string",
      "Environment": { 
		"SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
		"SAGEMAKER_PROGRAM": "inference.py",
		"SAGEMAKER_REGION": "us-east-1",
		"SAGEMAKER_SUBMIT_DIRECTORY": SAGEMAKER_SUBMIT_DIRECTORY
      },
      "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
      "Mode": "SingleModel",
      "ModelDataUrl":"s3://taturabe-dataset/anomaly_detection_mlops/inference_test/model2.tar.gz" ,
   },
   "Tags": [ 
      { 
         "Key": "string",
         "Value": "string"
      }
   ],
}

print("creating model...")
client.create_model(**params)
print("model created!")
