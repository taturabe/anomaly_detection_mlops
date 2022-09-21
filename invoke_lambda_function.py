import numpy as np
import json
import pandas as pd
import boto3
from sklearn.ensemble import IsolationForest
import joblib
import os
import tarfile
import subprocess
import shutil
import datetime
import glob
import configparser

SHINGLE_SIZE = 10
USE_SHINGLE = True

config = configparser.ConfigParser()
config.read('config.ini')

DATA_BUCKET = config['PATH']['DATA_BUCKET']
DATA_PREFIX = config['PATH']['DATA_PREFIX']
DATA_FILENAME = config['PATH']['DATA_FILENAME']
MODEL_BUCKET = config['PATH']['MODEL_BUCKET']
MODEL_PREFIX = config['PATH']['MODEL_PREFIX']
BASE_JOBNAME = config['PATH']['BASE_JOBNAME']




def invoke_lambda(client, func_name, payload, **kwargs):
    res = client.invoke(
        FunctionName=func_name,
        Payload=payload,
        **kwargs)

    if 'InvocationType' in kwargs:
        if kwargs['InvocationType'] == 'Event':
            return


    res_payload = res['Payload'].read()
    res_json =json.loads(res_payload)
    arr = res_json['anomaly']
    arr = np.array([float(s) for s in arr.split(',')])
    return arr




