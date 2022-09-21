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

bucket = 'taturabe-dataset'
data_path = "data/EQUIPMENT_029/2022-09-29_00:00:00.csv"
model_dir = 'anomaly_detection_mlops/isolation_forest'
SHINGLE_SIZE = 10
USE_SHINGLE = True
base_job_name = 'isolation_forest220914-2'

def shingle(data, shingle_size):
    if np.ndim(data) ==2:
        data = data[:,0]
    num_data = len(data)
    shingled_data = np.zeros((num_data - shingle_size, shingle_size))

    for n in range(num_data - shingle_size):
        shingled_data[n] = data[n : (n + shingle_size)]
    return shingled_data


def handler(event, context):
    model_name = 'isolation_forest.joblib'

    s3 = boto3.resource('s3')

    req = json.loads(event)
    s3_path = req['body']['s3_path']
    print(s3_path)
    df = pd.read_csv(s3_path)
    X = df.values
    X = np.sin(np.linspace(0,np.pi * 8 , 1000)).reshape(-1,1) + 2.
    X[100:110,0] = np.random.rand(10) +1.

    if USE_SHINGLE:
        X = shingle(X, SHINGLE_SIZE)

    print("training IsolationForest")
    clf = IsolationForest(random_state=0).fit(X)
    anomaly = clf.score_samples(X) # [-1, 1]
    anomaly = anomaly*(-0.5) + 0.5
   
    if os.path.exists(base_job_name):
        shutil.rmtree(base_job_name)
    os.makedirs(base_job_name)

    with open(os.path.join(base_job_name, 'training_job.json'), 'w') as f:
        json.dump(req, f)

    joblib.dump(clf, os.path.join(base_job_name, model_name))
    exec_str = f"aws s3 sync {base_job_name}/ s3://{bucket}/{model_dir}/{base_job_name}/"
    subprocess.call(exec_str, shell=True)
    



    return anomaly, clf

request_dict = {'body':{'s3_path':data_path, }}
request_json_str = json.dumps(request_dict)
anomaly, model = handler(request_json_str, 'context')
