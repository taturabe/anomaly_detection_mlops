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

SHINGLE_SIZE = 10
USE_SHINGLE = True


def shingle(data, shingle_size):
    if np.ndim(data) ==2:
        data = data[:,0]
    num_data = len(data)
    shingled_data = np.zeros((num_data - shingle_size, shingle_size))

    for n in range(num_data - shingle_size):
        shingled_data[n] = data[n : (n + shingle_size)]
    return shingled_data


def handler(event, context):
    
    body = event['body']

    s3 = boto3.resource('s3')
    s3.Bucket(body['data_bucket']).download_file(Filename=os.path.join('/tmp',
                                                     body['data_filename']),
                                                 Key=os.path.join(body['data_prefix'],
                                                     body['data_filename']
                                                     )
                                                 )


    df = pd.read_csv(os.path.join('/tmp',body['data_filename']), index_col=0)
    X = df.values

    if USE_SHINGLE:
        X = shingle(X, SHINGLE_SIZE)

    print("training IsolationForest")
    clf_params = body['clf_params']
    clf = IsolationForest(random_state=0, **clf_params)
    clf.fit(X)
    anomaly = clf.score_samples(X) # [-1, 1]
    anomaly = anomaly*(-0.5) + 0.5
   
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d%H%M%S%f')
    jobname = body['base_jobname'] + now
    write_dir = os.path.join('/tmp', jobname)
    if os.path.exists(write_dir):
        shutil.rmtree(write_dir)
    os.makedirs(write_dir)
    
    json_filename = 'training_job.json'
    csv_filename = 'anomaly.csv'
    model_filename = 'isolation_forest.joblib'

    with open(os.path.join(write_dir, json_filename), 'w') as f:
        json.dump(event, f)

    joblib.dump(clf, os.path.join(write_dir, model_filename))
    #np.save(os.path.join(write_dir, npy_filename), anomaly)
    np.savetxt(fname=os.path.join(write_dir, csv_filename), X=anomaly, fmt="%10.3f")

    
    for f in [json_filename, csv_filename, model_filename]:
        # upload file
        s3.Bucket(body['model_bucket']).upload_file(Filename=os.path.join(write_dir, f),
                                                     Key=os.path.join(body['model_prefix'],
                                                         jobname,
                                                         f
                                                         )
                                                     )

 
    shutil.rmtree(write_dir)
    os.remove(os.path.join('/tmp', body['data_filename']))

    anomaly_str = str(list(anomaly)).replace('[','').replace(']','')

    return_dict = {'anomaly':anomaly_str,
                    }

    return return_dict


