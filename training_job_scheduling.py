import boto3
import glob
import os
from invoke_lambda_function import invoke_lambda
import configparser
import json
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

# s3のprefix中のフォルダを検索する
def get_all_s3_objects(s3_client, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3_client.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')


config = configparser.ConfigParser()
config.read('config.ini')

DATA_BUCKET = config['PATH']['DATA_BUCKET']
#DATA_PREFIX = config['PATH']['DATA_PREFIX'] # not used
#DATA_FILENAME = config['PATH']['DATA_FILENAME'] # not used
MODEL_BUCKET = config['PATH']['MODEL_BUCKET']
MODEL_PREFIX = config['PATH']['MODEL_PREFIX']
BASE_JOBNAME = config['PATH']['BASE_JOBNAME']

request_dict = {'body':{
                'data_bucket':DATA_BUCKET,
                #'data_prefix':DATA_PREFIX, # not used
                #'data_filename':filename,
                'model_bucket':MODEL_BUCKET,
                'model_prefix':MODEL_PREFIX,
                'base_jobname':BASE_JOBNAME,
                'clf_params':{
                    'n_estimators':50,
                    'max_samples':'auto',
                    },
                }
            }


function_name='sklearn-function-20220921140734'
search_prefix = 'anomaly_detection_mlops/data/data1/'
s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')


contents = list(get_all_s3_objects(boto3.client('s3'), Bucket=DATA_BUCKET, Prefix=search_prefix))
data_path = [c['Key'] for c in contents if c['Key'][-4:] == ".csv"]
data_prefix_filename_tuple_list = [(f[:f.rfind('/')], f[f.rfind('/')+1:]) for f in data_path]

def f(x):
    created = multiprocessing.Process()
    current = multiprocessing.current_process()
    print('running:', current.name, current._identity)
    print('created:', created.name, created._identity)
    return x * x

def lambda_exec_loop(pref_fn_tuple):
    request_dict['body']['data_prefix'] = pref_fn_tuple[0]
    request_dict['body']['data_filename'] = pref_fn_tuple[1]
    res = invoke_lambda(lambda_client, function_name, json.dumps(request_dict), InvocationType='Event')



# invoke lambda func. with multi process
if __name__ == '__main__':
    with Pool() as pool:
        pool.map(lambda_exec_loop, data_prefix_filename_tuple_list[:1])
        #pool.map(f, range(9))

