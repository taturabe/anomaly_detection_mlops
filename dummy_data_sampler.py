import pandas as pd
import numpy as np
import datetime
import hashlib
import boto3
import os
import shutil
from tqdm import tqdm

start = '2022-09-01'
end = '2022-09-30'
LOCAL_TMP_DATA_DIR = 'anomaly_detection_mlops/data/data1'
S3_BUCKET = 's3://dummy-sensor-us-east-1'
FILE_FREQ = 'D'
DATA_FREQ = 'min'
NUM_EQUIPMENT = 30 # less than 1000
NUM_SENSOR = 5
ANOMALY_PROBABILITY = 0.02
SPIKE_INTENSITY = 2.
SPIKE_DATA_WIDTH = 10

anomaly_label_filename = "anomaly_labels.txt"
f = open(anomaly_label_filename, 'w')
f.write("equipment_id, sensor_id, start_time, end_time\n")

file_range = pd.date_range(start=start, end=end, freq=FILE_FREQ)

if os.path.exists(LOCAL_TMP_DATA_DIR):
    shutil.rmtree(LOCAL_TMP_DATA_DIR)
for i in range(NUM_EQUIPMENT):
    os.makedirs(os.path.join(LOCAL_TMP_DATA_DIR, "EQUIPMENT_%03d" % i))
    
for i in tqdm(range(NUM_EQUIPMENT)):
    for d in file_range:
        now = d
        next_time = d + datetime.timedelta(days=1)
        #print(f"processing between [{now}, {next_time})")
        min_range = pd.date_range(start=now, end=next_time, freq=DATA_FREQ)
        min_range = min_range[:-1]
    
        # make seed for reproducibility from timestamp {now}
        hash_str = "%03d" % i + str(now)
        hash_object = hashlib.md5(hash_str.encode('utf-8'))
        seed = np.frombuffer(hash_object.digest(), dtype='uint32')
        np.random.seed(seed=seed)
   
        val_arr = np.random.rand(len(min_range), NUM_SENSOR).astype('float16')
        df = pd.DataFrame(val_arr, index=min_range)
        df.columns = ["sensor_%02d" % i for i in range(NUM_SENSOR)]
        filename = os.path.join("EQUIPMENT_%03d" % i, str(now).replace(' ','_') +  ".csv")
    
        # impute spike noize with a set probability at random location
        if np.random.rand() < ANOMALY_PROBABILITY:
            #print("\tanomary detected!!!!")
            x_index = np.random.randint(df.shape[0]-SPIKE_DATA_WIDTH)
            #y_index = np.random.randint(df.shape[1])
            #df.iloc[x_index:(x_index+SPIKE_DATA_WIDTH), y_index] = np.random.randn(SPIKE_DATA_WIDTH).astype('float16') + SPIKE_INTENSITY
            df.iloc[x_index:(x_index+SPIKE_DATA_WIDTH), :] = np.random.randn(SPIKE_DATA_WIDTH, NUM_SENSOR).astype('float16') + SPIKE_INTENSITY
            
            # write to anomaly label file
            out_txt = f"{filename}, {i}, \
                        {str(df.iloc[x_index].name)},\
                        {str(df.iloc[x_index+SPIKE_DATA_WIDTH].name)}\n"
            f.write(out_txt)
        # save dataframe
        df.to_csv(os.path.join(LOCAL_TMP_DATA_DIR, filename))
f.close()        
    
# sync 
sync_command = f"aws s3 sync {LOCAL_TMP_DATA_DIR}/ {S3_BUCKET}/{LOCAL_TMP_DATA_DIR}/ --delete"
os.system(sync_command)

