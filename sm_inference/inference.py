import joblib
import os
from io import BytesIO, StringIO
import numpy as np

SHINGLE_SIZE = 10 # TODO: to be set externally

def model_fn(model_dir):
    print("test")
    clf = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
    return clf

def shingle(data, shingle_size):
    if np.ndim(data) ==2:
        data = data[:,0]
    num_data = len(data)
    shingled_data = np.zeros((num_data - shingle_size, shingle_size))

    for n in range(num_data - shingle_size):
        shingled_data[n] = data[n : (n + shingle_size)]
    return shingled_data



def input_fn(request_body, request_content_type):
    if request_content_type == "application/python-pickle":
        array = np.load(StringIO(request_body))
    elif request_content_type == "text/csv":
        array = np.genfromtxt(StringIO(request_body), delimiter=",")
    elif request_content_type == "application/x-npy":
        buf = BytesIO(request_body).read()
        array = np.frombuffer(buf, dtype=np.dtype('float64'), offset=128)
    else:
        raise ValueError("Invalid content type")
    print(array)
    array = shingle(array, SHINGLE_SIZE)
    return array


def predict_fn(input_data, model):
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1,1)
    print(input_data)
    prediction = model.score_samples(input_data)
    return np.array(prediction)

def output_fn(prediction, content_type):
    anomaly = prediction*(-0.5) + 0.5 # normalize to [0,1]
    anomaly_str = str(list(anomaly)).replace('[','').replace(']','')
    print(anomaly_str)

    return anomaly_str