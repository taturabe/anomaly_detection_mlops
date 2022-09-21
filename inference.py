from sklearn.externals import joblib
import os
import numpy asNp

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "isolation_forest.joblib")
    return clf

	if request_content_type == "application/python-pickle":
	        array = np.load(StringIO(request_body))
	        return array
	    else:
	        # Handle other content-types here or raise an Exception
	        # if the content type is not supported.
	        pass


def predict_fn(input_data, model):
    prediction = model.score_samples(input_data)
    return np.array(prediction)

def output_fn(prediction, content_type):
	anomaly = prediction*(-0.5) + 0.5 # normalize to [0,1]
	return anomaly	
