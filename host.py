import sagemaker
from sagemaker.sklearn.model import SKLearnModel

sagemaker_session = sagemaker.local.LocalSession() # sagemaker.Session()から変更

role="AmazonSageMaker-ExecutionRole-20220524T162997"

sklearn_model = SKLearnModel(model_data="s3://taturabe-dataset/anomaly_detection_mlops/model/isolation_forest20220921111613",
                             role=role,
                             entry_point="inference.py",
                             framework_version="0.23-1")

predictor = sklearn_model.deploy(instance_type="local", initial_instance_count=1)
