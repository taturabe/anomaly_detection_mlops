import boto3
import numpy as np

X = np.random.randn(100)
payload = str(list(X)).replace('[','').replace(']','')

client = boto3.client('sagemaker-runtime')
endpoint_name = "test-endpoint-sklearn2"
endpoint_name = "sagemaker-scikit-learn-2022-09-21-11-25-07-672"
print("invoking endpoint")

res = client.invoke_endpoint(
    EndpointName = endpoint_name,
    ContentType="text/csv",
    Accept = "text/csv",
    Body = payload)

print("endpoint invoked!")
