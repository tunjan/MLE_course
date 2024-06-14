import json
import boto3

runtime_client = boto3.client("sagemaker-runtime")

endpoint_name = "2024-06-14-12-04"
input_data = {
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 1,
}

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(input_data),
)

result = json.loads(response["Body"].read().decode())
print(result)
