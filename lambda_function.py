
#CODE THAT RUNS IN AWS LAMBDA TO TRIGGER THE RETRAINING WHEN NEW DATA ARRIVES

import json
import boto3
import urllib.parse

ssm = boto3.client("ssm")

EC2_INSTANCE_ID = "i-03b72114df8258c43"

def lambda_handler(event, context):
    record = event["Records"][0]

    bucket = record["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])

    s3_path = f"s3://{bucket}/{key}"

    response = ssm.send_command(
        InstanceIds=[EC2_INSTANCE_ID],
        DocumentName="AWS-RunShellScript",
        Parameters={
            "commands": [
                "source /home/ec2-user/mlenv/bin/activate",
                "export MLFLOW_TRACKING_URI=http://127.0.0.1:5000",
                f"aws s3 cp {s3_path} /tmp/new_data.csv",
                "python /home/ec2-user/jobs/retrain_prophet.py /tmp/new_data.csv"
            ]
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps("EC2 job triggered successfully")
    }
