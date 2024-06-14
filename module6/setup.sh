#!/bin/bash

# Check if AWS access key and secret access key are provided
if [ -z "$1" ] || [ -z "$2" ]; then
	echo "Error: AWS access key and secret access key are required."
	echo "Usage: $0 <access_key_id> <secret_access_key>"
	exit 1
fi

# Configure AWS credentials
aws configure set aws_access_key_id "$1"
aws configure set aws_secret_access_key "$2"
aws configure set default.region eu-north-1

# Check if the configuration was successful
if ! aws sts get-caller-identity --query Account --output text >/dev/null 2>&1; then
	echo "Error: Failed to configure AWS credentials."
	exit 1
fi

echo "AWS credentials configured successfully."

# Check if Python 3 is installed
if ! command -v python3 >/dev/null 2>&1; then
	echo "Error: Python 3 is not installed."
	exit 1
fi

# Run the Python script
python3 endpoint-test.py

# Check if the Python script ran successfully
if [ $? -ne 0 ]; then
	echo "Error: Python script failed to run successfully."
	exit 1
fi

echo "Script executed successfully."
