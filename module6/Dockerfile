# Use the official Python base image
FROM python:3.9-slim
# Install AWS CLI and boto3
RUN apt-get update && \
  apt-get install -y awscli && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  pip install boto3

# Set the working directory
WORKDIR /app

# Copy the Bash script and Python script into the container
COPY setup.sh .
COPY endpoint_test.py .

# Make the entrypoint script executable
RUN chmod +x setup.sh

# Set the entrypoint script
ENTRYPOINT ["./setup.sh"]
