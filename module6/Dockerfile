FROM python:3.11-slim

# Set up the working directory
WORKDIR /opt/program

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference script and model artifacts
COPY inference.py model.pkl scaler.pkl /opt/program/

ENV PYTHONPATH=/opt/program
EXPOSE 8080

# Set environment variables for SageMaker
ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/program

# Define the entry point
ENTRYPOINT ["python", "inference.py"]
