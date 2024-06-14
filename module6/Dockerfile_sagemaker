FROM python:3.11-slim

WORKDIR /opt/program

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


ENV PYTHONPATH=/opt/program
EXPOSE 8080

ENV SAGEMAKER_PROGRAM inference.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/program

# Define the entry point
ENTRYPOINT ["python", "inference.py"]
