# Base Image: Use a Python image with TensorFlow pre-installed
FROM tensorflow/tensorflow

# Working directory 
WORKDIR /app

# Copy your project files
COPY example.py requirements.txt image.png /app/

# Install dependencies 
RUN pip install -r requirements.txt

# Expose a port for potential web interfaces later
EXPOSE 8000 

# Command to run inference when the container starts 
CMD ["python", "example.py"]
