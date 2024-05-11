# Use a slim-buster base for smaller size and faster builds
FROM python:3.11-slim-buster

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to utilize Docker layer caching
COPY requirements.txt .

# Install dependencies; use a virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY src /app/src


# Copy artifacts (consider if these change frequently, as it might affect caching)
COPY artifacts /app/artifacts
COPY output /app/output
COPY input /app/input

# Expose port 5000 for the Flask application
EXPOSE 5000

# Run the Flask application using the virtual environment
CMD ["/opt/venv/bin/python", "-m", "src.titanic_project.online.app"] 
