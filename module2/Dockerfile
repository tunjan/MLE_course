FROM python:3.9.16-slim

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser

WORKDIR /app


RUN mkdir -p /app/output && chown -R appuser:appuser /app/output

# Copy requirements first to take advantage of Docker's caching mechanism
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY example.py /app/

# Switch to the non-root user
USER appuser

CMD ["python", "example.py"]
