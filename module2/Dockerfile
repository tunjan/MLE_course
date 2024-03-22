FROM python:3.9.16-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY example.py image.png /app/

CMD ["python", "example.py"]
