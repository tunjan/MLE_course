docker build -t my-mlflow-server -f Dockerfile.server .
docker build -t my-mlflow-client -f Dockerfile.client .
docker run -d --name mlflow-server -p 5000:5000 my-mlflow-server
docker run --name mlflow-client --link mlflow-server:mlflow-server my-mlflow-client
