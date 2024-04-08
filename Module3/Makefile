CLIENT_NAME = client
SERVER_NAME = mlflow-server

deploy: 
	mlflow models build-docker --model-uri "models:/best_model/latest" --name "best_model"

build:
	docker build -t $(SERVER_NAME) -f Dockerfile.server .
	docker build -t $(CLIENT_NAME) -f Dockerfile.client .

run:
	docker run -d --name $(SERVER_NAME) -p 5000:5000 $(SERVER_NAME)
	docker run --rm --link mlflow-server:mlflow-server $(CLIENT_NAME)

