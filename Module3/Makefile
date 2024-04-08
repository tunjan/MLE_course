SERVER_NAME = mlflow-server
CLIENT_NAME = docker-client
MLFLOW_NETWORK=mlflow-network

network:
	docker network create $(MLFLOW_NETWORK)

server:
	docker build -t $(SERVER_NAME) -f Dockerfile.server .
	docker run -d --name $(SERVER_NAME) --network $(MLFLOW_NETWORK) $(SERVER_NAME)

client:
	docker build -t $(CLIENT_NAME) -f Dockerfile.client .
	docker run --name $(CLIENT_NAME) --network $(MLFLOW_NETWORK) $(CLIENT_NAME)

save:
	docker cp $(CLIENT_NAME):/home/appuser/figures .
	docker cp $(SERVER_NAME):/home/appuser/artifacts .
	docker cp $(SERVER_NAME):/home/appuser/mlruns .

stop:
	docker stop $(shell docker ps -a -q)
	docker rm $(shell docker ps -a -q)

deploy: 
	$(shell export MLFLOW_TRACKING_URI=http://127.0.0.1:5000)
	mlflow models build-docker --model-uri "models:/best_model/latest" --name "best_model"
