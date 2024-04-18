SERVER_NAME = mlflow-server
CLIENT_NAME = docker-client
MLFLOW_NETWORK = mlflow-network


run:
	docker-compose up -d

save:
	docker cp $(CLIENT_NAME):/home/appuser/figures .
	docker cp $(SERVER_NAME):/home/appuser/artifacts .
	docker cp $(SERVER_NAME):/home/appuser/mlruns .

deploy: 
	$(shell export MLFLOW_TRACKING_URI=http://127.0.0.1:5000)
	mlflow models build-docker --model-uri "models:/best_model/latest" --name "best_model"

delete:
	docker-compose down
