SERVER_NAME = mlflow-server
CLIENT_NAME = docker-client
MLFLOW_NETWORK = mlflow-network


run:
	docker compose up -d

save:
	docker cp $(CLIENT_NAME):/home/appuser/figures .
	docker cp $(SERVER_NAME):/home/appuser/artifacts .
	docker cp $(SERVER_NAME):/home/appuser/mlruns .

delete:
	docker compose down
