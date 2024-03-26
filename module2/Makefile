IMAGE_NAME = image-classifier
container_id=$(shell docker ps -alq)

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it $(IMAGE_NAME)

#Missing separator
all: build run

save:
	docker cp $(container_id):/app/output/mnist_model.keras model.keras
	docker cp $(container_id):/app/output/test_predictions.csv test_predictions.csv

clean:
	docker rmi $(IMAGE_NAME)

.PHONY: build run all save clean
