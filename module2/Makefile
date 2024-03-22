# Variables
IMAGE_NAME = image-classifier

build:
	docker build -t $(IMAGE_NAME) .

# Run the inference script inside the container
run:
	docker run -it $(IMAGE_NAME)

# A convenient target for both building and running
all: build run

# Clean up the Docker image (optional)
clean:
	docker rmi $(IMAGE_NAME)
