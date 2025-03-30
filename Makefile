# Variables
DOCKER_USERNAME = carlosrosado
PROJECT_NAME = edo_ml
SERVING_IMAGE = serving-edo-image
TRAINING_IMAGE = training-edo-image
TAG = latest
K8S_DIR = ./kubernetes

# Default target
.PHONY: all
all: build push deploy

# Build Docker images
.PHONY: build
build:
	@echo "Building Docker images..."
	docker build -t $(DOCKER_USERNAME)/$(SERVING_IMAGE):$(TAG) -f docker_server/Dockerfile .
	docker build -t $(DOCKER_USERNAME)/$(TRAINING_IMAGE):$(TAG) -f docker_training/Dockerfile .
    
# Push Docker images to the registry
.PHONY: push
push:
	@echo "Pushing Docker images to the registry..."
	docker push $(DOCKER_USERNAME)/$(SERVING_IMAGE):$(TAG)
	docker push $(DOCKER_USERNAME)/$(TRAINING_IMAGE):$(TAG)

# Deploy to Kubernetes
.PHONY: deploy
deploy:
	@echo "Deploying all Kubernetes resources..."
	kubectl apply -f $(K8S_DIR)/configmap.yaml
	kubectl apply -f $(K8S_DIR)/mlflow-deployment.yaml
	kubectl apply -f $(K8S_DIR)/mlflow-service.yaml
	kubectl apply -f $(K8S_DIR)/serving-deployment.yaml
	kubectl apply -f $(K8S_DIR)/serving-service.yaml
	kubectl apply -f $(K8S_DIR)/training-deployment.yaml
	kubectl apply -f $(K8S_DIR)/pvc.yaml

# Clean up Kubernetes resources
.PHONY: clean
clean:
	@echo "Cleaning up all Kubernetes resources..."
	kubectl delete -f $(K8S_DIR)/configmap.yaml || true
	kubectl delete -f $(K8S_DIR)/mlflow-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/mlflow-service.yaml || true
	kubectl delete -f $(K8S_DIR)/serving-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/serving-service.yaml || true
	kubectl delete -f $(K8S_DIR)/training-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/pvc.yaml || true

.PHONY: test
test:
	@echo "Running tests..."
	pytest

# Rebuild and redeploy everything
.PHONY: rebuild
rebuild: clean all
	@echo "Rebuilt and redeployed everything."