# rag-doc-assistant
A document assistant using RAG and vector DB

A document assistant using Retrieval-Augmented Generation (RAG) and a local vector database. Ask natural-language questions about a PDF and receive answers grounded exclusively in that document — no hallucinations from general knowledge.


How to run Kubernetes

1. Start MinikubeBash

    minikube start --driver=docker --memory=8g --cpus=4

2. Point Docker to Minikube’s Docker daemon (very important!)Bash

    eval $(minikube docker-env)

3. Build your Docker images inside MinikubeBash

    cd rag-doc-assistant          

4. Initialize TerraformBash

    terraform init

4.1 Plan

    terraform plan

5. Apply the configurationBash

    terraform apply -auto-approve

6. Check if everything is runningBash

    kubectl get pods -n rag-app
    kubectl get svc -n rag-app

7. Get the URL to access the appBash

    terraform output frontend_url

8. (Optional) See logsBash

    kubectl logs -f -n rag-app deployment/backend
    kubectl logs -f -n rag-app statefulset/ollama
