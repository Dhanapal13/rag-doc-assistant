# rag-doc-assistant
A document assistant using RAG, Ollama, LlamaIndex, Chroma DB

A document assistant using Retrieval-Augmented Generation (RAG) and a local vector database. Ask natural-language questions about a PDF and receive answers grounded exclusively in that document — no hallucinations from general knowledge.

System Architecture:

<img width="350" height="757" alt="image" src="https://github.com/user-attachments/assets/4a5105f0-c68b-4035-a6fb-348815bcb07e" />


<img width="273" height="346" alt="image" src="https://github.com/user-attachments/assets/91240ed6-0a01-4dd7-bc6a-e2089cb0e0c4" />

-------------------------------------------------------------------------------------

RAG Data Flow:

<img width="360" height="697" alt="image" src="https://github.com/user-attachments/assets/118d3f53-1f56-4aea-8a9f-fc5cf3eb66e4" />

---------------------------------------------------------------------------------------
How to run Kubernetes

1. Start MinikubeBash

    minikube start --driver=docker --memory=8g --cpus=4

2. Point Docker to Minikube’s Docker daemon (very important!)Bash

    eval $(minikube docker-env)

3. Build your Docker images inside MinikubeBash

    cd rag-doc-assistant          

4. Initialize TerraformBash

    terraform init

    terraform plan

5. Apply the configurationBash

    terraform apply -auto-approve

6. Check if everything is runningBash

    kubectl get pods -n rag-app
    kubectl get svc -n rag-app

7. Get the URL to access the appBash

    terraform output frontend_url

8. See logsBash

    kubectl logs -f -n rag-app deployment/backend
    kubectl logs -f -n rag-app statefulset/ollama
