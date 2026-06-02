terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
    external = {
      source = "hashicorp/external"
      version = "~> 2.0"
    }
  }
}

provider "kubernetes" {
  config_path    = "~/.kube/config"
  config_context = "minikube"
}

variable "namespace" {
  default = "rag-app"
}

# ====================== NAMESPACE ======================
resource "kubernetes_namespace" "rag" {
  metadata {
    name = var.namespace
  }
}

# ====================== PERSISTENT VOLUMES ======================
resource "kubernetes_persistent_volume_claim" "chroma_pvc" {
  metadata {
    name      = "chroma-db-pvc"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = { storage = "5Gi" }
    }
    storage_class_name = "standard"
  }
}

resource "kubernetes_persistent_volume_claim" "ollama_pvc" {
  metadata {
    name      = "ollama-models-pvc"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }
  spec {
    access_modes = ["ReadWriteOnce"]
    resources {
      requests = { storage = "7Gi" }
    }
    storage_class_name = "standard"
  }
}

# ====================== OLLAMA ======================
resource "kubernetes_stateful_set" "ollama" {
  metadata {
    name      = "ollama"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }

  spec {
    service_name = "ollama"
    replicas     = 1

    selector {
      match_labels = { app = "ollama" }
    }

    template {
      metadata {
        labels = { app = "ollama" }
      }
      spec {
        container {
          image             = "ollama/ollama:latest"
          name              = "ollama"
          image_pull_policy = "IfNotPresent"

          port {
            container_port = 11434
          }

          # Health checks for Ollama
          startup_probe {
            http_get {
              path = "/api/tags"
              port = 11434
            }
            failure_threshold = 30
            period_seconds    = 10
          }

          readiness_probe {
            http_get {
              path = "/api/tags"
              port = 11434
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }

          liveness_probe {
            http_get {
              path = "/api/tags"
              port = 11434
            }
            initial_delay_seconds = 15
            period_seconds        = 20
          }

          volume_mount {
            mount_path = "/root/.ollama"
            name       = "ollama-models"
          }
        }
      }
    }

    volume_claim_template {
      metadata {
        name = "ollama-models"
      }
      spec {
        access_modes = ["ReadWriteOnce"]
        resources { requests = { storage = "10Gi" } }
        storage_class_name = "standard"
      }
    }
  }
}

resource "kubernetes_service" "ollama" {
  metadata {
    name      = "ollama"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }
  spec {
    selector = { app = "ollama" }
    port {
      port        = 11434
      target_port = 11434
    }
    type = "ClusterIP"
  }
}

# ====================== BACKEND ======================
resource "kubernetes_deployment" "backend" {
  metadata {
    name      = "backend"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = { app = "backend" }
    }

    template {
      metadata {
        labels = { app = "backend" }
      }
      spec {
        container {
          image             = "rag-doc-assistant-backend"
          name              = "backend"
          image_pull_policy = "Never"

          port {
            container_port = 8000
          }

          env {
            name  = "CHROMA_DB_DIR"
            value = "/app/chroma_db"
          }
          env {
            name  = "OLLAMA_BASE_URL"
            value = "http://ollama:11434"
          }

          # Health checks for Backend (FastAPI)
          readiness_probe {
            http_get {
              path = "/docs"          # or "/health" if you have a health endpoint
              port = 8000
            }
            initial_delay_seconds = 10
            period_seconds        = 10
          }

          liveness_probe {
            http_get {
              path = "/docs"
              port = 8000
            }
            initial_delay_seconds = 15
            period_seconds        = 20
          }

          volume_mount {
            mount_path = "/app/chroma_db"
            name       = "chroma-storage"
          }
        }

        volume {
          name = "chroma-storage"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim.chroma_pvc.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "backend" {
  metadata {
    name      = "backend-service"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }
  spec {
    selector = { app = "backend" }
    port {
      port        = 8000
      target_port = 8000
    }
    type = "ClusterIP"
  }
}

# ====================== FRONTEND ======================
resource "kubernetes_deployment" "frontend" {
  metadata {
    name      = "frontend"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }

  spec {
    replicas = 1

    selector {
      match_labels = { app = "frontend" }
    }

    template {
      metadata {
        labels = { app = "frontend" }
      }
      spec {
        container {
          image             = "rag-doc-assistant-frontend:latest"
          name              = "frontend"
          image_pull_policy = "Never"

          port {
            container_port = 3000
          }

          env {
            name  = "REACT_APP_API_URL"
            value = "http://backend-service.${kubernetes_namespace.rag.metadata[0].name}.svc.cluster.local:8000"
          }

          # Simple TCP probe for React (no HTTP health endpoint usually)
          readiness_probe {
            tcp_socket {
              port = 3000
            }
            initial_delay_seconds = 5
            period_seconds        = 10
          }

          liveness_probe {
            tcp_socket {
              port = 3000
            }
            initial_delay_seconds = 15
            period_seconds        = 20
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "frontend" {
  metadata {
    name      = "frontend-service"
    namespace = kubernetes_namespace.rag.metadata[0].name
  }
  spec {
    selector = { app = "frontend" }
    port {
      port        = 3000
      target_port = 3000
    }
    type = "NodePort"
  }
}

# ====================== OUTPUT ======================
output "frontend_url" {
  value = "http://${data.external.minikube_ip.result.ip}:${kubernetes_service.frontend.spec[0].port[0].node_port}"
}

data "external" "minikube_ip" {
  program = [
    "powershell",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "$ip = (minikube ip).Trim(); @{ip = $ip} | ConvertTo-Json -Compress"
  ]
}