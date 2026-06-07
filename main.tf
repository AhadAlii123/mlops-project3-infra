resource "minikube_cluster" "default" {
  cluster_name       = "mlops-local-cluster"
  driver             = "docker"
  nodes              = 1
  kubernetes_version = "v1.27.3"
}