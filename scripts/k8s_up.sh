#!/usr/bin/env bash
# Create the kind cluster and install Strimzi, KEDA, kube-prometheus-stack, and DriftGuard
# (SPEC 8.1). Refuses to start while the Compose stack or another kind cluster is running.
set -euo pipefail

CLUSTER=driftguard
STRIMZI_VERSION=0.45.0
KEDA_VERSION=2.16.1
KPS_VERSION=66.3.1
export KUBECONFIG="${PWD}/.kube/config"
mkdir -p .kube

if kind get clusters 2> /dev/null | grep -q .; then
  echo "A kind cluster already exists: $(kind get clusters | tr '\n' ' '). Run 'make k8s-down'." >&2
  exit 1
fi
./scripts/check_stack_free.sh k8s

set -a
# shellcheck disable=SC1091
source .env
set +a
: "${ADMIN_TOKEN:?ADMIN_TOKEN must be set in .env}"
: "${IP_HASH_SALT:?IP_HASH_SALT must be set in .env}"
: "${GF_ADMIN_PASSWORD:?GF_ADMIN_PASSWORD must be set in .env}"

kind create cluster --config deploy/k8s/kind-config.yaml --kubeconfig "${KUBECONFIG}"
docker build -f docker/app.Dockerfile -t driftguard-app:dev .
kind load docker-image driftguard-app:dev --name "${CLUSTER}"

helm repo add strimzi https://strimzi.io/charts/ --force-update
helm repo add kedacore https://kedacore.github.io/charts --force-update
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update

kubectl create namespace driftguard
kubectl create namespace monitoring
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --version "${KPS_VERSION}" --namespace monitoring \
  -f deploy/k8s/kube-prometheus-values.yaml \
  --set grafana.adminPassword="${GF_ADMIN_PASSWORD}" --wait --timeout 10m
helm install strimzi strimzi/strimzi-kafka-operator --version "${STRIMZI_VERSION}" \
  --namespace driftguard --set watchNamespaces="{driftguard}" --wait
helm install keda kedacore/keda --version "${KEDA_VERSION}" --namespace keda --create-namespace \
  --wait

kubectl apply -f deploy/k8s/strimzi/kafka.yaml
kubectl wait kafka/driftguard -n driftguard --for=condition=Ready --timeout=10m
kubectl apply -f deploy/k8s/strimzi/topics.yaml

helm install driftguard deploy/k8s/charts/driftguard --namespace driftguard \
  -f deploy/k8s/values-kind.yaml \
  --set secrets.adminToken="${ADMIN_TOKEN}" --set secrets.ipHashSalt="${IP_HASH_SALT}" \
  --wait --timeout 20m
kubectl apply -f deploy/k8s/keda/scaledobject.yaml
echo "DriftGuard is running on kind: serving http://127.0.0.1:8001, Prometheus :9090, Grafana :3000"
