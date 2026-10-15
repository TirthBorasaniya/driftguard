#!/usr/bin/env bash
# Delete the DriftGuard kind cluster and everything in it.
set -euo pipefail
export KUBECONFIG="${PWD}/.kube/config"
kind delete cluster --name driftguard
