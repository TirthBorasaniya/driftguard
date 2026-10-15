#!/usr/bin/env bash
# Refuse to start a stack while another Compose project or kind cluster is running (CLAUDE.md).
set -euo pipefail

mode="${1:-compose}"
project="$(basename "$(pwd)")"

others="$(docker ps --format '{{.Label "com.docker.compose.project"}} {{.Names}}' \
  | awk -v p="$project" '$1 != p {print $2}')"
if [[ -n "${others}" ]]; then
  echo "Other containers are running:" >&2
  echo "${others}" >&2
  echo "Stop them first; DriftGuard runs alone." >&2
  exit 1
fi
if command -v kind > /dev/null 2>&1; then
  clusters="$(kind get clusters 2> /dev/null || true)"
  if [[ "${mode}" == "compose" && -n "${clusters}" ]]; then
    echo "kind clusters are running: ${clusters}. Run 'make k8s-down' first." >&2
    exit 1
  fi
fi
if [[ "${mode}" == "k8s" ]]; then
  if docker ps --format '{{.Label "com.docker.compose.project"}}' | grep -qx "${project}"; then
    echo "The Compose stack is running. Run 'make down' first." >&2
    exit 1
  fi
fi
