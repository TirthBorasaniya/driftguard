{{/* Common labels. */}}
{{- define "driftguard.labels" -}}
app.kubernetes.io/name: driftguard
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: driftguard
{{- end }}

{{/* Selector labels for one component. */}}
{{- define "driftguard.selector" -}}
app.kubernetes.io/name: driftguard
app.kubernetes.io/component: {{ . }}
{{- end }}

{{/* Application image. */}}
{{- define "driftguard.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag }}
{{- end }}

{{/* Environment shared by every DriftGuard Python container. */}}
{{- define "driftguard.env" -}}
- {name: DATASET, value: {{ .Values.dataset | quote }}}
- {name: KAFKA_BOOTSTRAP_SERVERS, value: {{ .Values.kafka.bootstrap | quote }}}
- {name: SCHEMA_REGISTRY_URL, value: "http://driftguard-schema-registry:8081"}
- {name: REDIS_URL, value: "redis://driftguard-redis:6379/0"}
- {name: MLFLOW_TRACKING_URI, value: "http://driftguard-mlflow:5000"}
- {name: PREFECT_API_URL, value: "http://driftguard-prefect-server:4200/api"}
- {name: SERVING_URL, value: "http://driftguard-serving:8001"}
- {name: PROMETHEUS_URL, value: "http://{{ .Values.prometheusRelease }}-prometheus.monitoring:9090"}
- {name: DATA_DIR, value: /app/data}
- {name: OFFLINE_DIR, value: /app/data/offline}
- {name: RESULTS_DIR, value: /app/data/results}
- {name: PROMOTIONS_LOG_PATH, value: /app/data/results/promotions.jsonl}
- {name: ORCHESTRATOR_DB_PATH, value: /app/data/orchestrator.db}
- {name: FEAST_REPO_PATH, value: /app/data/feast/repo}
- name: ADMIN_TOKEN
  valueFrom: {secretKeyRef: {name: driftguard-secrets, key: admin-token}}
- name: IP_HASH_SALT
  valueFrom: {secretKeyRef: {name: driftguard-secrets, key: ip-hash-salt}}
{{- end }}

{{/* Volumes for components that use the shared data PVC and the raw datasets. */}}
{{- define "driftguard.dataVolumes" -}}
- name: data
  persistentVolumeClaim: {claimName: driftguard-data}
- name: raw
  hostPath: {path: {{ .Values.storage.rawHostPath }}, type: DirectoryOrCreate}
{{- end }}

{{- define "driftguard.dataMounts" -}}
- {name: data, mountPath: /app/data}
- {name: raw, mountPath: /app/data/raw, readOnly: true}
{{- end }}

{{/* Pod security context for the application image (uid 10001 from the Dockerfile). */}}
{{- define "driftguard.podSecurity" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  fsGroup: 10001
{{- end }}
