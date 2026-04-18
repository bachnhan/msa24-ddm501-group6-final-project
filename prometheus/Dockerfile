FROM prom/prometheus:v2.48.0
COPY prometheus/prometheus.yml /etc/prometheus/prometheus.template.yml
COPY prometheus/alerts/ /etc/prometheus/alerts/

USER root
ENTRYPOINT [ "/bin/sh", "-c", "sed -e \"s/\\${API_TARGET}/$API_TARGET/g\" -e \"s/\\${API_SCHEME}/$API_SCHEME/g\" /etc/prometheus/prometheus.template.yml > /etc/prometheus/prometheus.yml && /bin/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus --web.listen-address=0.0.0.0:${PORT:-9090}" ]
