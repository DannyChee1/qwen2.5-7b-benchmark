#!/usr/bin/env bash
# Starts Prometheus + Grafana, then vLLM until min(MAX_HOURS, next DEADLINE_HHMM in DEADLINE_TZ).
set -uo pipefail

sudo docker compose -f observability/docker-compose.yaml up -d

now=$(date +%s)
deadline=$(TZ="$DEADLINE_TZ" date -d "$DEADLINE_HHMM" +%s)
[ "$deadline" -gt "$now" ] || deadline=$(TZ="$DEADLINE_TZ" date -d "tomorrow $DEADLINE_HHMM" +%s)
cap=$(( deadline - now < MAX_HOURS * 3600 ? deadline - now : MAX_HOURS * 3600 ))
echo "vllm stops at $(date -u -d @$((now + cap)) '+%H:%M UTC'), in $((cap / 60)) min"

source ~/vllm-env/bin/activate
vllm --version
timeout --signal=INT --kill-after=60 "$cap" vllm serve "$MODEL" --host 127.0.0.1 --port 8000 || [ $? -eq 124 ]
