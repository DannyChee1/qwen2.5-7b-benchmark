#!/usr/bin/env bash
# closed: fixed concurrency (CS), each point sized to run ~STEP_S seconds.
# open: fixed Poisson arrival rate in req/s (RS), each point runs DURATION_S seconds.
# Every run is marked as a region annotation on the Grafana dashboard.
set -euo pipefail
source ~/vllm-env/bin/activate

: "${MODE:?}"
IN_LEN=${IN_LEN:-512}
OUT_LEN=${OUT_LEN:-128}
DURATION_S=${DURATION_S:-180}
STEP_S=${STEP_S:-60}
PEAK_RPS=${PEAK_RPS:-60}
TAG=${TAG:-}
GRAFANA=http://127.0.0.1:3000
BASE_URL=http://127.0.0.1:8000

until curl -sf "$BASE_URL/health" >/dev/null; do
  echo "waiting for vLLM"
  sleep 5
done

MODEL=$(curl -sf "$BASE_URL/v1/models" | python -c "import json, sys; print(json.load(sys.stdin)['data'][0]['id'])")
OUT_DIR=~/results/$MODEL/$MODE
mkdir -p "$OUT_DIR"
echo "serving $MODEL, results in $OUT_DIR"

DASH_UID=$(curl -sf -u admin:admin "$GRAFANA/api/search?type=dash-db" | python -c "import sys, json; print(json.load(sys.stdin)[0]['uid'])" 2>/dev/null || true)
mark_start() {
  [ -n "$DASH_UID" ] || return 0
  curl -sf -u admin:admin -H 'Content-Type: application/json' -X POST "$GRAFANA/api/annotations" \
    -d "{\"dashboardUID\":\"$DASH_UID\",\"time\":$(date +%s000),\"tags\":[\"bench\",\"$MODE\"],\"text\":\"$1\"}" \
    | python -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null || true
}
mark_end() {
  [ -n "$1" ] || return 0
  curl -sf -u admin:admin -H 'Content-Type: application/json' -X PATCH "$GRAFANA/api/annotations/$1" \
    -d "{\"timeEnd\":$(date +%s000)}" >/dev/null 2>&1 || true
}

# --seed differs per run: the random dataset is seeded, so repeated prompts would hit the prefix cache.
# Two $RANDOM draws give a 30-bit seed; bare $RANDOM is 15 bits and collides ~0.4% per sweep.
seed() { echo $(( (RANDOM << 15) | RANDOM )); }
common=(
  --backend openai --base-url "$BASE_URL" --endpoint /v1/completions --model "$MODEL"
  --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" --ignore-eos
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
  --goodput ttft:500 tpot:50
  --save-result --result-dir "$OUT_DIR"
)

echo "warmup $(date -u +%T)"
id=$(mark_start "warmup")
vllm bench serve "${common[@]}" --seed "$(seed)" --max-concurrency 8 --num-prompts 64 \
  --result-filename warmup.json >/dev/null 2>&1 || true
mark_end "$id"

case "$MODE" in
  closed)
    for C in $CS; do
      # ~1.3 req/s per stream when unloaded, capped at PEAK_RPS; only sets the run length.
      N=$(python -c "import math; print(max(64, math.ceil($STEP_S * min(1.3 * $C, $PEAK_RPS))))")
      SEED=$(seed)
      echo "closed C=$C N=$N seed=$SEED $(date -u +%T)"
      id=$(mark_start "closed C=$C")
      vllm bench serve "${common[@]}" --seed "$SEED" --max-concurrency "$C" --num-prompts "$N" \
        --metadata mode=closed concurrency="$C" seed="$SEED" --result-filename "closed-c${C}${TAG}.json"
      mark_end "$id"
    done
    ;;
  open)
    [ -n "${RS:-}" ] || { echo "MODE=open needs RS"; exit 2; }
    for R in $RS; do
      N=$(python -c "import math; print(max(200, math.ceil($R * $DURATION_S)))")
      SEED=$(seed)
      echo "open R=$R N=$N seed=$SEED $(date -u +%T)"
      id=$(mark_start "open R=$R req/s")
      vllm bench serve "${common[@]}" --seed "$SEED" --request-rate "$R" --burstiness 1.0 --num-prompts "$N" \
        --metadata mode=open request_rate="$R" --result-filename "open-r${R}${TAG}.json"
      mark_end "$id"
    done
    ;;
  *)
    echo "MODE must be closed or open"; exit 2 ;;
esac

ls -la "$OUT_DIR"
