#!/usr/bin/env bash
# closed: fixed concurrency (CS). open: fixed Poisson arrival rate in req/s (RS).
set -euo pipefail
source ~/vllm-env/bin/activate

: "${MODE:?}"
IN_LEN=${IN_LEN:-512}
OUT_LEN=${OUT_LEN:-128}
DURATION_S=${DURATION_S:-180}
MULT=${MULT:-8}
TAG=${TAG:-}
BASE_URL=http://127.0.0.1:8000

until curl -sf "$BASE_URL/health" >/dev/null; do
  echo "waiting for vLLM"
  sleep 5
done

MODEL=$(curl -sf "$BASE_URL/v1/models" | python -c "import json, sys; print(json.load(sys.stdin)['data'][0]['id'])")
OUT_DIR=~/results/$MODEL/$MODE
mkdir -p "$OUT_DIR"
echo "serving $MODEL, results in $OUT_DIR"

# --seed differs per run: the random dataset is seeded, so repeated prompts would hit the prefix cache.
common=(
  --backend openai --base-url "$BASE_URL" --endpoint /v1/completions --model "$MODEL"
  --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" --ignore-eos
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
  --goodput ttft:500 tpot:50
  --save-result --result-dir "$OUT_DIR"
)

echo "warmup $(date -u +%T)"
vllm bench serve "${common[@]}" --seed "$RANDOM" --max-concurrency 8 --num-prompts 64 \
  --result-filename warmup.json >/dev/null 2>&1 || true

case "$MODE" in
  closed)
    for C in $CS; do
      N=$(( C * MULT > 256 ? C * MULT : 256 ))
      echo "closed C=$C N=$N $(date -u +%T)"
      vllm bench serve "${common[@]}" --seed "$RANDOM" --max-concurrency "$C" --num-prompts "$N" \
        --metadata mode=closed concurrency="$C" --result-filename "closed-c${C}${TAG}.json"
    done
    ;;
  open)
    [ -n "${RS:-}" ] || { echo "MODE=open needs RS"; exit 2; }
    for R in $RS; do
      N=$(python -c "import math; print(max(200, math.ceil($R * $DURATION_S)))")
      echo "open R=$R N=$N $(date -u +%T)"
      vllm bench serve "${common[@]}" --seed "$RANDOM" --request-rate "$R" --burstiness 1.0 --num-prompts "$N" \
        --metadata mode=open request_rate="$R" --result-filename "open-r${R}${TAG}.json"
    done
    ;;
  *)
    echo "MODE must be closed or open"; exit 2 ;;
esac

ls -la "$OUT_DIR"
