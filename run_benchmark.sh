#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config
# =========================
DATASET_PATH="data/parallel_sentences.csv"
RESULTS_ROOT="results"
SEED=0
MAX_ROWS=""   # leave empty for full
EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


# Paths to evaluation scripts
WSD_SCRIPT="wsd.py"
CROSS_LINGUAL_WSD_SCRIPT="cross_lingual_wsd.py"

# Models/baselines to evaluate
# - include "random" for chance baseline
# - include "lesk" or "gloss_similarity" for non-generative baselines
# - include any HF model id/path
MODELS=(
  # "random"
  "lesk"
  "gloss_similarity"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "mistralai/Mistral-7B-Instruct-v0.3"
  # "Qwen/Qwen2.5-0.5B-Instruct"
  # "Qwen/Qwen2.5-1.5B-Instruct"
  # "Qwen/Qwen2.5-7B-Instruct"
  # "Qwen/Qwen2.5-14B-Instruct"
  # "google/gemma-3-4b-it"
  # "google/gemma-3-12b-it"
  # "BSC-LT/salamandra-7b-instruct"
)

# =========================
# Helpers
# =========================
timestamp() { date +"%Y%m%d_%H%M%S"; }

run_wsd () {
  local model="$1"
  local outdir="$2"
  local shuffle="$3"   # "0" or "1"

  mkdir -p "$outdir"

  cmd=(python "$WSD_SCRIPT" -m "$model" -r "$outdir" -d "$DATASET_PATH" --seed "$SEED" --embedding_model "$EMBEDDING_MODEL")
  if [[ -n "${MAX_ROWS}" ]]; then cmd+=(--max_rows "$MAX_ROWS"); fi
  if [[ "$shuffle" == "1" ]]; then cmd+=(--shuffle_senses); fi

  echo ">> WSD | model=$model | shuffle=$shuffle | out=$outdir"
  "${cmd[@]}"
}

run_cross_lingual_wsd () {
  local model="$1"
  local outdir="$2"
  local shuffle="$3"   # "0" or "1"

  mkdir -p "$outdir"

  cmd=(python "$CROSS_LINGUAL_WSD_SCRIPT" -m "$model" -r "$outdir" -d "$DATASET_PATH" --seed "$SEED" --embedding_model "$EMBEDDING_MODEL")
  if [[ -n "${MAX_ROWS}" ]]; then cmd+=(--max_rows "$MAX_ROWS"); fi
  if [[ "$shuffle" == "1" ]]; then cmd+=(--shuffle_senses); fi

  echo ">> Cross-lingual WSD | model=$model | shuffle=$shuffle | out=$outdir"
  "${cmd[@]}"
}

# =========================
# Main
# =========================
RUN_ID="$(timestamp)"
BASE_OUT="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "$BASE_OUT"

echo "Benchmark run id: $RUN_ID"
echo "Dataset: $DATASET_PATH"
echo "Results: $BASE_OUT"
echo

for model in "${MODELS[@]}"; do
  safe_model="$(echo "$model" | sed 's#[^A-Za-z0-9._-]#_#g')"
  model_out="${BASE_OUT}/${safe_model}"
  mkdir -p "$model_out"

  if [[ "$model" == "random" ]]; then
    # Random baseline: no need to run shuffle variant (it is invariant)
    run_wsd "$model" "${model_out}/wsd" "0"
    run_cross_lingual_wsd "$model" "${model_out}/cross_lingual_wsd" "0"
  else
    # Non-random models: run both unshuffled and shuffled
    run_wsd "$model" "${model_out}/wsd_unshuffled" "0"
    run_wsd "$model" "${model_out}/wsd_shuffled"   "1"

    run_cross_lingual_wsd "$model" "${model_out}/cross_lingual_wsd_unshuffled" "0"
    run_cross_lingual_wsd "$model" "${model_out}/cross_lingual_wsd_shuffled"   "1"
  fi

  echo
done

echo "Done. Results in: $BASE_OUT"
