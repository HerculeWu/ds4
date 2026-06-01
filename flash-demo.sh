#!/usr/bin/env bash
# =============================================================================
# flash-demo.sh — run DeepSeek-V4-Flash on this 6 GB GPU and answer a prompt.
#
# What this is showing off:
#   The model file is ~81 GiB. The GPU has 6 GB of VRAM. It still runs, because
#   ds4 streams the Mixture-of-Experts weights through a three-tier cache:
#
#         SSD (81 GiB model)  ->  host RAM (warm expert cache)  ->  VRAM (6 GB)
#
#   Each token only touches a handful of experts. We keep the hot ones in a
#   host-RAM LRU tier so most tokens never hit the disk, and a VRAM slot-bank
#   holds the working set the GPU actually computes on. That is the whole point
#   of the project: a too-big model running on a too-small GPU at a usable speed.
#
# Usage:
#   ./flash-demo.sh "Explain why the sky is blue."
#   ./flash-demo.sh                 # uses a built-in demo prompt
#   echo "long prompt..." | ./flash-demo.sh -   # read prompt from stdin
#
# Tunables (environment variables):
#   TOKENS=200    max tokens to generate            (-n)
#   CTX=4096      context window                     (-c)
#   SYS="..."     system prompt
#   VERBOSE=1     print the live cache tier stats (SSD/RAM/VRAM hit rates)
#   MODEL=...     gguf path        (default: ds4flash.gguf)
#   BIN=...       ds4 binary       (default: ./ds4)
# =============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

BIN="${BIN:-./ds4}"
MODEL="${MODEL:-ds4flash.gguf}"
TOKENS="${TOKENS:-200}"
CTX="${CTX:-4096}"
SYS="${SYS:-You are a helpful assistant.}"
VERBOSE="${VERBOSE:-0}"

# --- the prompt --------------------------------------------------------------
DEFAULT_PROMPT="In three short paragraphs, explain how a Mixture-of-Experts language model works and why it lets a very large model run on a small GPU."
if [ "${1:-}" = "-" ]; then
  PROMPT="$(cat)"                      # read from stdin
elif [ "$#" -gt 0 ]; then
  PROMPT="$*"                          # everything on the command line is the prompt
else
  PROMPT="$DEFAULT_PROMPT"
fi

# --- sanity checks -----------------------------------------------------------
[ -x "$BIN" ]   || { echo "ERROR: '$BIN' not found or not executable. Build it first:  make ds4 CUDA_ARCH=native" >&2; exit 2; }
[ -e "$MODEL" ] || { echo "ERROR: model '$MODEL' not found." >&2; exit 2; }

# --- banner ------------------------------------------------------------------
echo "=================================================================="
echo " DeepSeek-V4-Flash  ·  three-tier expert cache demo"
echo "=================================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  echo "  GPU      : ${gpu}"
fi
if [ -r /proc/meminfo ]; then
  awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "  Host RAM : %.0f GiB total, %.0f GiB free\n", t/1048576, a/1048576}' /proc/meminfo
fi
msize="$(stat -Lc %s "$MODEL" 2>/dev/null)"
[ -n "${msize:-}" ] && awk -v s="$msize" 'BEGIN{printf "  Model    : %s  (%.0f GiB on disk)\n", "'"$MODEL"'", s/1073741824}'
echo "  Tiers    : SSD  ->  host-RAM expert cache  ->  VRAM slot-bank"
echo "------------------------------------------------------------------"
echo "  Prompt: ${PROMPT}"
echo "------------------------------------------------------------------"
echo "  (first token is slow — the cache is warming from disk; then it"
echo "   settles to roughly one token every couple of seconds)"
echo "------------------------------------------------------------------"
echo

# --- run ---------------------------------------------------------------------
# Tier ON is the default (no DS4_CUDA_EXPERT_RAM_CACHE_GB=0). VERBOSE=1 turns on
# the weight-cache logging so you can watch the host-RAM hit rate climb.
ENVPREFIX=()
[ "$VERBOSE" = "1" ] && ENVPREFIX=(env DS4_CUDA_WEIGHT_CACHE_VERBOSE=1)

"${ENVPREFIX[@]}" "$BIN" \
  -m "$MODEL" \
  -c "$CTX" \
  -n "$TOKENS" \
  --temp 0 \
  -sys "$SYS" \
  -p "$PROMPT"

rc=$?
echo
echo "------------------------------------------------------------------"
echo "  (the 'generation: N t/s' line above is the steady-state decode speed)"
exit $rc
