#!/usr/bin/env bash
# multigpu_ab.sh — validate the Phase 8 multi-GPU pipeline split on a 2-card box.
#
# WHAT THIS PROVES (the decisive Phase-3 gate)
#   The layer split must be BYTE-IDENTICAL to the single-GPU path. We run the SAME
#   greedy --temp 0 decode twice on the SAME binary:
#     (1) DS4_CUDA_DEVICES=1  -> forced single-GPU topology (the byte-identical
#         oracle; every layer maps to device 0, no device switch, no managed scratch).
#     (2) DS4_CUDA_DEVICES unset -> all visible devices; contiguous layer blocks run
#         on each card, the inter-layer carry crosses the one boundary.
#   These MUST be byte-identical (greedy). NO floating-point accumulation crosses a
#   device boundary -- each layer's whole FFN + the fused expert down-sum run on one
#   device in the same order as single-GPU -- so the logits cannot drift. If they
#   differ, the split broke accumulation order (or a tensor landed on the wrong card)
#   and the run FAILS here.
#
#   It also surfaces the residency win: with both cards the ~81 GB model shards across
#   ~96 GB VRAM, so per-device nvidia-smi should show each card holding ~half the
#   expert pool, and the per-token latency curve's cold-start (token 1) should shrink
#   versus the single-card reactive tier.
#
# REQUIREMENTS
#   A CUDA build (make cuda-bigmem CUDA_ARCH=sm_86) on a node with >=2 CUDA devices
#   and enough host RAM for the model map. Run from the repo root.
#
# USAGE
#   DS4_MODEL=/path/flash.gguf NTOK=64 tests/multigpu_ab.sh
#   env: DS4_MODEL (default ds4flash.gguf), NTOK (64), CTX (8192), DS4_BIN (./ds4).
set -u

DS4_MODEL="${DS4_MODEL:-ds4flash.gguf}"
DS4_BIN="${DS4_BIN:-./ds4}"
NTOK="${NTOK:-64}"
CTX="${CTX:-8192}"
SYS="${SYS:-You are a helpful assistant.}"
PROMPT="${PROMPT:-Explain, in detail and across several paragraphs, how a Mixture-of-Experts transformer routes tokens to experts, why only a few experts run per token, and what that means for memory and throughput.}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
say(){ printf '%s\n' "$*"; }
rule(){ printf -- '----------------------------------------------------------------------\n'; }

rule; say "[1] CONFIG"
[ -e "$DS4_MODEL" ] || { say "  ERROR: model '$DS4_MODEL' not found — set DS4_MODEL=/path/flash.gguf"; exit 2; }
[ -x "$DS4_BIN" ]   || { say "  ERROR: '$DS4_BIN' not built (make cuda-bigmem CUDA_ARCH=sm_86)"; exit 3; }
say "  Model : $DS4_MODEL   NTOK=$NTOK CTX=$CTX bin=$DS4_BIN"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  GPU   : /'
NGPU_VISIBLE="$(nvidia-smi -L 2>/dev/null | wc -l)"
say "  visible CUDA devices: ${NGPU_VISIBLE:-?}"
[ "${NGPU_VISIBLE:-0}" -ge 2 ] || say "  WARNING: <2 devices visible — the 'multi' run will equal the 'single' run."

# run <tag> <extra-env...> : greedy decode; capture stdout, stderr, and peak per-GPU mem.
run(){
  local tag="$1"; shift
  local memfile="$WORK/$tag.mem"; : > "$memfile"
  ( while :; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | paste -sd' ' >> "$memfile"; sleep 1; done ) & local mp=$!
  env DS4_BIGMEM=1 DS4_TOKEN_TIMING=1 "$@" \
      "$DS4_BIN" -m "$DS4_MODEL" -c "$CTX" -n "$NTOK" --temp 0 -sys "$SYS" -p "$PROMPT" \
      > "$WORK/$tag.out" 2> "$WORK/$tag.err"
  local rc=$?
  kill "$mp" 2>/dev/null; wait "$mp" 2>/dev/null
  return $rc
}

# per-token latency from "gpu decode eval N took X ms"
curve(){
  grep -oE 'gpu decode eval [0-9]+ took [0-9.]+ ms' "$WORK/$1.err" \
  | awk '{n++; ms=$6; a[n]=ms; s+=ms; if(min==""||ms<min)min=ms}
    END{ if(!n){print "    (no per-token lines)"; exit}
      ls=(n>32?n-31:1); l=0;c=0; for(i=ls;i<=n;i++){l+=a[i];c++}
      printf "    tokens=%d  token1=%.0fms  steady(last%d)=%.0fms  fastest=%.0fms\n", n, a[1], c, l/c, min }'
}

rule; say "[2] RUN  single (DS4_CUDA_DEVICES=1 — forced single-GPU oracle)"
run single DS4_CUDA_DEVICES=1; RC1=$?
say "  exit: $RC1"

rule; say "[3] RUN  multi  (all visible devices — pipeline layer split)"
run multi; RC2=$?
say "  exit: $RC2"
grep -E "CUDA device [0-9]|multi-GPU layer split|P2P" "$WORK/multi.err" | sed 's/^/    /'

rule; say "[4] CORRECTNESS  (single vs multi must be byte-identical, greedy)"
if [ "$RC1" = 0 ] && [ "$RC2" = 0 ] && cmp -s "$WORK/single.out" "$WORK/multi.out"; then
  CORRECT="PASS (byte-identical)"
else
  CORRECT="FAIL"
  say "  first diff:"; diff <(head -c 2000 "$WORK/single.out") <(head -c 2000 "$WORK/multi.out") | head -20 | sed 's/^/    /'
fi
say "  $CORRECT"

rule; say "[5] PER-TOKEN LATENCY"
say "  single:"; curve single
say "  multi :"; curve multi

peakmem(){ awk '{for(i=2;i<=NF;i+=2){g=$(i-1);if($i+0>m[g])m[g]=$i+0}} END{for(g in m)printf "    GPU %s peak used: %d MiB\n",g,m[g]}' "$WORK/$1.mem" 2>/dev/null; }
rule; say "[6] PEAK PER-GPU MEMORY"
say "  single run:"; peakmem single
say "  multi run :"; peakmem multi

rule; say "===================== SUMMARY ====================="
say "  CORRECTNESS : $CORRECT   (this is the gate)"
say "  Exit        : single=$RC1 multi=$RC2"
say "  With the split, the multi run should spread expert VRAM across both cards"
say "  (see [6]) and shrink token-1 cold-start (see [5]); steady t/s is ~unchanged"
say "  (single-stream decode is a depth-1 pipeline). The WIN is residency / no"
say "  cold-start / capacity, not a per-token throughput multiplier."
rule
