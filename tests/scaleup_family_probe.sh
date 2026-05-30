#!/usr/bin/env bash
# =============================================================================
# scaleup_family_probe.sh — validate the DS4 scale-up + V4-family generalization
# on a DIFFERENT machine than the 6 GB/31 GB baseline (e.g. A40 / 128 GB).
#
# It answers, with evidence, the two things that CANNOT be observed on the
# baseline box:
#   (1) HARDWARE SCALE-UP: does the engine auto-grow its tiers to the new box,
#       and do the two "jumps" fire —
#         - VRAM jump: slotbank holds the FULL expert set -> no eviction;
#         - RAM  jump: expert RAM tier auto-scales to the FULL pool -> disk inert?
#   (2) CORRECTNESS UNDER SCALE-UP: tier-on vs tier-off greedy decode is
#       byte-identical (the tier only changes weight SOURCE, never values).
#
# Optionally, given a SECOND gguf (a V4-Pro, or a different quant mix), it checks
# that the engine FAILS LOUD (named error) instead of silently producing garbage
# — the honest boundary for configs that can't be validated on the baseline box.
#
# Everything runs SEQUENTIALLY (the engine holds an instance lock; never run two
# model processes at once). Paste the final SUMMARY block back as feedback.
#
# Usage:
#   tests/scaleup_family_probe.sh
#   DS4_BIN=./ds4 DS4_MODEL=ds4flash.gguf tests/scaleup_family_probe.sh
#   DS4_MODEL_ALT=/path/to/v4pro.gguf tests/scaleup_family_probe.sh   # fail-loud test
#
# Env:
#   DS4_BIN        ds4 binary           (default: ./ds4)
#   DS4_MODEL      primary V4 gguf      (default: ds4flash.gguf)
#   DS4_MODEL_ALT  optional 2nd gguf    (Pro / other-quant; tests fail-loud)
#   PROBE_NTOK     tokens to generate   (default: 48)
#   PROBE_CTX      context size         (default: 4096)
# =============================================================================
set -u

DS4_BIN="${DS4_BIN:-./ds4}"
DS4_MODEL="${DS4_MODEL:-ds4flash.gguf}"
DS4_MODEL_ALT="${DS4_MODEL_ALT:-}"
PROBE_NTOK="${PROBE_NTOK:-48}"
PROBE_CTX="${PROBE_CTX:-4096}"
PROMPT="Write a detailed multi-paragraph explanation of how photosynthesis works, covering the light reactions and the Calvin cycle, with examples."

WORK="$(mktemp -d /tmp/ds4_probe.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

say()  { printf '%s\n' "$*"; }
rule() { printf '%s\n' "------------------------------------------------------------------"; }

# --- result accumulators -----------------------------------------------------
HW_GPU="unknown"; HW_VRAM="unknown"; HW_RAM="unknown"
SLOTBANK_LINE=""; EXPRAM_LINE=""; VRAM_JUMP="no"; RAM_JUMP="no"; AUTO_POOL_LINE=""
EQUIV="NOT-RUN"; TPS_OFF=""; TPS_ON=""
FAILLOUD="NOT-RUN"; FAILLOUD_MSG=""

# =============================================================================
say "=================================================================="
say " DS4 scale-up + V4-family probe"
say "=================================================================="
[ -x "$DS4_BIN" ] || { say "ERROR: DS4_BIN '$DS4_BIN' not executable. Build first: make ds4 CUDA_ARCH=native"; exit 2; }
[ -e "$DS4_MODEL" ] || { say "ERROR: DS4_MODEL '$DS4_MODEL' not found."; exit 2; }

# --- 0. hardware ------------------------------------------------------------
rule; say "[0] Detected hardware"
if command -v nvidia-smi >/dev/null 2>&1; then
  HW_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  HW_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
  say "  GPU:  ${HW_GPU}"
  say "  VRAM: ${HW_VRAM}"
else
  say "  GPU:  (nvidia-smi not found — is this a CUDA box?)"
fi
if [ -r /proc/meminfo ]; then
  kb="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
  akb="$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
  HW_RAM="$(awk -v t="$kb" -v a="$akb" 'BEGIN{printf "%.1f GiB total, %.1f GiB available", t/1048576, a/1048576}')"
  say "  RAM:  ${HW_RAM}"
fi

# --- 1. tier-sizing probe (verbose) -----------------------------------------
# A single short generation with verbose tier logging. We parse what the engine
# chose so the user sees the auto-scaling without reading raw logs.
rule; say "[1] Tier sizing on this box (env unset — pure auto-scaling)"
DS4_CUDA_WEIGHT_CACHE_VERBOSE=1 \
  "$DS4_BIN" -m "$DS4_MODEL" -c "$PROBE_CTX" --temp 0 -n "$PROBE_NTOK" -p "$PROMPT" \
  >"$WORK/size.out" 2>"$WORK/size.err"
SIZE_RC=$?
cat "$WORK/size.err" > "$WORK/size.log"; cat "$WORK/size.out" >> "$WORK/size.log"

SLOTBANK_LINE="$(grep -m1 'CUDA slotbank .* slots' "$WORK/size.log" || true)"
EXPRAM_LINE="$(grep -m1 'CUDA expert RAM tier .* slots' "$WORK/size.log" || true)"
AUTO_POOL_LINE="$(grep -m1 'auto-scaled to full pool' "$WORK/size.log" || true)"
grep -q 'holds the full model expert set' "$WORK/size.log" && VRAM_JUMP="yes"
{ [ -n "$AUTO_POOL_LINE" ] || grep -q 'disk inert after warmup' "$WORK/size.log"; } && RAM_JUMP="yes"

say "  slotbank: ${SLOTBANK_LINE:-<none — non-CUDA build or no MoE reached>}"
say "  exp RAM : ${EXPRAM_LINE:-<none>}"
[ -n "$AUTO_POOL_LINE" ] && say "  RAM jump: ${AUTO_POOL_LINE}"
say "  VRAM jump (slotbank holds full expert set): ${VRAM_JUMP}"
say "  RAM  jump (tier auto-scaled to full pool) : ${RAM_JUMP}"
if [ "$SIZE_RC" -ne 0 ]; then
  say "  NOTE: generation exited rc=$SIZE_RC — last stderr lines:"
  tail -5 "$WORK/size.err" | sed 's/^/    /'
fi

# --- 2. correctness: tier-on vs tier-off greedy decode must be byte-identical -
rule; say "[2] Correctness: tier-off (disk) vs tier-on (RAM) greedy decode"
say "    (--temp 0; the tier changes weight SOURCE only, so tokens must match)"

DS4_CUDA_EXPERT_RAM_CACHE_GB=0 \
  "$DS4_BIN" -m "$DS4_MODEL" -c "$PROBE_CTX" --temp 0 -n "$PROBE_NTOK" -p "$PROMPT" \
  >"$WORK/off.out" 2>"$WORK/off.err"
TPS_OFF="$(grep -hoE 'generation: *[0-9.]+ *t/s' "$WORK/off.err" "$WORK/off.out" 2>/dev/null | head -1)"

# tier ON: env UNSET so the auto-scaling path (incl. the RAM jump) is exercised.
DS4_CUDA_WEIGHT_CACHE_VERBOSE=1 \
  "$DS4_BIN" -m "$DS4_MODEL" -c "$PROBE_CTX" --temp 0 -n "$PROBE_NTOK" -p "$PROMPT" \
  >"$WORK/on.out" 2>"$WORK/on.err"
TPS_ON="$(grep -hoE 'generation: *[0-9.]+ *t/s' "$WORK/on.err" "$WORK/on.out" 2>/dev/null | head -1)"

# Compare generated stdout. Strip any perf/timing lines that legitimately differ.
clean() { grep -avE 't/s|tokens/sec|elapsed|[0-9]+ *ms\b' "$1" > "$2"; }
clean "$WORK/off.out" "$WORK/off.clean"
clean "$WORK/on.out"  "$WORK/on.clean"
if diff -q "$WORK/off.clean" "$WORK/on.clean" >/dev/null 2>&1; then
  EQUIV="PASS (byte-identical)"
else
  EQUIV="DIFF — inspect below"
fi
say "  tier-off t/s: ${TPS_OFF:-?}    tier-on t/s: ${TPS_ON:-?}"
say "  decode equivalence: ${EQUIV}"
if [ "$EQUIV" != "PASS (byte-identical)" ]; then
  say "  --- diff (off vs on); if this is only a timing/format line, it's benign:"
  diff "$WORK/off.clean" "$WORK/on.clean" | head -40 | sed 's/^/    /'
fi

# --- 3. optional fail-loud test for a Pro / other-quant gguf -----------------
if [ -n "$DS4_MODEL_ALT" ] && [ -e "$DS4_MODEL_ALT" ]; then
  rule; say "[3] Fail-loud test on alternate gguf: $DS4_MODEL_ALT"
  say "    (expect a NAMED fatal, not silent output, if kernels don't support it)"
  "$DS4_BIN" -m "$DS4_MODEL_ALT" -c "$PROBE_CTX" --temp 0 -n 4 -p "$PROMPT" \
    >"$WORK/alt.out" 2>"$WORK/alt.err"
  ALT_RC=$?
  FAILLOUD_MSG="$(grep -m1 -E 'FATAL unsupported model/hardware config|routed MoE expert quant|router_select(_batch)? n_expert' "$WORK/alt.err" "$WORK/alt.out" || true)"
  if [ -n "$FAILLOUD_MSG" ]; then
    FAILLOUD="PASS (died loud)"
  elif [ "$ALT_RC" -eq 0 ]; then
    FAILLOUD="FAIL (exited 0 — produced output; check it is not garbage)"
  else
    FAILLOUD="UNCLEAR (rc=$ALT_RC, no named fatal — see stderr)"
  fi
  say "  rc=$ALT_RC  verdict: ${FAILLOUD}"
  [ -n "$FAILLOUD_MSG" ] && say "  message: ${FAILLOUD_MSG}"
  [ -z "$FAILLOUD_MSG" ] && { say "  last stderr:"; tail -6 "$WORK/alt.err" | sed 's/^/    /'; }
else
  rule; say "[3] Fail-loud test: SKIPPED (set DS4_MODEL_ALT=/path/to/other.gguf to run)"
fi

# --- SUMMARY ----------------------------------------------------------------
rule; rule
say "SUMMARY  (paste this block back as feedback)"
rule
say "  box GPU            : ${HW_GPU}  | VRAM ${HW_VRAM}"
say "  box RAM            : ${HW_RAM}"
say "  slotbank sizing    : ${SLOTBANK_LINE:-none}"
say "  expert RAM tier    : ${EXPRAM_LINE:-none}"
say "  VRAM elision jump  : ${VRAM_JUMP}   (slotbank holds full expert set?)"
say "  RAM  elision jump  : ${RAM_JUMP}   (tier auto-scaled to full pool -> disk inert?)"
say "  decode equivalence : ${EQUIV}"
say "  throughput         : off=${TPS_OFF:-?}  on=${TPS_ON:-?}"
say "  fail-loud (alt)    : ${FAILLOUD}"
[ -n "$FAILLOUD_MSG" ] && say "  fail-loud message  : ${FAILLOUD_MSG}"
rule
# Exit non-zero if the one thing that MUST hold here (correctness) regressed.
case "$EQUIV" in
  "PASS (byte-identical)") exit 0 ;;
  "NOT-RUN") exit 0 ;;
  *) exit 1 ;;
esac
