#!/usr/bin/env bash
# =============================================================================
# bigmem_scaleup_probe.sh — rebuild ds4 in scale-up mode and A/B the DS4_BIGMEM
# profile on a LARGE box (e.g. A40 48 GB VRAM + 128 GB RAM), then paste the final
# SUMMARY block back so the defaults (headroom / margin / prefill ceiling) can be
# tuned to the real hardware.
#
# What it proves / measures, all on ONE freshly built `cuda-bigmem` binary
# (the gate honors the runtime DS4_BIGMEM env in BOTH directions, so the same
# binary is its own A/B):
#
#   1. BUILD            `make clean && make cuda-bigmem` succeeds, -DDS4_BIGMEM in flags.
#   2. RESIDENCY (ON)   the three scale-up lines fire under DS4_BIGMEM=1:
#                         - backbone VRAM-resident
#                         - expert RAM tier auto-scaled to the full pool (disk inert)
#                         - slotbank sizing
#                       plus the lifted prefill_chunk, peak VRAM, and a "skipped"
#                       line if the backbone slab did NOT fit (so we can see it).
#   3. SPEED            prefill & generation t/s for DS4_BIGMEM=1 vs =0 (speedup).
#   4. CORRECTNESS      greedy (--temp 0) generated text is BYTE-IDENTICAL between
#                       DS4_BIGMEM=1 and =0 — the tiers only change the SOURCE of
#                       the weights, never their values. A short prompt keeps
#                       prefill to a single chunk so this isolates the residency
#                       change (run the golden / scaleup_family_probe.sh for the
#                       long-prefill correctness gate).
#
# Runs are SEQUENTIAL (the engine holds an instance lock — never two at once).
#
# Usage:
#   tests/bigmem_scaleup_probe.sh
#   DS4_MODEL=/path/to/flash.gguf NTOK=400 tests/bigmem_scaleup_probe.sh
#   REBUILD=0 tests/bigmem_scaleup_probe.sh        # skip the rebuild, A/B the current ./ds4
#
# Env:
#   DS4_BIN    ds4 binary                 (default: ./ds4)
#   DS4_MODEL  gguf path                  (default: ds4flash.gguf)
#   NTOK       tokens to generate         (default: 256)
#   CTX        context window             (default: 8192)
#   DS4_ARCH   nvcc -arch for the build   (default: native)
#   REBUILD    1 = make clean && cuda-bigmem; 0 = use existing binary (default: 1)
#   PROMPT     override the prompt
# =============================================================================
set -u

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

DS4_BIN="${DS4_BIN:-./ds4}"
DS4_MODEL="${DS4_MODEL:-ds4flash.gguf}"
NTOK="${NTOK:-256}"
CTX="${CTX:-8192}"
DS4_ARCH="${DS4_ARCH:-native}"
REBUILD="${REBUILD:-1}"
SYS="You are a helpful assistant."
PROMPT="${PROMPT:-Explain, in one paragraph, what a Mixture-of-Experts language model is and why only a few experts run per token.}"

STAMP="$(date +%Y%m%d-%H%M%S 2>/dev/null || echo run)"
WORK="$(mktemp -d /tmp/ds4_bigmem.XXXXXX)"
LOG="$HERE/bigmem_probe_${STAMP}.log"
trap 'rm -rf "$WORK"' EXIT

# tee everything to a log file the user can attach
exec > >(tee "$LOG") 2>&1

rule(){ printf -- '------------------------------------------------------------------\n'; }
say(){ printf '%s\n' "$*"; }

say "=================================================================="
say " ds4 BIGMEM scale-up probe   ($STAMP)"
say "=================================================================="

# --- 1. hardware ------------------------------------------------------------
rule; say "[1] HARDWARE"
GPU="?"; VRAM="?"; PCIE="?"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
  PCIE="$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader 2>/dev/null | head -1)"
  say "  GPU         : $GPU"
  say "  VRAM total  : $VRAM"
  say "  PCIe (gen,width): $PCIE"
fi
RAM_TOTAL="?"; RAM_AVAIL="?"
if [ -r /proc/meminfo ]; then
  RAM_TOTAL="$(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  RAM_AVAIL="$(awk '/MemAvailable/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  say "  Host RAM    : $RAM_TOTAL total, $RAM_AVAIL available"
fi
if [ -e "$DS4_MODEL" ]; then
  MSZ="$(stat -Lc %s "$DS4_MODEL" 2>/dev/null)"
  [ -n "${MSZ:-}" ] && say "  Model       : $DS4_MODEL  ($(awk -v s="$MSZ" 'BEGIN{printf "%.1f GiB", s/1073741824}'))"
  FS="$(stat -f -c %T "$DS4_MODEL" 2>/dev/null || echo '?')"
  say "  Model fs    : $FS"
else
  say "  ERROR: model '$DS4_MODEL' not found — set DS4_MODEL=/path/to/flash.gguf"; exit 2
fi

# --- 2. rebuild -------------------------------------------------------------
rule; say "[2] BUILD"
BUILD_OK="skipped"; BIGMEM_FLAG="?"
if [ "$REBUILD" = "1" ]; then
  say "  make clean && make cuda-bigmem CUDA_ARCH=$DS4_ARCH"
  if make clean >/dev/null 2>&1 && make cuda-bigmem CUDA_ARCH="$DS4_ARCH" > "$WORK/build.log" 2>&1; then
    BUILD_OK="ok"
  else
    BUILD_OK="FAILED"; say "  BUILD FAILED — tail of build log:"; tail -25 "$WORK/build.log"; exit 3
  fi
  BIGMEM_FLAG="$(grep -oE '\-DDS4_BIGMEM' "$WORK/build.log" | head -1)"
  say "  build: $BUILD_OK   (-DDS4_BIGMEM present in nvcc line: ${BIGMEM_FLAG:-NO})"
else
  say "  REBUILD=0 — using existing $DS4_BIN (A/B still works via runtime DS4_BIGMEM)"
fi
[ -x "$DS4_BIN" ] || { say "  ERROR: '$DS4_BIN' not built"; exit 3; }

# --- run helper -------------------------------------------------------------
# run_ds4 <tag> <bigmem 0|1> -> writes $WORK/<tag>.out (stdout) + .err (stderr),
# polls peak VRAM into $WORK/<tag>.vram
run_ds4(){
  local tag="$1" bm="$2"
  local vramfile="$WORK/$tag.vram" pollpid=""
  : > "$vramfile"
  if command -v nvidia-smi >/dev/null 2>&1; then
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 >> "$vramfile"; sleep 0.5; done ) &
    pollpid=$!
  fi
  env DS4_BIGMEM="$bm" DS4_CUDA_WEIGHT_CACHE_VERBOSE=1 \
      "$DS4_BIN" -m "$DS4_MODEL" -c "$CTX" -n "$NTOK" --temp 0 \
      -sys "$SYS" -p "$PROMPT" > "$WORK/$tag.out" 2> "$WORK/$tag.err"
  local rc=$?
  [ -n "$pollpid" ] && { kill "$pollpid" 2>/dev/null; wait "$pollpid" 2>/dev/null; }
  return $rc
}

peak_vram(){ awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{print m" MiB"}' "$WORK/$1.vram" 2>/dev/null; }
ts_line(){ grep -oE 'prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s' "$WORK/$1.err" | tail -1; }
gen_ts(){ grep -oE 'generation: [0-9.]+ t/s' "$WORK/$1.err" | tail -1 | grep -oE '[0-9.]+'; }

# --- 3. BIGMEM ON -----------------------------------------------------------
rule; say "[3] RUN  DS4_BIGMEM=1  (scale-up ON)  — first token warms the tiers from disk"
run_ds4 on 1; RC_ON=$?
say "  exit: $RC_ON"
say "  --- residency plan (from verbose stderr) ---"
grep -E "backbone VRAM-resident|backbone VRAM residency skipped|expert RAM tier auto-scaled|CUDA expert RAM tier [0-9]|CUDA slotbank [0-9]|slotbank holds the full|VRAM-elision|context buffers" "$WORK/on.err" | sed 's/^/    /' || true
say "  --- prefill chunk / KV ---"
grep -E "context buffers|prefill_chunk=" "$WORK/on.err" | tail -2 | sed 's/^/    /' || true
say "  --- last expert-tier hit/miss counters (warm = disk inert) ---"
grep -iE "host hit|expram L|host_hit|vram hit" "$WORK/on.err" | tail -4 | sed 's/^/    /' || say "    (no per-layer counters printed)"
say "  --- any FATAL / OOM / error ---"
grep -iE "FATAL|out of memory|OOM|failed|abort" "$WORK/on.err" | grep -viE "decode failed: $" | head -8 | sed 's/^/    /' || true
say "  peak VRAM used (ON): $(peak_vram on)"
say "  timing (ON): $(ts_line on)"

# --- 4. BIGMEM OFF (baseline) ----------------------------------------------
rule; say "[4] RUN  DS4_BIGMEM=0  (baseline: streaming backbone + LRU)"
run_ds4 off 0; RC_OFF=$?
say "  exit: $RC_OFF"
say "  peak VRAM used (OFF): $(peak_vram off)"
say "  timing (OFF): $(ts_line off)"

# --- 5. correctness: byte-identical greedy output --------------------------
rule; say "[5] CORRECTNESS  (greedy text must be byte-identical ON vs OFF)"
if [ "$RC_ON" -eq 0 ] && [ "$RC_OFF" -eq 0 ]; then
  if diff -q "$WORK/on.out" "$WORK/off.out" >/dev/null 2>&1; then
    CORRECT="PASS (byte-identical)"
  else
    CORRECT="FAIL (outputs differ — see diff below)"
    say "  --- first differing lines ---"; diff "$WORK/off.out" "$WORK/on.out" | head -20 | sed 's/^/    /'
  fi
else
  CORRECT="N/A (a run exited non-zero)"
fi
say "  $CORRECT"

# --- 6. summary -------------------------------------------------------------
GON="$(gen_ts on)"; GOFF="$(gen_ts off)"
SPEEDUP="?"
if [ -n "${GON:-}" ] && [ -n "${GOFF:-}" ]; then
  SPEEDUP="$(awk -v a="$GON" -v b="$GOFF" 'BEGIN{ if(b>0) printf "%.2fx", a/b; else print "?" }')"
fi
BB_RESIDENT="no"; grep -q "backbone VRAM-resident" "$WORK/on.err" && BB_RESIDENT="YES"
BB_SKIP="$(grep -oE 'backbone VRAM residency skipped[^\n]*' "$WORK/on.err" | head -1)"
RAM_FULL="no"; grep -q "auto-scaled to full pool" "$WORK/on.err" && RAM_FULL="YES"

rule
say "==================  SUMMARY (paste this back)  =================="
say "  GPU / VRAM      : $GPU / $VRAM"
say "  Host RAM        : $RAM_TOTAL total, $RAM_AVAIL avail at start"
say "  Model           : $DS4_MODEL ($(awk -v s="${MSZ:-0}" 'BEGIN{printf "%.1f GiB", s/1073741824}')), fs=$FS"
say "  Build           : $BUILD_OK   -DDS4_BIGMEM: ${BIGMEM_FLAG:-N/A}"
say "  ctx=$CTX  ntok=$NTOK"
say "  ----"
say "  backbone VRAM-resident : $BB_RESIDENT   ${BB_SKIP:+[$BB_SKIP]}"
say "  expert RAM -> full pool: $RAM_FULL"
say "  peak VRAM  ON / OFF    : $(peak_vram on)  /  $(peak_vram off)"
say "  generation t/s ON / OFF: ${GON:-?}  /  ${GOFF:-?}   speedup: $SPEEDUP"
say "  prefill+gen  ON        : $(ts_line on)"
say "  correctness            : $CORRECT"
say "  exits  ON / OFF        : $RC_ON / $RC_OFF"
say "================================================================"
say ""
say "Full log saved to: $LOG"
say "(If 'backbone VRAM-resident' is 'no' with a skipped line, the slab did not"
say " fit free VRAM — paste the SUMMARY and I'll lower the 2 GiB headroom.)"
