#!/usr/bin/env bash
# eager_prefill_ab.sh — validate the CUDA host-RAM eager-prefill tier on a big box.
#
# WHAT THIS PROVES
#   The bigmem decode floor on the A40 was dominated by COLD-DISK WARMUP, not
#   compute: the first decode token took ~21.6 s (a third of a 227-token run) and
#   the per-token latency then ramped from ~700 ms down to a ~105 ms steady state
#   as the working set populated VRAM/RAM from disk. Eager-prefill streams the WHOLE
#   routed-expert pool disk->host RAM once at decode start so every later VRAM miss
#   is a guaranteed PCIe RAM->VRAM serve instead of a disk read — collapsing the ramp.
#
#   This script runs the SAME greedy decode twice and checks two things:
#     (1) CORRECTNESS  — eager ON vs OFF must be BYTE-IDENTICAL (greedy, --temp 0).
#         Eager-prefill only changes WHEN expert bytes load, never which bytes; if
#         the registered GGUF offsets ever drifted from the decode path the text
#         would diverge here (and the engine would also FATAL before serving).
#     (2) SPEED SHAPE  — the per-token latency curve (DS4_TOKEN_TIMING) should show
#         the warmup ramp gone with eager ON: first-token and early tokens drop
#         toward the steady state, modulo the one-time disk->RAM load reported at
#         startup ("eager expert prefill loaded X GiB ... in Ys").
#
# REQUIREMENTS
#   A CUDA build (make cuda-bigmem) on a box where the routed-expert pool (~78 GiB)
#   fits host RAM with margin — otherwise eager-prefill self-gates OFF (it will say
#   "skipped: tier ... < full pool") and the ON run silently equals OFF. NTOK should
#   be large enough to reach a clear steady state (>=512; 1024 is better).
#
# USAGE
#   DS4_MODEL=/path/flash.gguf NTOK=1024 tests/eager_prefill_ab.sh
#   env knobs: DS4_MODEL (default ds4flash.gguf), NTOK (512), CTX (8192),
#              DS4_BIN (./ds4), DS4_ARCH (native, only if REBUILD=1), REBUILD (0).
set -u

DS4_MODEL="${DS4_MODEL:-ds4flash.gguf}"
DS4_BIN="${DS4_BIN:-./ds4}"
NTOK="${NTOK:-512}"
CTX="${CTX:-8192}"
DS4_ARCH="${DS4_ARCH:-native}"
REBUILD="${REBUILD:-0}"
SYS="${SYS:-You are a helpful assistant.}"
PROMPT="${PROMPT:-Explain, in detail and across several paragraphs, how a Mixture-of-Experts transformer routes tokens to experts, why only a few experts run per token, and what that means for memory and throughput.}"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
say(){ printf '%s\n' "$*"; }
rule(){ printf -- '----------------------------------------------------------------------\n'; }

rule; say "[1] CONFIG"
if [ -e "$DS4_MODEL" ]; then
  MSZ="$(stat -Lc %s "$DS4_MODEL" 2>/dev/null || echo 0)"
  say "  Model : $DS4_MODEL ($(awk -v s="${MSZ:-0}" 'BEGIN{printf "%.1f GiB", s/1073741824}'))"
  FS="$(stat -f -c %T "$DS4_MODEL" 2>/dev/null || echo '?')"; say "  FS    : $FS"
else
  say "  ERROR: model '$DS4_MODEL' not found — set DS4_MODEL=/path/flash.gguf"; exit 2
fi
say "  NTOK=$NTOK  CTX=$CTX  bin=$DS4_BIN"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU   : /'
awk '/MemTotal|MemAvailable/{printf "  %s\n",$0}' /proc/meminfo 2>/dev/null

if [ "$REBUILD" = "1" ]; then
  rule; say "[1b] BUILD  make clean && make cuda-bigmem CUDA_ARCH=$DS4_ARCH"
  make clean >/dev/null 2>&1 && make cuda-bigmem CUDA_ARCH="$DS4_ARCH" > "$WORK/build.log" 2>&1 \
    || { say "  BUILD FAILED:"; tail -25 "$WORK/build.log"; exit 3; }
  say "  build ok"
fi
[ -x "$DS4_BIN" ] || { say "  ERROR: '$DS4_BIN' not built (make cuda-bigmem)"; exit 3; }

# run_ab <tag> <extra-env...> : greedy decode with token timing + verbose; poll RSS.
PROC="$(basename "$DS4_BIN")"
run_ab(){
  local tag="$1"; shift
  local rssfile="$WORK/$tag.rss"; : > "$rssfile"
  ( while :; do ps --no-headers -o rss -C "$PROC" 2>/dev/null | sort -rn | head -1 >> "$rssfile"; sleep 0.5; done ) &
  local pp=$!
  env DS4_BIGMEM=1 DS4_TOKEN_TIMING=1 DS4_CUDA_WEIGHT_CACHE_VERBOSE=1 "$@" \
      "$DS4_BIN" -m "$DS4_MODEL" -c "$CTX" -n "$NTOK" --temp 0 \
      -sys "$SYS" -p "$PROMPT" > "$WORK/$tag.out" 2> "$WORK/$tag.err"
  local rc=$?
  kill "$pp" 2>/dev/null; wait "$pp" 2>/dev/null
  return $rc
}

peak_rss(){ awk 'BEGIN{m=0}{if($1+0>m)m=$1+0}END{printf "%.1f GiB", m/1048576}' "$WORK/$1.rss" 2>/dev/null; }
gen_ts(){ grep -oE 'generation: [0-9.]+ t/s' "$WORK/$1.err" | tail -1 | grep -oE '[0-9.]+'; }

# per-token latency curve from "ds4: gpu decode eval N took X ms"
curve(){
  grep -oE 'gpu decode eval [0-9]+ took [0-9.]+ ms' "$WORK/$1.err" \
  | awk '{n++; ms=$6; a[n]=ms; s+=ms; if(min==""||ms<min)min=ms}
    END{ if(!n){print "    (no per-token lines)"; exit}
      fn=(n<16?n:16); f=0; for(i=1;i<=fn;i++)f+=a[i];
      ls=(n>64?n-63:1); l=0; c=0; for(i=ls;i<=n;i++){l+=a[i];c++}
      printf "    tokens   = %d\n", n;
      printf "    token 1  = %.0f ms  (%.2f t/s)\n", a[1], 1000/a[1];
      printf "    overall  = %.0f ms  (%.2f t/s)\n", s/n, 1000/(s/n);
      printf "    first %-2d = %.0f ms  (%.2f t/s)   <- warmup\n", fn, f/fn, 1000/(f/fn);
      printf "    last  %-2d = %.0f ms  (%.2f t/s)   <- steady\n", c, l/c, 1000/(l/c);
      printf "    fastest  = %.0f ms  (%.2f t/s)   <- compute floor\n", min, 1000/min }'
}

rule; say "[2] RUN  eager ON  (DS4_BIGMEM=1 DS4_CUDA_EXPERT_PREFILL=1 — opt-in eager prefill)"
run_ab on DS4_CUDA_EXPERT_PREFILL=1; RC_ON=$?
say "  exit: $RC_ON"
grep -E "eager expert prefill" "$WORK/on.err" | sed 's/^/    /' || say "    (no eager-prefill line — did it self-gate off? check pool-vs-RAM)"
grep -iE "FATAL|mismatch|abort|out of memory|OOM" "$WORK/on.err" | head -5 | sed 's/^/    !! /' || true
say "  peak host RSS (ON): $(peak_rss on)"

rule; say "[3] RUN  eager OFF (DS4_BIGMEM=1, default — lazy reactive tier)"
run_ab off; RC_OFF=$?
say "  exit: $RC_OFF"
grep -iE "FATAL|mismatch|abort|out of memory|OOM" "$WORK/off.err" | head -5 | sed 's/^/    !! /' || true
say "  peak host RSS (OFF): $(peak_rss off)"

rule; say "[4] CORRECTNESS  (eager ON vs OFF must be byte-identical, greedy)"
if [ "$RC_ON" = 0 ] && [ "$RC_OFF" = 0 ] && cmp -s "$WORK/on.out" "$WORK/off.out"; then
  CORRECT="PASS (byte-identical)"
else
  CORRECT="FAIL"
  say "  first diff:"; diff <(head -c 2000 "$WORK/on.out") <(head -c 2000 "$WORK/off.out") | head -20 | sed 's/^/    /'
fi
say "  $CORRECT"

rule; say "[5] PER-TOKEN LATENCY CURVE"
say "  ON (eager prefill):";  curve on
say "  OFF (lazy warmup):";   curve off

ELOAD="$(grep -oE 'eager expert prefill loaded [0-9.]+ GiB \([0-9]+ experts\) disk->host RAM in [0-9.]+s' "$WORK/on.err" | tail -1)"

rule; say "===================== SUMMARY ====================="
say "  Model           : $DS4_MODEL ($(awk -v s="${MSZ:-0}" 'BEGIN{printf "%.1f GiB",s/1073741824}'), fs=${FS:-?})"
say "  NTOK / CTX      : $NTOK / $CTX     exit ON=$RC_ON OFF=$RC_OFF"
say "  CORRECTNESS     : $CORRECT"
say "  Eager load      : ${ELOAD:-<none — eager self-gated off; pool did not fit RAM with margin>}"
say "  Aggregate t/s   : ON $(gen_ts on)   OFF $(gen_ts off)   (aggregate is dragged by the one-time load; read the curve's 'steady')"
say "  Peak host RSS   : ON $(peak_rss on)   OFF $(peak_rss off)   (keep under usable RAM)"
say ""
say "  READ: CORRECTNESS must be PASS — that is the gate for the new disk->host loader."
say "  With eager ON the curve's 'token 1' and 'first 16' should fall toward 'last 64'"
say "  (ramp gone). The ON aggregate may look similar/worse than OFF on a short run"
say "  because it pays the whole disk->RAM load up front; the STEADY rate and a long"
say "  session are where it wins. If 'Eager load' is <none>, the pool did not fit RAM."
rule
