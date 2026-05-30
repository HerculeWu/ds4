# Scale-up + DeepSeek-V4 family generalization

**Goal:** make the engine (a) auto-scale UP from the 6 GB/31 GB baseline so the
SSD→RAM→VRAM tiers exploit a bigger box (and the disk tier goes *inert* once RAM
holds the whole expert pool), and (b) stop silently mis-running any non-Flash-IQ2XXS
DeepSeek-V4 config — converting every silent-garbage path into a loud, named error,
while keeping shapes/quants flowing from GGUF metadata.

**Non-goal (explicit):** this stays a DeepSeek-V4-arch-specific engine, NOT a generic
GGUF runner (per CLAUDE.md). The architecture stays closed (shape allowlist = Flash or
Pro; quant set = the V4 family's known routed/backbone types). We only make code
*downstream* of selection read runtime fields instead of re-baking constants, and we
make untested-but-structurally-expressible configs (Pro 384-expert router, Q4_K prefill,
non-Q8 attention) **fail loud** instead of silently wrong.

**Validation reality:** the only assets here are the V4-Flash-IQ2XXS gguf + a 6 GB /
31 GB box. So every change is proven **byte-identical on Flash** via the existing harness
(golden forced-q8 `DS4_CUDA_Q8_F16_CACHE_MB=0` + decode tier-on==tier-off `--temp 0`
greedy). The Pro / big-RAM / big-VRAM behaviors are **unobservable here** — they are
delivered as code + a portable test script (`tests/scaleup_family_probe.sh`) the user
runs on the A40/128 GB box and feeds back.

---

## Change set (all in `ds4_cuda.cu`, `ds4.c`, `ds4_gpu.h`)

### Plumbing — model topology setter
The CUDA TU can't see the `DS4_N_*` shape macros (it only gets `n_total_expert`/`n_expert`
as call params). P3/P4 need `n_layer` and `n_total_expert` to compute the full expert pool.

- `ds4_gpu.h:48` — add `void ds4_gpu_set_model_topology(uint32_t n_layer, uint32_t n_total_expert);`
- `ds4_cuda.cu` — add globals `g_model_n_layer`, `g_model_n_total_expert`; define the setter.
- `ds4.c:18221` (accelerator setup) — call `ds4_gpu_set_model_topology(DS4_N_LAYER, DS4_N_EXPERT);`

### P0 — Fail-loud capability gates (no Flash behavior change)
Today these `return 0`, which the caller chains (`if (ok) ok = …`) turn into a *skipped*
MoE write → stale `routed_out` → garbage logits, no error. Convert the **capability**
gates (not the geometry soft-fails) to a named fatal via a new `cuda_unsupported_fatal()`.

- `ds4_cuda.cu:10795` routed_moe unsupported `(gate_type,down_type)` → fatal naming the types.
- `ds4_cuda.cu:10796` Q4_K used outside decode/n_expert=6 → fatal (no Q4_K prefill kernel).
- `ds4_cuda.cu:8489` / `8526` router_select / _batch `n_expert≠256 ∥ used≠6 ∥ scale≠1.5` → fatal.
- Keep the null/bounds/`model_size` `return 0`s (legit graph-validation soft-fails).
- **Pro consequence:** a Pro gguf (n_expert=384) dies *loud* at the first router call with a
  clear message — the honest boundary, since the real 384-kernel parameterization (~120 LOC,
  resizes `__shared__ float sprob[4][256]`) is unverifiable here.

### P1 — Shape-derived reserve math (highest reader consensus; provable Flash no-op)
- `ds4_cuda.cu:2266` `planned_reserve`: `256*7078KiB` → `g_model_n_total_expert × (g_slot_bytes ?: 7078KiB)`.
  Flash pre-init: 256 × 7078 KiB = old value exactly. Pro: 384 × … scales. Live path uses exact `g_slot_bytes`.
- `ds4.c:6698` `per_tok = 1572864` → `1572864 × DS4_N_EMBD / 4096`. Flash → 1.5 MiB exactly; Pro(7168) → ~2.6 MiB.

### P2 — Trust GGUF compress-ratios (provable Flash no-op)
- `ds4.c:2915-2921` drop the equality check vs the hardcoded Flash/Pro even-odd pattern; keep
  type/length/non-negative validation + store into `g_ds4_compress_ratios` (already the hot-path source).
- delete now-unused `ds4_expected_layer_compress_ratio` (`ds4.c:606-619`).
- Flash's GGUF ratios equal the old pattern → `g_ds4_compress_ratios` unchanged → byte-identical.

### P3 — Auto-scaling RAM tier = the disk-elision "jump" (sizing only; elision logic already exists)
The host-tier hit path *already* elides disk; only the flat 12 GiB default cap blocks reaching
full-pool coverage. Keep the memoized conservative path **byte-identical** and add an auto-grow
step in `cuda_expram_init` (where `g_slot_bytes` is valid), gated on **env unset AND whole pool
fits with margin**:

```
pool = g_model_n_layer * g_model_n_total_expert * g_slot_bytes
if env DS4_CUDA_EXPERT_RAM_CACHE_GB unset and avail >= pool + 16 GiB and pool > cap:
    cap = pool   # whole expert pool RAM-resident → disk inert after warmup
```
- 31 GB box: pool(~78) + 16 > avail(~27) → branch NOT taken → cap stays 12 GiB default (identical to today).
- 128 GB box: pool + 16 ≤ avail → cap = full pool → after warmup, zero expert disk reads.

### P4 — Diagnostic-only tier-inert logs (no logic change)
- `ds4_cuda.cu:1327` slotbank_init: when `n_slots ≥ g_model_n_layer × g_model_n_total_expert`,
  log "slotbank holds full model expert set — LRU eviction will never fire" (the already-implicit
  VRAM elision). No special no-eviction branch (that path is already correct).

## Explicitly NOT doing (documented, behind P0 fatals)
- Pro 384-expert router kernel parameterization (~120 LOC; resizes shared mem) — unverifiable here.
- Q4_K prefill/batch expert kernel family (~300 LOC) — unverifiable; Q4_K is decode-only today.
- Non-Q8 attention/output/shexp + non-F16 token_embd matmul/embed branches — the `exit(1)`
  validators already fail loud; leave them.
- Perf-only Flash whitelists (q8→f16 dim pairs, wmma `head_dim==128`, router fast-path) — correct
  generic fallback already; defer (no value on a box that can't run Pro).
- `moe_down_sum6` `slot<6u` unroll — unreachable for Flash AND Pro (both n_expert_used=6); a Pro
  gguf dies earlier at the P0 router fatal. Documented, not guarded.

## Verification (each phase, on Flash, here)
1. `make cuda CUDA_ARCH=sm_75` clean; `make` links.
2. Golden forced-q8 (`DS4_CUDA_Q8_F16_CACHE_MB=0`) byte-identical to pristine.
3. Decode tier-on (`GB=12`) vs tier-off (`GB=0`) `--temp 0` greedy → byte-identical tokens.
4. `DS4_CUDA_WEIGHT_CACHE_VERBOSE=1`: confirm planned_reserve ≈ 1.81 GiB, expram cap ≈ unchanged
   on the 31 GB box, and no new fatal fires for Flash.

## Test script for the better box (deliverable)
`tests/scaleup_family_probe.sh` — run on the A40/128 GB box (and against any V4 gguf):
- prints detected VRAM/RAM and the tier sizing the engine chose (slots, expram cap, full-pool/inert logs);
- runs a short greedy generation and a tier-on==tier-off decode-equivalence check;
- if a Pro / other-quant gguf is supplied, exercises the P0 fatals and reports the exact message;
- emits a single summary block for the user to paste back as feedback.
