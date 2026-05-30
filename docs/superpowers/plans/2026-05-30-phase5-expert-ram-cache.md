# Phase 5 — Routed-Expert Host-RAM Cache (the SSD→RAM→VRAM "RAM" tier)

**Goal:** Give the routed experts the host-RAM residency tier the backbone got in
Phase 4, so a VRAM-slot miss during decode is served over PCIe (~13 GB/s) instead
of an O_DIRECT disk read (~0.42 GB/s). Measured target: routed_moe 2.75 s → ~0.7–0.9 s,
warm token 3.4 s → ~1.5 s → ~0.30 → ~0.6 t/s (~2×), composing later with MTP (Phase 6).

**Architecture:** A *second* `cuda_slotbank` instance, `g_expram`, structurally
identical to the VRAM `g_slotbank` but with its slot pointers indexing one pinned
host slab (`cudaHostAlloc`) instead of a device slab. It sits between disk and the
VRAM slotbank as a larger, LRU-bounded tier. The pure-C LRU/hash core in
`ds4_slotbank_core.h` is reused verbatim (it has zero CUDA symbols by design).

**Tech stack:** C/CUDA in `ds4_cuda.cu`; reuse `ds4_slotbank_core.h`.

---

## Why this shape

- The VRAM slotbank holds ~273 slots; one decode token needs 258 distinct
  (layer,expert) pairs → it thrashes, so ~every expert is a VRAM miss every token,
  and today every miss reads disk. The host tier (~1700–2700 slots at 12–18 GiB)
  holds a cross-token working set, so most VRAM misses become host hits (H2D), not disk.
- Experts are uniform fixed-size (gate|up|down), keyed by (layer, expert_id) — the
  exact shape `cuda_slotbank` + `ds4_slotbank_core.h` already model. So the host tier
  is the same data structure, just host-backed. No new LRU code.
- Decode-only (reuse `g_bb_cache_armed`): prefill streams experts from disk exactly
  as today → prefill stays byte-identical → the local-golden gate is unaffected.
- Kept **separate** from `bb_ram` (not unified) on purpose: backbone (8.8 GB) fits
  RAM unbounded/append-only; experts (77.9 GB) cannot and need LRU+cap. bb_ram is
  correctness-critical (the prefill zero-logit history) and works — do not touch it.

## Invariants preserved (from the code map)

1. Key = (layer, expert_id); same `sb_*` primitives; `hits/misses/evictions` come free.
2. Host serve = `cudaMemcpyAsync` H2D on `g_model_upload_stream`, batched into the
   one `cudaStreamSynchronize` at `cuda_slotbank_ensure_union` (ds4_cuda.cu:1506).
   Do NOT add a sync inside the serve.
3. Host capture = `cudaMemcpy` D2H from the freshly disk-filled VRAM slot, run AFTER
   that sync (mirrors `bb_ram_insert`'s "after the stream sync" discipline).
4. Best-effort everywhere: any `cudaHostAlloc`/`cudaMemcpy`/`malloc` failure leaves
   the expert disk-only and correct. Never abort, never hand a host ptr to a kernel.
5. The bytes served from RAM are a D2H copy of a prior correct disk fill of the SAME
   expert → bit-identical to streaming from disk → decode output unchanged.
6. `cuda_is_device_ptr` assertions apply only to `g_slotbank` (device) slots; never
   to `g_expram` (host) slots.

## Files / edits (all in ds4_cuda.cu unless noted)

- **Globals** after `g_slot_id_scratch_cap` (~ds4_cuda.cu:263): `g_expram`,
  `g_expram_base`, `g_expram_ready`.
- **Helpers** just above `cuda_slotbank_ensure_union` (~ds4_cuda.cu:1446):
  `host_mem_available_bytes`, `expram_cap_bytes`, `expram_enabled`,
  `cuda_expram_init`, `cuda_expram_serve`, `cuda_expram_capture`.
- **`cuda_slotbank_ensure_union`** (1457): lazy-init the tier and compute
  `use_expram = g_bb_cache_armed && expram_enabled()` at the top; in the miss branch,
  consult the host tier before `cuda_slotbank_fill`; collect disk-miss captures;
  flush them after the sync; free scratch on every exit path.
- **`cuda_slotbank_release_all`** (1532): also `cudaFreeHost` the host slab + free arrays.

## Env

- `DS4_CUDA_EXPERT_RAM_CACHE_GB` — host tier cap in GiB. Unset = default **12 GiB**.
  `0` = disable the tier (experts stream from disk as before). Clamped so the pinned
  slab never exceeds `MemAvailable − 10 GiB` (headroom for bb_ram ~7.2 GiB + OS slack).
- `DS4_CUDA_WEIGHT_CACHE_VERBOSE` — prints the tier size at init and per-layer
  host/VRAM hit/miss counters (the "first cheap step": turns "81% disk-bound" into a
  measured number).

## Validation

1. `make ds4_cuda.o CUDA_ARCH=sm_75` — compiles clean.
2. `make ds4 CUDA_ARCH=sm_75` — links.
3. **Golden gate (correctness):** `DS4_TEST_MODEL=ds4flash.gguf ./ds4_test
   --local-golden-vectors` stays green. The tier is decode-armed only, so prefill
   (what the golden test exercises) is byte-identical; this is the safety guarantee.
4. **Warm-decode profile (the win):** run a short decode with
   `DS4_METAL_DECODE_STAGE_PROFILE=1 DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` and confirm
   routed_moe collapses on warm tokens and the host hit rate climbs after token 1.
   Manage the 6 GB VRAM with the existing footprint knobs (`-c 2048`,
   `DS4_CUDA_BACKBONE_RING_MB=256`, `DS4_CUDA_SLOTBANK_RESERVE_MB=1100`) as in the
   Phase-5 measurement run, without killing the user's desktop apps.

## Risks

- **Host RAM pressure** (pinned, non-swappable): bb_ram ~7.2 GiB + expert tier ≤ ~17 GiB.
  Mitigated by the conservative default + `MemAvailable − 10 GiB` clamp + graceful
  shrink-on-`cudaHostAlloc`-failure (halve until it fits, else disable the tier).
- **Token-1 warmup cost:** up to ~258×3 D2H captures on the first decode token
  (~130 ms one-time); steady-state captures only the host-miss tail.
- **Prefill safety:** guaranteed byte-identical because `use_expram` is false whenever
  `g_bb_cache_armed == 0` (set/cleared unconditionally by the decode/prefill layer tops).
