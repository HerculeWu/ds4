# Tiered-MoE Roadmap Revision (post Phase-4) — Phase 5+

**Date:** 2026-05-30
**Trigger:** Phase 4 shipped the backbone host-RAM cache (decode ~0.07→0.30 t/s). Its
parting hypothesis ("next lever = coalesce 1185 tiny H2D transfers; check PCIe Gen1")
was never validated. This revision **re-measures on the now-running model** and
**reverses that hypothesis**, backed by both local measurement and a fact-checked
web survey (FlexGen / ZeRO-Inference / PIPO / PowerInfer / nvCOMP / GDS).

**Goal of the project (unchanged):** *run* DeepSeek-V4-Flash (86.7 GB, 43 layers,
256 routed experts) on a 6 GB GTX 1660 Ti (sm_75, PCIe 3.0) via SSD→RAM→VRAM
tiering. Push the floor down; not max throughput. "Floor model" = no sqlite yet.

---

## 1. What the measurements say (this is the whole story)

### 1a. PCIe and transfer coalescing are NOT levers (both refuted)
Microbenchmark `misc/h2d_bw.cu` (pinned H2D, nvcc sm_75), with `nvidia-smi` polled under load:

- The 1660 Ti idles at PCIe **Gen1/2** (power save) but **trains to Gen3 x16 under load** (P0/P2).
- Pinned H2D hits **13.0 GB/s** = the full PCIe 3.0 x16 realizable ceiling.
- "1185 copies + per-copy sync" pattern = **12.81 GB/s**; one coalesced copy = **13.02 GB/s**.
  → **Coalescing buys ~1.6%.** Small *pinned* transfers already saturate the link
  (≥256 KiB → 12+ GB/s; only 64 KiB dips to 10). **Do not pursue coalescing or PCIe tuning.**

### 1b. Warm decode is dominated by ROUTED EXPERTS, not the backbone
Per-stage decode profile (`DS4_METAL_DECODE_STAGE_PROFILE=1`), summed over 43 layers,
steady warm token ≈ 3.4 s (cache armed). Cold token (cache fill) for contrast:

| stage              | **warm (cached)** | cold     | warm drop vs cold |
|--------------------|-------------------|----------|-------------------|
| **routed_moe**     | **2753 ms (81%)** | 5630 ms  | only **2×**       |
| attn_output (bb)   | 262 ms            | 12239 ms | **47×**           |
| q_path (bb)        | 166 ms            | 7729 ms  | 46×               |
| shared_gate_up (bb)| 75 ms             | 3346 ms  | 45×               |
| compressor_indexer | 70 ms             | 2637 ms  | 38×               |
| shared_down (bb)   | 51 ms             | 1681 ms  | 33×               |
| everything else    | <20 ms each       | —        | —                 |

**The Phase-4 backbone RAM cache works spectacularly** — every backbone stage collapsed ~45×.
But **routed_moe (the 6 routed experts/layer) is now 81% of the warm token and barely moved**,
because experts have **no RAM tier**: a slotbank miss reads straight from disk
(`cuda_slotbank_fill → cuda_slotbank_one_component → cuda_model_stage_read`, O_DIRECT ~420 MB/s).
The code comment at `ds4_cuda.cu:95` ("experts stay on disk — they rarely miss") is **false for decode**.

**Why experts thrash (structural, not a measurement artifact):** one decode token's working set is
6 experts × 43 layers = **258 distinct (layer,expert) pairs**. The VRAM slotbank holds ~273 slots
(6 GB minus ~1.1 GB desktop graphics minus reserve), so it barely fits *one* token; the next token
routes to a different 258 → near-total eviction → ~half re-read from disk every token. The full expert
space (256×43 = 11008 instances / 77.9 GB) dwarfs any 6 GB VRAM slotbank, so **experts will always
thrash without a host-RAM tier**, on any headless/normal config too.

> Note: routed_moe is **not** compute — 6×6.75 MiB/layer ≈ 40 MiB of weights, ~0.06 ms of FLOPs.
> It is pure weight *movement* (disk vs RAM). This is why overlap can't help (1c).

### 1c. Web survey converges with the measurements (15 confirmed / 10 killed claims)
- **Compute/transfer OVERLAP does not help single-stream decode** — confirmed 3-0 from ZeRO-Inference's
  own result: "prefetching gave NO improvement during token generation; layer compute time too small to
  hide fetch." [deepspeed.ai/2022/09/09/zero-inference, arxiv 2303.06865]
- **FlexGen / PIPO / ZeRO speedups are *batched throughput*, not per-token latency** — confirmed
  (FlexGen batch 144; PIPO "max throughput"). Their zig-zag "load weights once, reuse across batch"
  was **refuted 0-3 for our single-stream case** — it *requires* batching. [arxiv 2303.06865, 2504.03664]
- **Volume reduction is mostly a dead end on this rig:** 4-bit cuts H2D 4× but **sm_75 has no INT4**
  + 20-90% dequant overhead, and the model is already IQ2 [arxiv 2411.01433]; lossless/nvCOMP on float
  weights only **1.1–1.5×** [nvCOMP blog, arxiv 2603.17435]; activation-sparsity offload is **ReLU-family
  only** (DeepSeek isn't) and CPU-hybrid cold-neuron compute **refuted 0-3** [PowerInfer, arxiv 2312.12456];
  **GDS/cuFile unavailable on GeForce** [docs.nvidia.com/gpudirect-storage].
- What *remains viable* = exactly the **SSD→RAM→VRAM tiered residency** (FlexGen/ZeRO's core: weights in
  host RAM, streamed over PCIe) + **speculative/multi-token decode** as the only single-stream "batching."

**Conclusion: the one big, viable, measured lever is to give the experts the host-RAM tier the backbone
already has.** That is the project's original thesis; Phase 4 built it for the backbone only.

---

## 2. Revised roadmap (priority order)

### ✅ Phase 4 (done): backbone host-RAM residency cache — decode 0.07→0.30 t/s

### ▶ Phase 5 (NEXT, dominant lever): routed-expert host-RAM cache
Give slotbank misses a host-RAM tier before disk — the SSD→RAM→VRAM "RAM" tier (Phase-1 validated
~77% LRU hit at 17.5 GB; box has ~27 GB free). Mirrors the Phase-4 `bb_ram_*` design but **LRU-bounded**
(the 77.9 GB expert pool can't fully fit RAM, unlike the 8.8 GB backbone which fits).

**Expected:** routed_moe 2.75 s → ~0.4 s (RAM 13 GB/s vs disk 0.42 GB/s on the ~77% that hit),
warm token 3.4 s → ~1.0 s → **~0.30 → ~0.85–1.0 t/s (~3× on top of Phase 4).**

**Design (where it hooks):**
- New host-RAM expert cache: pinned (`cudaHostAlloc`) arena, ~17–20 GB (env `DS4_CUDA_EXPERT_RAM_CACHE_GB`,
  default sized from free RAM; `=0` disables). Keyed by absolute GGUF offset (gate/up/down are 3 keys,
  or one key per slot's contiguous span). LRU eviction (host-side, big and cheap; reuse the slotbank's
  intrusive-LRU pattern from `ds4_slotbank_core.h`).
- Hook in `cuda_slotbank_one_component(off, bytes, dst)` — the single disk primitive both expert-fill and
  backbone-miss share. On entry: lookup `off` in the RAM cache; **hit** → `cudaMemcpyAsync` H2D from the
  pinned host copy (skip disk); **miss** → existing O_DIRECT staged read into `dst`, then D2H-capture
  `dst`→RAM (respecting the cap + LRU evict). This automatically covers experts AND any backbone miss,
  superseding the separate `bb_ram` layer — consider unifying (one tiered cache, two size classes) to
  keep the codebase minimal (project rule: no permanent parallel variants).
- Keep it **decode-armed only**, exactly like Phase 4 (`g_bb_cache_armed`), because the 4096-token batch
  prefill path has the known zero-logits bug with host-cache serving. Prefill stays byte-identical/streamed.

**First cheap step (confirm the miss rate):** the counters `g_slotbank.hits/.misses` already exist but are
never printed. Add a one-line per-token report (behind `DS4_CUDA_WEIGHT_CACHE_VERBOSE`) to quantify the
decode miss rate before/after — turns the "81% disk-bound" inference into a measured number and validates the win.

**Risks:** (a) host RAM pressure — cap it, leave ≥8 GB headroom (31 GB box). (b) pinned-memory limit — if
`cudaHostAlloc` of 17 GB fails, fall back to pageable (still ~6–10 GB/s, far above disk) or shrink the cap.
(c) the prefill batch-path bug must stay scoped out (decode-armed only).

**Gate:** `./ds4_test --local-golden-vectors` must stay green (decode-armed cache leaves prefill
byte-identical). Plus a manual warm-decode profile showing routed_moe collapse.

### Phase 6 (complementary, stacks multiplicatively): multi-token / speculative decode (MTP)
The web survey's clearest "big win" is **amortizing weight movement across a batch** — impossible for a
single token, but DeepSeek-V4-Flash ships an **MTP head**. Verifying K draft tokens in one forward pass
reuses each weight load (backbone *and* experts) across K tokens → divides per-token weight streaming by
the acceptance length. This is the single-stream analog of FlexGen's batching and was already flagged in
Phase-1 planning. Do it **after** the RAM tier (so the reused loads come from RAM, not disk).

### Phase 7 (situational): trim per-token VRAM↔residency
- If run **headless** (frees ~1.1 GB desktop VRAM), pin the hottest experts and/or a backbone slice
  resident — the slotbank is currently *greedy* (claims all free VRAM beyond reserve; floor = full
  256-union, `ds4_cuda.cu:10603`). With a RAM tier behind it, a deliberately *smaller* VRAM slotbank
  hurts far less, freeing VRAM for resident hot weights.
- Output head (~0.99 GiB transient/token) is the largest single span; if VRAM allows, keep it resident.

### ❌ Killed / deprioritized (with evidence — do not revisit without new data)
| lever | verdict | why |
|---|---|---|
| coalesce 1185 H2D transfers | **dead** | local: 12.81 vs 13.02 GB/s (<2%) |
| PCIe Gen1 "fix" | **dead** | trains to Gen3 x16 under load |
| compute/transfer overlap, double-buffer prefetch | **dead (single-stream)** | compute ~nil; ZeRO/FlexGen confirm no gain without batch |
| further quantization (INT4) | **dead on sm_75** | no INT4 HW + dequant cost; already IQ2 |
| weight compression (nvCOMP/lossless) | **not worth it** | 1.1–1.5× on float weights |
| activation-sparsity / CPU-hybrid (PowerInfer) | **N/A** | ReLU-family only; CPU-compute refuted 0-3 |
| GPUDirect Storage / cuFile | **unavailable** | GeForce fails the SKU gate |

---

## 3. One-line summary
Phase 4 cached the backbone in RAM and the backbone problem vanished (45×). The remaining 81% of decode is
routed experts still hitting disk because they were never given the RAM tier. **Phase 5 = expert host-RAM
LRU cache** (the project's original SSD→RAM→VRAM design), est. **~3× → ~1 t/s**. Then **MTP** to amortize
the per-token streaming. Overlap, coalescing, PCIe, quant, and compression are measured/cited dead ends here.
