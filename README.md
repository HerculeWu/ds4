# 🌋 ds4 — run a frontier-scale LLM on the computer you already own

> 🚀 **An 81 GB DeepSeek-V4-Flash model, generating coherent text on a 6 GB
> gaming GPU.** Not in the cloud, not on a $10k workstation — on an ordinary
> desktop. This fork breaks the assumption that big models need big iron.
>
> A fork of [antirez/ds4](https://github.com/antirez/ds4) (the **DwarfStar**
> engine). Fork: <https://github.com/HerculeWu/ds4> · Full engine docs:
> [`README.upstream.md`](./README.upstream.md)

---

## 💥 The breakthrough

Local LLMs have a memory wall. The conventional rule is that the model has to
*fit*: enough RAM, or enough VRAM, to hold the weights. That rule quietly
locks frontier-scale models away from everyone without a high-memory Mac or a
datacenter card.

✅ **This project breaks that rule.** We run DeepSeek-V4-Flash — an 81 GB
mixture-of-experts model that punches far above small local models — on a
consumer machine whose GPU holds barely a fifteenth of it in VRAM. The model
never fits, and it runs anyway.

The key is that a MoE model doesn't *use* all of itself at once. Each token
activates only **6 of 256 experts per layer**. So instead of demanding the
model fit in memory, we stream the few weights each token actually needs
through a three-tier cache — disk to RAM to VRAM — and let the GPU act as a
pure accelerator over whatever fits. Most tokens never touch the disk.

The point isn't one clever machine. The point is generality: if a too-big
model can run on a too-small GPU at a usable speed, then the floor of "what
hardware can run a real LLM" drops out from under the old assumption. 🏠 **The
computer on your desk is very likely already enough.**

## ⚙️ How it works: SSD → RAM → VRAM

```
   SSD (the full ~81 GB model, mmap-backed)
      │   cold experts stream up on demand
      ▼
   Host RAM (warm-expert LRU cache)
      │   hot experts served over PCIe, no disk hit
      ▼
   VRAM (the small slot-bank the GPU computes on)
```

The CPU is the memory manager; the GPU is the accelerator:

- A **VRAM slot-bank** holds only as many experts as your VRAM allows, against
  a ~258-expert-per-token working set. An LRU keeps the hot set resident and
  reuses cross-token carryover.
- A **host-RAM LRU tier** sits between disk and VRAM. On a VRAM miss it serves
  from RAM over PCIe with no disk read. Warm, this tier hits **~65–80 %** of
  the time — close to the locality ceiling we measured before writing a line of
  GPU code.
- The **dense backbone** (attention, shared expert, output head) lives in a
  host-RAM residency cache instead of being re-read from disk every token —
  which, it turned out, was the single biggest early bottleneck.

📈 Whatever memory you bring, more of it simply means a higher hit rate and a
faster decode. The design scales *up* with your machine, not just down to ours.

## 🔧 What we changed relative to upstream

Upstream DwarfStar is a superb, narrow engine for DeepSeek V4 Flash — but its
entry point is big iron (96–128 GB Macs, DGX Spark). Our work, ~3,600 lines
confined to the CUDA/loader path (the Metal path and public contract are
untouched), brings the floor down to a commodity GPU. It went in as phases:

- **Phase 1 — measurement.** CPU diagnostics (`--tensor-budget`,
  `DS4_LOG_ROUTER`) to measure byte budget and expert locality *before*
  writing cache code. Verdict: ~77 % LRU hit rate achievable → proceed.
- **Phase 2 — CUDA slot-bank.** A pure-C, host-unit-tested LRU/hash core
  (`ds4_slotbank_core.h`) wired into the routed-MoE launch — the VRAM tier, and
  the fix for the Turing host-pointer crash.
- **Phase 3 — backbone streaming ring.** A lazy, VRAM-aware streaming ring for
  the dense backbone, per-row `token_embd` gather (no 1 GiB transient per
  embed), VRAM-aware prefill caps, and fail-loud guards so a cache miss can
  never silently corrupt logits.
- **Phase 4 — backbone host-RAM cache.** Keep the backbone in RAM instead of
  re-reading ~8.8 GB from disk every token. ⚡ ~**4.3×** on decode by itself.
- **Phase 5 — routed-expert host-RAM LRU tier.** The RAM tier of the
  hierarchy: a pinned-host slab reusing the same LRU core, feeding the VRAM
  slot-bank. ⚡ ~**1.9×** more on decode.
- **Generalization / scale-up.** Fail-loud capability gates (unsupported
  quant or a 384-expert PRO router now `abort()`s with a named message instead
  of emitting silent garbage), shape-derived sizing, and a RAM tier that
  auto-grows to the full expert pool on a big-memory box — so disk drops out
  entirely after warmup. Still V4-architecture-specific by design; **not** a
  generic GGUF runner.

We kept upstream's rules: mmap-backed loading (no eager full copy), no C++, CPU
path as reference only, diagnostics behind flags.

## 📊 The progress, in token speed

Here is the honest journey, measured on the kind of machine that was never
supposed to load this model — a single GTX 1660 Ti with 6 GB of VRAM, 31 GB of
system RAM, and a plain SSD. Same model, same prompt, greedy decode:

- 🐌 **0.06 t/s** — the naive floor, every token bottlenecked on disk reads.
- 🚶 **~0.30 t/s** — once the backbone stays resident in RAM (Phase 4).
- 🏃 **~0.42 t/s** — once hot experts are cached in RAM too (Phase 5).
- 🚀 **~0.44 t/s** — with a larger warm cache.

That's roughly a **7×** climb end to end, on a six-year-old gaming GPU. If
*that* card can sustain a coherent stream of tokens from an 81 GB model, a more
modern desktop — more VRAM, more RAM, a faster SSD — has more of every resource
the cache feeds on, and the same code simply runs warmer and quicker on it.

🔒 Correctness was non-negotiable: decode with the cache **on** vs **off**
produces **byte-identical** tokens (the cache changes only *where* a weight
comes from, never its value), and the loud capability gates mean an unsupported
configuration stops immediately rather than producing plausible nonsense.

We also learned where the ceiling is. After Phase 5, warm decode is dominated
by fixed per-layer overhead and dequant/matmul compute, **not** weight
movement — the tiering approach is near its practical floor here. The remaining
*multiplicative* lever is MTP / speculative decode, currently blocked because
this GGUF ships with the MTP head stripped.

## 🛠️ Install & use

You need an NVIDIA GPU with CUDA, `nvcc`, a C compiler, and the
DeepSeek-V4-Flash GGUF (~81 GB) on an SSD.

### 1. Get the model

```sh
./download_model.sh          # see the script for the GGUF it fetches
ln -s /path/to/DeepSeek-V4-Flash-IQ2XXS-...gguf ds4flash.gguf
```

### 2. Build (CUDA)

```sh
make cuda CUDA_ARCH=native     # detect the local GPU (easiest)
make cuda CUDA_ARCH=sm_75      # or name your card's compute capability
make cpu                       # CPU-only reference build (debug, slow)
```

This produces `./ds4` (CLI), `./ds4-server` (OpenAI/Anthropic-compatible HTTP
API), plus `ds4-bench`, `ds4-eval`, and `ds4-agent`.

### 3. Run the demo

```sh
./flash-demo.sh "Explain why the sky is blue."
VERBOSE=1 ./flash-demo.sh           # watch the SSD/RAM/VRAM hit rates climb
echo "long prompt..." | ./flash-demo.sh -
```

Or drive the CLI directly:

```sh
./ds4 -m ds4flash.gguf -c 4096 -n 200 --temp 0 \
      -sys "You are a helpful assistant." \
      -p "Write a haiku about tiered memory."
```

The **first token is slow** — the cache is warming from disk. After that it
settles into steady-state decode.

### Tuning knobs

| Environment variable                 | Effect                                                              |
| ------------------------------------ | ------------------------------------------------------------------- |
| `DS4_CUDA_EXPERT_RAM_CACHE_GB`       | Host-RAM expert-cache size in GiB. `0` disables; unset = sensible default; larger = higher hit rate, more RAM used. |
| `DS4_CUDA_WEIGHT_CACHE_VERBOSE=1`    | Print per-layer host/VRAM hit/miss and tier init.                   |
| `DS4_CUDA_SLOTBANK_RESERVE_MB`       | VRAM reserved outside the slot-bank.                                |

A bigger cache helps most in **long, single-session** runs (a coding agent,
say), where cross-token expert reuse keeps climbing past a short benchmark's
average.

## ⚠️ Honest caveat: testing is thin

We want to be upfront: **this fork is under-tested.**

- Almost all measurement happened on **one machine** with **one** IQ2-XXS GGUF.
  We have **not** validated the scale-up path on a bigger box; the portable
  probe (`tests/scaleup_family_probe.sh`) exists for that but hasn't been run
  there yet.
- The **PRO (384-expert)** variant and non-Flash quantizations are **not
  exercised**. They're gated to fail loud — a safety net, not proof they work.
- The **golden long-context vector** (`long_story_4096`) is currently **red**
  on the 6 GB card — identically on pristine upstream and on our branch, so it's
  a pre-existing wide-prefill / low-VRAM issue, not a regression we introduced.
  Still: we have not demonstrated a correct full 4096-token prefill on this
  hardware. We've proven decode is byte-identical cache-on vs cache-off, and
  that short prompts work.
- Performance numbers are wall-clock observations from individual runs, **not**
  a rigorous statistical benchmark, and they're sensitive to VRAM pressure.

Treat this as **beta-quality research code** that clearly works on the hardware
it was built for, with correctness invariants checked carefully where we
checked them — but broad cross-hardware validation is exactly the gap we know
about. Reports from other machines are very welcome.

## 🙏 Acknowledgements

This fork stands entirely on [antirez/ds4 / DwarfStar](https://github.com/antirez/ds4),
which in turn exists thanks to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and GGML. See [`README.upstream.md`](./README.upstream.md) for the full engine
documentation and acknowledgements — all of which we keep and second.
