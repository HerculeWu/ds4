#ifndef DS4_BACKBONE_RING_CORE_H
#define DS4_BACKBONE_RING_CORE_H
/* Pure-C core of the Phase 3 backbone streaming tier. No CUDA symbols here:
   this header is shared verbatim by ds4_cuda.cu (the real ring + cudaMalloc +
   staged uploads) and tests/cuda_backbone_ring_test.c (host-only unit tests).
   It owns two policy structures:
     - bbr_registry: the set of every backbone tensor's (abs_offset,bytes),
       built once at model open, queried by the resolver to POSITIVELY decide
       "is this offset backbone?" (vs. inferring "anything not cached").
     - bbr_ring_state: a per-layer epoch ring. Each layer reset bumps the epoch
       and frees all bytes; within one epoch a repeated offset is deduped (hit),
       a new offset that fits is appended (miss-fits), and one too big for the
       ring or one beyond the dedup-table capacity returns -1 (caller uses the
       transient cudaMalloc/free path). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t off; uint64_t bytes; } bbr_span;

typedef struct {
    bbr_span *spans;
    uint32_t  n;
    uint32_t  cap;
    int       sorted;
} bbr_registry;

static inline void bbr_registry_init(bbr_registry *r, uint32_t cap_hint) {
    r->cap = cap_hint < 16 ? 16 : cap_hint;
    r->spans = (bbr_span *)malloc((size_t)r->cap * sizeof(bbr_span));
    r->n = 0;
    r->sorted = 0;
}
static inline void bbr_registry_free(bbr_registry *r) {
    free(r->spans); r->spans = NULL; r->n = 0; r->cap = 0; r->sorted = 0;
}
static inline void bbr_registry_add(bbr_registry *r, uint64_t off, uint64_t bytes) {
    if (bytes == 0) return;
    if (r->n == r->cap) {
        r->cap *= 2;
        r->spans = (bbr_span *)realloc(r->spans, (size_t)r->cap * sizeof(bbr_span));
    }
    r->spans[r->n].off = off;
    r->spans[r->n].bytes = bytes;
    r->n++;
    r->sorted = 0;
}
static int bbr_span_cmp(const void *a, const void *b) {
    const bbr_span *x = (const bbr_span *)a, *y = (const bbr_span *)b;
    if (x->off < y->off) return -1;
    if (x->off > y->off) return 1;
    return 0;
}
static inline void bbr_registry_sort(bbr_registry *r) {
    if (r->n > 1) qsort(r->spans, r->n, sizeof(bbr_span), bbr_span_cmp);
    r->sorted = 1;
}
/* 1 if [off, off+bytes) is fully contained in a single registered span. */
static inline int bbr_registry_contains(const bbr_registry *r, uint64_t off, uint64_t bytes) {
    if (r->n == 0 || !r->sorted) return 0;
    const uint64_t end = off + bytes;
    if (end < off) return 0;            /* overflow */
    /* binary search for the last span whose .off <= off */
    uint32_t lo = 0, hi = r->n;          /* find first span with .off > off */
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (r->spans[mid].off <= off) lo = mid + 1; else hi = mid;
    }
    if (lo == 0) return 0;               /* off is before the first span */
    const bbr_span *s = &r->spans[lo - 1];
    return (off >= s->off && end <= s->off + s->bytes) ? 1 : 0;
}

#define BBR_DEDUP_CAP 96   /* worst layer touches attn+shared+compressor+indexer+
                              mixers+norms+router+sinks distinct offsets; 96 is
                              comfortably above the measured ~30 (see Task 6). */

typedef struct { uint64_t off; uint64_t ring_off; uint32_t epoch; uint8_t used; } bbr_dedup_ent;

typedef struct {
    bbr_dedup_ent ents[BBR_DEDUP_CAP];
    uint32_t n_ents;   /* entries used THIS epoch */
    uint64_t cap;      /* ring capacity in bytes */
    uint64_t used;     /* bytes consumed this epoch */
    uint32_t epoch;
    /* observability */
    uint64_t hits, miss_fits, no_fits, hiwater;
} bbr_ring_state;

static inline void bbr_ring_init(bbr_ring_state *s, uint64_t cap_bytes) {
    memset(s, 0, sizeof(*s));
    s->cap = cap_bytes;
    s->epoch = 1;   /* epoch 0 reserved as "never seen" sentinel for ents */
}
static inline void bbr_ring_reset(bbr_ring_state *s) {
    s->epoch++;
    s->used = 0;
    s->n_ents = 0;
}
/* Returns: 1 = hit (ring_off = existing slot, no re-upload needed),
            0 = miss-but-fits (ring_off = freshly allocated slot, used advanced),
           -1 = does not fit (oversized, or dedup table full) -> caller uses transient. */
static inline int bbr_ring_acquire(bbr_ring_state *s, uint64_t off, uint64_t bytes_aligned,
                                   uint64_t *ring_off) {
    for (uint32_t i = 0; i < s->n_ents; i++) {
        if (s->ents[i].epoch == s->epoch && s->ents[i].used && s->ents[i].off == off) {
            *ring_off = s->ents[i].ring_off;
            s->hits++;
            return 1;
        }
    }
    if (s->used + bytes_aligned > s->cap) { s->no_fits++; return -1; }   /* oversized */
    if (s->n_ents >= BBR_DEDUP_CAP)        { s->no_fits++; return -1; }   /* table full */
    const uint64_t ro = s->used;
    s->ents[s->n_ents].off = off;
    s->ents[s->n_ents].ring_off = ro;
    s->ents[s->n_ents].epoch = s->epoch;
    s->ents[s->n_ents].used = 1;
    s->n_ents++;
    s->used += bytes_aligned;
    if (s->used > s->hiwater) s->hiwater = s->used;
    *ring_off = ro;
    s->miss_fits++;
    return 0;
}

#endif /* DS4_BACKBONE_RING_CORE_H */
