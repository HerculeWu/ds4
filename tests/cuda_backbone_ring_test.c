/* Host-only unit tests for ds4_backbone_ring_core.h. No CUDA. Built by
   `make cuda-backbone-ring-test`, mirroring tests/cuda_slotbank_test.c. */
#include "../ds4_backbone_ring_core.h"
#include <stdio.h>
#include <assert.h>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

int main(void) {
    /* --- registry: add / sort / containment (binary search) --- */
    bbr_registry reg;
    bbr_registry_init(&reg, 4);
    bbr_registry_add(&reg, 5000, 100);   /* [5000,5100) */
    bbr_registry_add(&reg, 1000, 200);   /* [1000,1200) */
    bbr_registry_add(&reg, 8000, 50);    /* [8000,8050) */
    bbr_registry_sort(&reg);
    /* exact span */
    CHECK(bbr_registry_contains(&reg, 1000, 200) == 1);
    /* sub-range inside a span (e.g. a single token_embd row is a subset) */
    CHECK(bbr_registry_contains(&reg, 5040, 10) == 1);
    CHECK(bbr_registry_contains(&reg, 8000, 50) == 1);
    /* miss: gap between spans */
    CHECK(bbr_registry_contains(&reg, 1200, 10) == 0);
    CHECK(bbr_registry_contains(&reg, 2000, 10) == 0);
    /* miss: query straddles span end (off inside, end past) */
    CHECK(bbr_registry_contains(&reg, 5090, 50) == 0);
    /* miss: off before the first span start */
    CHECK(bbr_registry_contains(&reg, 999, 1) == 0);
    /* max-bytes: largest span (used to size the slotbank reserve so the
       output-head transient always fits) */
    CHECK(bbr_registry_max_bytes(&reg) == 200);   /* [1000,1200) is biggest */
    bbr_registry_add(&reg, 20000, 4096);
    CHECK(bbr_registry_max_bytes(&reg) == 4096);   /* picks up the new largest */
    bbr_registry_free(&reg);
    { bbr_registry empty; bbr_registry_init(&empty, 2);
      CHECK(bbr_registry_max_bytes(&empty) == 0);   /* empty -> 0 */
      bbr_registry_free(&empty); }

    /* --- ring: reset / acquire hit / miss-fits / doesn't-fit / table-full --- */
    bbr_ring_state ring;
    bbr_ring_init(&ring, 1000);   /* 1000-byte ring */
    bbr_ring_reset(&ring);
    uint64_t ro = 0xdead;

    /* miss-fits: first acquire of off=100,bytes=256 -> ring_off=0, used=256 */
    CHECK(bbr_ring_acquire(&ring, 100, 256, &ro) == 0);
    CHECK(ro == 0);
    /* hit: same offset same epoch -> returns existing ring_off, used unchanged */
    ro = 0xdead;
    CHECK(bbr_ring_acquire(&ring, 100, 256, &ro) == 1);
    CHECK(ro == 0);
    /* miss-fits: second distinct offset -> ring_off=256 */
    CHECK(bbr_ring_acquire(&ring, 200, 256, &ro) == 0);
    CHECK(ro == 256);
    /* doesn't-fit: 600 more would exceed 1000 (512 used) -> -1 */
    CHECK(bbr_ring_acquire(&ring, 300, 600, &ro) == -1);
    /* reset: new epoch, old dedup entries invalid, used back to 0 */
    bbr_ring_reset(&ring);
    CHECK(bbr_ring_acquire(&ring, 100, 256, &ro) == 0);   /* miss again, fresh epoch */
    CHECK(ro == 0);

    /* --- ring: dedup table full returns -1 (transient), never corrupts --- */
    bbr_ring_state tiny;
    bbr_ring_init(&tiny, 1ull << 40);   /* huge ring so BYTES never the limiter */
    bbr_ring_reset(&tiny);
    int got_full_signal = 0;
    for (uint32_t i = 0; i < BBR_DEDUP_CAP + 5; i++) {
        int r = bbr_ring_acquire(&tiny, 100000 + (uint64_t)i * 4096, 256, &ro);
        if (r == -1) got_full_signal = 1;   /* once table is full, further distinct offsets -> -1 */
    }
    CHECK(got_full_signal == 1);

    /* --- subset semantics relied on by the resolver: a registered FULL tensor
           span must contain any in-bounds sub-slice query (used to reject a
           query that does NOT match; token_embd is handled separately). --- */
    {
        bbr_registry e; bbr_registry_init(&e, 2);
        bbr_registry_add(&e, 1000000, 1059u * 1024u * 1024u); /* ~token_embd span */
        bbr_registry_sort(&e);
        CHECK(bbr_registry_contains(&e, 1000000, 8192) == 1);            /* a row-sized slice */
        CHECK(bbr_registry_contains(&e, 1000000, 1059u*1024u*1024u) == 1); /* full span */
        CHECK(bbr_registry_contains(&e, 999999, 8192) == 0);             /* before start */
        bbr_registry_free(&e);
    }

    if (failures) { fprintf(stderr, "cuda_backbone_ring_test: %d FAILURES\n", failures); return 1; }
    printf("cuda_backbone_ring_test: all passed\n");
    return 0;
}
