#include <stdio.h>
#include <stdlib.h>
#include "../ds4_slotbank_core.h"

/* Build a bank of n empty slots, all chained on the LRU list, no pinned. Slab
   pointers faked so the core is exercised with zero CUDA. */
static cuda_slotbank *make_bank(uint32_t n) {
    cuda_slotbank *sb = calloc(1, sizeof(*sb));
    sb->n_slots = n;
    sb->slots = calloc(n, sizeof(cuda_expert_slot));
    uint32_t cap = 1; while (cap < n * 2u) cap <<= 1;
    sb->hmask = cap - 1u;
    sb->htab = malloc(cap * sizeof(uint32_t));
    for (uint32_t i = 0; i < cap; i++) sb->htab[i] = SLOT_NIL;
    sb->lru_head = sb->lru_tail = SLOT_NIL;
    for (uint32_t i = 0; i < n; i++) {
        sb->slots[i].layer = SB_FREE_LAYER;
        sb->slots[i].lru_prev = sb->slots[i].lru_next = sb->slots[i].hnext = SLOT_NIL;
        sb_lru_push_head(sb, i); /* empties available; tail = first victim */
    }
    return sb;
}
static uint32_t fill_slot(cuda_slotbank *sb, uint32_t s, uint32_t layer, uint32_t eid) {
    sb->slots[s].layer = layer; sb->slots[s].expert_id = eid;
    sb->slots[s].resident = 1;
    sb_hash_insert(sb, s);
    sb_touch(sb, s);
    return s;
}
/* Mimic the device-side ensure-union admission core (no CUDA): for each id,
   hit-touch or miss-evict-fill. Returns 1 on success, 0 if no evictable slot. */
static int ensure_union(cuda_slotbank *sb, uint32_t layer, const int32_t *ids, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t eid = (uint32_t)ids[i];
        uint32_t s = sb_lookup(sb, layer, eid);
        if (s != SLOT_NIL) { sb->hits++; sb_touch(sb, s); continue; }
        sb->misses++;
        s = sb_acquire(sb);
        if (s == SLOT_NIL) return 0;
        sb_evict(sb, s);
        fill_slot(sb, s, layer, eid);
    }
    return 1;
}

static int test_lookup_hit_miss(void) {
    cuda_slotbank *sb = make_bank(4);
    if (sb_lookup(sb, 3, 10) != SLOT_NIL) { fprintf(stderr,"miss expected\n"); return 1; }
    uint32_t s = sb_acquire(sb); fill_slot(sb, s, 3, 10);
    if (sb_lookup(sb, 3, 10) != s) { fprintf(stderr,"hit expected\n"); return 1; }
    if (sb_lookup(sb, 3, 11) != SLOT_NIL) { fprintf(stderr,"other miss expected\n"); return 1; }
    fprintf(stderr,"test_lookup_hit_miss: PASS\n"); return 0;
}
static int test_lru_evicts_oldest(void) {
    cuda_slotbank *sb = make_bank(2);
    uint32_t a = sb_acquire(sb); fill_slot(sb, a, 0, 1);   /* fill A */
    uint32_t b = sb_acquire(sb); fill_slot(sb, b, 0, 2);   /* fill B (now MRU) */
    sb_touch(sb, a);                                        /* A MRU, B LRU */
    uint32_t v = sb_acquire(sb);                            /* victim must be B */
    if (v != b) { fprintf(stderr,"expected B as victim got %u\n", v); return 1; }
    sb_evict(sb, v); fill_slot(sb, v, 0, 3);
    if (sb_lookup(sb, 0, 2) != SLOT_NIL) { fprintf(stderr,"B should be evicted\n"); return 1; }
    if (sb_lookup(sb, 0, 1) == SLOT_NIL) { fprintf(stderr,"A must survive\n"); return 1; }
    if (sb_lookup(sb, 0, 3) == SLOT_NIL) { fprintf(stderr,"C must be resident\n"); return 1; }
    fprintf(stderr,"test_lru_evicts_oldest: PASS\n"); return 0;
}
static int test_pinned_never_evicted(void) {
    cuda_slotbank *sb = make_bank(2);
    uint32_t a = sb_acquire(sb); fill_slot(sb, a, 0, 1);
    sb_lru_unlink(sb, a); sb->slots[a].pinned = 1; sb->n_pinned++;  /* pin A */
    uint32_t b = sb_acquire(sb); fill_slot(sb, b, 0, 2);
    uint32_t v = sb_acquire(sb);
    if (v != b) { fprintf(stderr,"pinned A must not be victim, got %u\n", v); return 1; }
    fprintf(stderr,"test_pinned_never_evicted: PASS\n"); return 0;
}
static int test_no_key_collision_across_layers(void) {
    cuda_slotbank *sb = make_bank(4);
    uint32_t a = sb_acquire(sb); fill_slot(sb, a, 1, 5);
    uint32_t b = sb_acquire(sb); fill_slot(sb, b, 2, 5);   /* same expert, diff layer */
    if (sb_lookup(sb,1,5) == sb_lookup(sb,2,5)) { fprintf(stderr,"key collision!\n"); return 1; }
    if (sb_lookup(sb,1,5) != a || sb_lookup(sb,2,5) != b) { fprintf(stderr,"wrong slot\n"); return 1; }
    fprintf(stderr,"test_no_key_collision_across_layers: PASS\n"); return 0;
}
/* The prefill blocker scenario: a union with duplicate ids (as in
   n_tokens*n_expert selected[]) must coalesce to distinct residents and fit
   when n_slots >= distinct count. */
static int test_union_dedup_fits(void) {
    cuda_slotbank *sb = make_bank(8);
    /* 12 selected entries (mimics 2 tokens * 6); the distinct union is {2,3,7,11}. */
    int32_t ids[12] = {3,7,3,7,11,3,2,7,11,3,2,7};
    if (!ensure_union(sb, 4, ids, 12)) { fprintf(stderr,"union must fit in 8 slots\n"); return 1; }
    uint32_t distinct = 0;
    for (uint32_t i = 0; i < sb->n_slots; i++) if (sb->slots[i].resident) distinct++;
    if (distinct != 4) { fprintf(stderr,"expected 4 distinct residents, got %u\n", distinct); return 1; }
    const int32_t want[5] = {3,7,11,2,3};
    for (uint32_t i = 0; i < 5; i++)
        if (sb_lookup(sb, 4, (uint32_t)want[i]) == SLOT_NIL && want[i] != 3) {
            fprintf(stderr,"missing union member %d\n", want[i]); return 1; }
    fprintf(stderr,"test_union_dedup_fits: PASS\n"); return 0;
}
/* Mirror of cuda_slotbank_ensure_union's out_slot contract: for the FULL
   n_tokens*n_expert id array (with duplicates) produce a per-id physical slot
   index. Returns 1 on success. This is the host analog of the device
   ensure_union + k_remap_selected_full pair: out_slot[i] is what every
   selected[i] entry is rewritten to before the count/scatter/tile kernels. */
static int ensure_union_remap(cuda_slotbank *sb, uint32_t layer,
                              const int32_t *ids, uint32_t n, uint32_t *out_slot) {
    for (uint32_t i = 0; i < n; i++) {
        uint32_t eid = (uint32_t)ids[i];
        uint32_t s = sb_lookup(sb, layer, eid);
        if (s == SLOT_NIL) {
            sb->misses++;
            s = sb_acquire(sb);
            if (s == SLOT_NIL) return 0;
            sb_evict(sb, s);
            fill_slot(sb, s, layer, eid);
        } else {
            sb->hits++;
            sb_touch(sb, s);
        }
        out_slot[i] = s;
    }
    return 1;
}
/* The Task 6 remap contract: rewriting EVERY selected[] entry to its physical
   slot index must (a) map duplicate expert ids to the SAME slot, (b) keep every
   slot index in [0,n_slots), and (c) preserve the kernel-visible expert->bucket
   identity (two entries share a bucket iff they share an expert id). */
static int test_union_out_slot_remap(void) {
    cuda_slotbank *sb = make_bank(8);
    /* 12 entries (2 tokens * 6), distinct union {2,3,7,11}. */
    int32_t ids[12] = {3,7,3,7,11,3,2,7,11,3,2,7};
    uint32_t slot[12];
    if (!ensure_union_remap(sb, 4, ids, 12, slot)) {
        fprintf(stderr,"remap union must fit in 8 slots\n"); return 1; }
    for (uint32_t i = 0; i < 12; i++) {
        if (slot[i] >= sb->n_slots) {
            fprintf(stderr,"slot %u out of range at i=%u\n", slot[i], i); return 1; }
        for (uint32_t j = 0; j < 12; j++) {
            int same_id = (ids[i] == ids[j]);
            int same_slot = (slot[i] == slot[j]);
            if (same_id != same_slot) {
                fprintf(stderr,"remap identity broken: ids[%u]=%d ids[%u]=%d "
                        "slot[%u]=%u slot[%u]=%u\n",
                        i, ids[i], j, ids[j], i, slot[i], j, slot[j]);
                return 1;
            }
        }
    }
    /* Each distinct slot must point at the right resident expert. */
    for (uint32_t i = 0; i < 12; i++) {
        if (sb->slots[slot[i]].layer != 4u ||
            sb->slots[slot[i]].expert_id != (uint32_t)ids[i] ||
            !sb->slots[slot[i]].resident) {
            fprintf(stderr,"slot %u does not hold L4 E%d resident\n", slot[i], ids[i]);
            return 1;
        }
    }
    fprintf(stderr,"test_union_out_slot_remap: PASS\n"); return 0;
}
static int test_acquire_full_returns_nil(void) {
    cuda_slotbank *sb = make_bank(2);
    uint32_t a = sb_acquire(sb); fill_slot(sb, a, 0, 1);
    sb_lru_unlink(sb, a); sb->slots[a].pinned = 1; sb->n_pinned++;
    uint32_t b = sb_acquire(sb); fill_slot(sb, b, 0, 2);
    sb_lru_unlink(sb, b); sb->slots[b].pinned = 1; sb->n_pinned++;
    /* All slots pinned -> no evictable victim -> acquire signals failure,
       never hands back a slot the admission path would treat as usable. */
    if (sb_acquire(sb) != SLOT_NIL) { fprintf(stderr,"acquire must fail when full\n"); return 1; }
    fprintf(stderr,"test_acquire_full_returns_nil: PASS\n"); return 0;
}
static int test_pin_after_fill(void) {
    cuda_slotbank *sb = make_bank(3);
    uint32_t a = sb_acquire(sb); fill_slot(sb, a, 7, 99);
    sb_lru_unlink(sb, a); sb->slots[a].pinned = 1; sb->n_pinned++;  /* pin A */
    uint32_t b = sb_acquire(sb); fill_slot(sb, b, 7, 1);
    uint32_t c = sb_acquire(sb); fill_slot(sb, c, 7, 2);
    uint32_t v = sb_acquire(sb);  /* must be b or c, never pinned a */
    if (v == a) { fprintf(stderr,"pinned slot chosen as victim\n"); return 1; }
    if (sb_lookup(sb,7,99) == SLOT_NIL) { fprintf(stderr,"pinned expert lost\n"); return 1; }
    fprintf(stderr,"test_pin_after_fill: PASS\n"); return 0;
}
int main(void) {
    int rc = 0;
    rc |= test_lookup_hit_miss();
    rc |= test_lru_evicts_oldest();
    rc |= test_pinned_never_evicted();
    rc |= test_no_key_collision_across_layers();
    rc |= test_union_dedup_fits();
    rc |= test_union_out_slot_remap();
    rc |= test_acquire_full_returns_nil();
    rc |= test_pin_after_fill();
    if (rc) fprintf(stderr,"SLOTBANK TESTS FAILED\n");
    else    fprintf(stderr,"ALL SLOTBANK TESTS PASS\n");
    return rc;
}
