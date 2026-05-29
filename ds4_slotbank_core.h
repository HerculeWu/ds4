#ifndef DS4_SLOTBANK_CORE_H
#define DS4_SLOTBANK_CORE_H
/* Pure-C core of the Phase 2 routed-expert residency cache. No CUDA symbols
   here: this header is shared verbatim by ds4_cuda.cu (the real cache) and
   tests/cuda_slotbank_test.c (host-only LRU/hash unit tests). All device work
   (slab cudaMalloc, staged fills, stream sync) lives in ds4_cuda.cu. */
#include <stdint.h>
#include <string.h>

#define SLOT_NIL 0xFFFFFFFFu
#define SB_FREE_LAYER 0xFFFFFFFFu   /* slot.layer sentinel: empty/free */

typedef struct {
    uint32_t layer;       /* il; SB_FREE_LAYER == empty */
    uint32_t expert_id;   /* 0..n_total_expert-1 */
    char    *gate_dev;    /* device ptr inside the slab; valid iff resident */
    char    *up_dev;
    char    *down_dev;
    uint32_t lru_prev;    /* intrusive LRU list, slot indices, SLOT_NIL terminates */
    uint32_t lru_next;
    uint32_t hnext;       /* hash-bucket chain (separate chaining) */
    uint8_t  pinned;      /* 1 == non-evictable */
    uint8_t  resident;    /* 1 == device-valid */
} cuda_expert_slot;

typedef struct {
    cuda_expert_slot *slots;
    uint32_t  n_slots;
    uint32_t  n_pinned;
    uint32_t  lru_head;   /* MRU */
    uint32_t  lru_tail;   /* LRU victim end */
    uint32_t *htab;       /* bucket -> slot index head, SLOT_NIL empty */
    uint32_t  hmask;      /* (power-of-two bucket count) - 1 */
    uint64_t  hits, misses, evictions;
} cuda_slotbank;

static inline uint64_t sb_key(uint32_t layer, uint32_t expert_id) {
    return ((uint64_t)layer << 16) | (uint64_t)expert_id;
}
static inline uint32_t sb_hash(uint64_t k, uint32_t mask) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return (uint32_t)k & mask;
}

/* Unlink slot s from the LRU list (no-op if already unlinked). */
static inline void sb_lru_unlink(cuda_slotbank *sb, uint32_t s) {
    cuda_expert_slot *p = &sb->slots[s];
    if (p->lru_prev != SLOT_NIL) sb->slots[p->lru_prev].lru_next = p->lru_next;
    else if (sb->lru_head == s)  sb->lru_head = p->lru_next;
    if (p->lru_next != SLOT_NIL) sb->slots[p->lru_next].lru_prev = p->lru_prev;
    else if (sb->lru_tail == s)  sb->lru_tail = p->lru_prev;
    p->lru_prev = p->lru_next = SLOT_NIL;
}
/* Insert slot s at the MRU head. */
static inline void sb_lru_push_head(cuda_slotbank *sb, uint32_t s) {
    cuda_expert_slot *p = &sb->slots[s];
    p->lru_prev = SLOT_NIL;
    p->lru_next = sb->lru_head;
    if (sb->lru_head != SLOT_NIL) sb->slots[sb->lru_head].lru_prev = s;
    sb->lru_head = s;
    if (sb->lru_tail == SLOT_NIL) sb->lru_tail = s;
}
static inline void sb_touch(cuda_slotbank *sb, uint32_t s) {
    if (sb->slots[s].pinned) return;   /* pinned slots stay out of the LRU list */
    sb_lru_unlink(sb, s);
    sb_lru_push_head(sb, s);
}

/* O(1) keyed lookup. Returns slot index or SLOT_NIL. */
static inline uint32_t sb_lookup(cuda_slotbank *sb, uint32_t layer, uint32_t expert_id) {
    uint64_t k = sb_key(layer, expert_id);
    uint32_t b = sb_hash(k, sb->hmask);
    uint32_t s = sb->htab[b];
    while (s != SLOT_NIL) {
        if (sb->slots[s].layer == layer && sb->slots[s].expert_id == expert_id) return s;
        s = sb->slots[s].hnext;
    }
    return SLOT_NIL;
}
static inline void sb_hash_insert(cuda_slotbank *sb, uint32_t s) {
    uint64_t k = sb_key(sb->slots[s].layer, sb->slots[s].expert_id);
    uint32_t b = sb_hash(k, sb->hmask);
    sb->slots[s].hnext = sb->htab[b];
    sb->htab[b] = s;
}
static inline void sb_hash_remove(cuda_slotbank *sb, uint32_t s) {
    uint64_t k = sb_key(sb->slots[s].layer, sb->slots[s].expert_id);
    uint32_t b = sb_hash(k, sb->hmask);
    uint32_t cur = sb->htab[b], prev = SLOT_NIL;
    while (cur != SLOT_NIL) {
        if (cur == s) {
            if (prev == SLOT_NIL) sb->htab[b] = sb->slots[cur].hnext;
            else sb->slots[prev].hnext = sb->slots[cur].hnext;
            return;
        }
        prev = cur; cur = sb->slots[cur].hnext;
    }
}
/* Pick an evictable victim from the LRU tail (skips pinned). Returns slot
   index or SLOT_NIL if none free-able. Caller must then evict + refill. */
static inline uint32_t sb_acquire(cuda_slotbank *sb) {
    uint32_t s = sb->lru_tail;
    while (s != SLOT_NIL) {
        if (!sb->slots[s].pinned) return s;
        s = sb->slots[s].lru_prev;
    }
    return SLOT_NIL;
}
/* Logically evict a slot: drop from hash, mark free. Device memory is reused
   in place by the next fill; no cudaFree. */
static inline void sb_evict(cuda_slotbank *sb, uint32_t s) {
    if (sb->slots[s].layer != SB_FREE_LAYER) {
        sb_hash_remove(sb, s);
        sb->evictions++;
    }
    sb->slots[s].layer = SB_FREE_LAYER;
    sb->slots[s].resident = 0;
}
#endif /* DS4_SLOTBANK_CORE_H */
