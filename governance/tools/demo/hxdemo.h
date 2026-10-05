/*
 * hxdemo.h — a small self-contained C module for demonstrating and testing
 * governance/tools/mutate.py. Not part of the engine; it mirrors a few engine
 * idioms (slab alignment, NUMERICS sum16, top-k tie rule, LFU eviction) so the
 * mutation demo exercises realistic code.
 */
#ifndef HXDEMO_H
#define HXDEMO_H

#include <stdint.h>

#define HXD_LFU_MAX 16

/* Round x up to a multiple of a (a must be a power of two). */
uint64_t hxd_align_up(uint64_t x, uint64_t a);

/* docs/NUMERICS.md §1 sum16: 16 lane accumulators, then a fixed pairwise tree. */
float hxd_sum16(const float *a, int n);

/* Indices of the k largest scores, descending; ties go to the lower index.
 * Writes min(k, n) ids and returns that count. */
int hxd_topk(const float *score, int n, int k, int *ids);

/* Tiny LFU cache of int keys: evicts the lowest count, ties -> least recently used. */
typedef struct hxd_lfu {
    int cap, used;
    int key[HXD_LFU_MAX];
    uint32_t count[HXD_LFU_MAX];
    uint32_t stamp[HXD_LFU_MAX];
    uint32_t clock;
    uint64_t hits, misses;
} hxd_lfu;

void hxd_lfu_init(hxd_lfu *c, int cap);      /* cap clamped to [1, HXD_LFU_MAX] */
int  hxd_lfu_access(hxd_lfu *c, int key);    /* 1 = hit, 0 = miss (key inserted) */
int  hxd_lfu_has(const hxd_lfu *c, int key);

#endif /* HXDEMO_H */
