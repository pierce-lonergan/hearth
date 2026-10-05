#include "hxdemo.h"

uint64_t hxd_align_up(uint64_t x, uint64_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

float hxd_sum16(const float *a, int n) {
    float L[16];
    int i, j, s;
    for (i = 0; i < 16; i++)
        L[i] = 0.0f;
    for (i = 0; i < n; i++)
        L[i & 15] = L[i & 15] + a[i];
    for (s = 8; s >= 1; s /= 2)
        for (j = 0; j < s; j++)
            L[j] = L[j] + L[j + s];
    return L[0];
}

int hxd_topk(const float *score, int n, int k, int *ids) {
    int got = 0;
    if (k > n)
        k = n;
    while (got < k) {
        int best = -1;
        for (int e = 0; e < n; e++) {
            int taken = 0;
            for (int j = 0; j < got; j++) {
                if (ids[j] == e) {
                    taken = 1;
                    break;
                }
            }
            if (taken)
                continue;
            if (best < 0 || score[e] > score[best])
                best = e;
        }
        ids[got++] = best;
    }
    return got;
}

void hxd_lfu_init(hxd_lfu *c, int cap) {
    if (cap > HXD_LFU_MAX)
        cap = HXD_LFU_MAX;
    if (cap < 1)
        cap = 1;
    c->cap = cap;
    c->used = 0;
    c->clock = 0;
    c->hits = 0;
    c->misses = 0;
}

int hxd_lfu_has(const hxd_lfu *c, int key) {
    for (int i = 0; i < c->used; i++)
        if (c->key[i] == key)
            return 1;
    return 0;
}

int hxd_lfu_access(hxd_lfu *c, int key) {
    int i, victim;
    c->clock++;
    for (i = 0; i < c->used; i++) {
        if (c->key[i] == key) {
            c->count[i]++;
            c->stamp[i] = c->clock;
            c->hits++;
            return 1;
        }
    }
    c->misses++;
    if (c->used < c->cap) {
        victim = c->used++;
    } else {
        victim = 0;
        for (i = 1; i < c->cap; i++) {
            if (c->count[i] < c->count[victim] ||
                (c->count[i] == c->count[victim] && c->stamp[i] < c->stamp[victim]))
                victim = i;
        }
    }
    c->key[victim] = key;
    c->count[victim] = 1;
    c->stamp[victim] = c->clock;
    return 0;
}
