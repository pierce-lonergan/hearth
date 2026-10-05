/* Contributor-style tests for hxdemo.c; mutate.py's C demo target. Exit 0 = pass. */
#include <stdio.h>
#include <string.h>
#include "hxdemo.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

/* Independent statement of NUMERICS §1: lanes by stride, explicit tree levels. */
static float ref_sum16(const float *a, int n) {
    float L[16] = {0};
    for (int lane = 0; lane < 16; lane++)
        for (int i = lane; i < n; i += 16)
            L[lane] = L[lane] + a[i];
    for (int j = 0; j < 8; j++) L[j] = L[j] + L[j + 8];
    for (int j = 0; j < 4; j++) L[j] = L[j] + L[j + 4];
    for (int j = 0; j < 2; j++) L[j] = L[j] + L[j + 2];
    L[0] = L[0] + L[1];
    return L[0];
}

static void test_align(void) {
    CHECK(hxd_align_up(0, 4096) == 0);
    CHECK(hxd_align_up(1, 4096) == 4096);
    CHECK(hxd_align_up(4095, 4096) == 4096);
    CHECK(hxd_align_up(4096, 4096) == 4096);
    CHECK(hxd_align_up(4097, 4096) == 8192);
    CHECK(hxd_align_up(65, 64) == 128);
    CHECK(hxd_align_up(7, 1) == 7);
}

static void test_sum16(void) {
    float a[70];
    for (int i = 0; i < 70; i++)
        a[i] = (i % 3 == 0) ? 1.0e7f + (float)i : 0.1f * (float)(i + 1);
    CHECK(hxd_sum16(a, 0) == 0.0f);
    CHECK(hxd_sum16(a, 1) == a[0]);
    for (int n = 2; n <= 70; n++) {
        float got = hxd_sum16(a, n), want = ref_sum16(a, n);
        if (memcmp(&got, &want, sizeof got) != 0) {
            printf("FAIL sum16 n=%d: %.9g != %.9g\n", n, got, want);
            failures++;
            break;
        }
    }
    float ones[16];
    for (int i = 0; i < 16; i++) ones[i] = (float)(i + 1);
    CHECK(hxd_sum16(ones, 16) == 136.0f);
}

static void test_topk(void) {
    const float s[5] = {0.1f, 0.9f, 0.5f, 0.9f, 0.2f};
    int ids[5] = {-7, -7, -7, -7, -7};
    CHECK(hxd_topk(s, 5, 2, ids) == 2);
    CHECK(ids[0] == 1 && ids[1] == 3);          /* tie -> lower index first */
    CHECK(hxd_topk(s, 5, 3, ids) == 3);
    CHECK(ids[0] == 1 && ids[1] == 3 && ids[2] == 2);
    CHECK(hxd_topk(s, 5, 9, ids) == 5);
    CHECK(ids[3] == 4 && ids[4] == 0);
    CHECK(hxd_topk(s, 5, 0, ids) == 0);
    const float first[3] = {0.9f, 0.1f, 0.2f};
    CHECK(hxd_topk(first, 3, 1, ids) == 1 && ids[0] == 0);
    CHECK(hxd_topk(first, 3, 2, ids) == 2 && ids[0] == 0 && ids[1] == 2);  /* id 0 taken, rest smaller */
    const float neg[4] = {-3.0f, -1.0f, -2.0f, -1.0f};
    CHECK(hxd_topk(neg, 4, 2, ids) == 2 && ids[0] == 1 && ids[1] == 3);
}

static void test_lfu(void) {
    hxd_lfu c;
    memset(&c, 0, sizeof c);   /* deterministic contents beyond `used` */
    hxd_lfu_init(&c, 0);
    CHECK(c.cap == 1 && c.used == 0);
    hxd_lfu_init(&c, 100);
    CHECK(c.cap == HXD_LFU_MAX);
    hxd_lfu_init(&c, HXD_LFU_MAX);
    CHECK(c.cap == HXD_LFU_MAX);

    hxd_lfu_init(&c, 2);
    CHECK(hxd_lfu_access(&c, 1) == 0);
    CHECK(hxd_lfu_access(&c, 1) == 1);
    CHECK(hxd_lfu_access(&c, 2) == 0);
    CHECK(c.used == 2);
    CHECK(hxd_lfu_access(&c, 3) == 0);           /* evicts 2 (count 1 < count 2) */
    CHECK(hxd_lfu_has(&c, 1) && hxd_lfu_has(&c, 3) && !hxd_lfu_has(&c, 2));
    CHECK(hxd_lfu_access(&c, 2) == 0);           /* evicts 3 */
    CHECK(hxd_lfu_has(&c, 1) && hxd_lfu_has(&c, 2) && !hxd_lfu_has(&c, 3));
    CHECK(hxd_lfu_access(&c, 1) == 1);
    CHECK(c.hits == 2 && c.misses == 4);
    CHECK(c.clock == 6);

    hxd_lfu_init(&c, 3);                          /* re-init: stale keys must not be visible */
    CHECK(!hxd_lfu_has(&c, 1));
    CHECK(hxd_lfu_access(&c, 1) == 0 && c.misses == 1 && c.hits == 0);

    hxd_lfu_init(&c, 3);                          /* equal counts -> least recent goes */
    hxd_lfu_access(&c, 5);
    hxd_lfu_access(&c, 6);
    hxd_lfu_access(&c, 7);
    hxd_lfu_access(&c, 5);
    hxd_lfu_access(&c, 6);
    hxd_lfu_access(&c, 7);                        /* all count 2; 5 is least recent */
    CHECK(hxd_lfu_access(&c, 8) == 0);
    CHECK(!hxd_lfu_has(&c, 5) && hxd_lfu_has(&c, 6) && hxd_lfu_has(&c, 7) && hxd_lfu_has(&c, 8));
    CHECK(hxd_lfu_access(&c, 9) == 0);            /* 8 has count 1 -> evicted before 6/7 */
    CHECK(!hxd_lfu_has(&c, 8) && hxd_lfu_has(&c, 9) && hxd_lfu_has(&c, 6) && hxd_lfu_has(&c, 7));
    CHECK(hxd_lfu_access(&c, 6) == 1 && c.count[0] + c.count[1] + c.count[2] == 6);
}

int main(void) {
    test_align();
    test_sum16();
    test_topk();
    test_lfu();
    if (failures)
        printf("test_hxdemo: %d failure(s)\n", failures);
    else
        printf("test_hxdemo: all passed\n");
    return failures ? 1 : 0;
}
