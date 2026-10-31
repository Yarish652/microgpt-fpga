// C-simulation testbench for vadd. Returns 0 on pass (required by csim/cosim).
#include <stdint.h>
#include <stdio.h>
#include <vector>

extern "C" void vadd(const int32_t *a, const int32_t *b, int32_t *c, int n);

static int run_case(int n) {
    std::vector<int32_t> a(n + 1), b(n + 1), c(n + 1, -1);  // +1 guard element: must stay untouched
    for (int i = 0; i < n; i++) {
        a[i] = 3 * i - 100;
        b[i] = 1000 - 7 * i;
    }
    vadd(a.data(), b.data(), c.data(), n);
    int bad = 0;
    for (int i = 0; i < n; i++)
        if (c[i] != a[i] + b[i]) bad++;
    if (c[n] != -1) bad++;
    printf("n=%4d  %s\n", n, bad ? "FAIL" : "ok");
    return bad;
}

int main() {
    int bad = 0;
    for (int n : {0, 1, 7, 64, 1000, 1024}) bad += run_case(n);
    printf(bad ? "TEST FAILED\n" : "TEST PASSED\n");
    return bad ? 1 : 0;
}
