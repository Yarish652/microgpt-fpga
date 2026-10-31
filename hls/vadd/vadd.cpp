// Stage 1 hello-world: c[i] = a[i] + b[i] over DDR via AXI master, controlled over AXI4-Lite.
// Exercises the same interface style the microgpt core will use (m_axi data + s_axilite control).
#include <stdint.h>

extern "C" void vadd(const int32_t *a, const int32_t *b, int32_t *c, int n) {
#pragma HLS INTERFACE m_axi port = a offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE m_axi port = b offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE m_axi port = c offset = slave bundle = gmem depth = 1024
#pragma HLS INTERFACE s_axilite port = a bundle = control
#pragma HLS INTERFACE s_axilite port = b bundle = control
#pragma HLS INTERFACE s_axilite port = c bundle = control
#pragma HLS INTERFACE s_axilite port = n bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE
        c[i] = a[i] + b[i];  // int32 wraps on overflow; the test data never overflows
    }
}
