#include <metal_stdlib>
using namespace metal;

// Inputs12..27 are object embeddings;28..32 are visible map flags.
kernel void sparse_first(device const float* delta, device const long* neighbors,
    device const long* groups, device const long* cells, device const float* anchors,
    device const float* weight, device float* out,
    constant uint& size, constant uint& count, uint index [[thread_position_in_grid]]) {
    if(index >= count * 16) return;
    uint row = index / 16, channel = index % 16;
    float sum = 0.0f;
    for(uint input = 0; input < 21; ++input) {
        for(uint k = 0; k < 9; ++k) {
            long neighbor = neighbors[row * 9 + k];
            sum += delta[neighbor * 21 + input] * weight[(channel * 33 + input + 12) * 9 + k];
        }
    }
    float base = anchors[(groups[row] * size + cells[row]) * 16 + channel];
    out[(row + 1) * 16 + channel] = tanh(base + sum) - tanh(base);
}

kernel void sparse_second(device const float* delta, device const long* neighbors,
    device const long* groups, device const long* cells, device const float* anchors,
    device const float* weight, device const long* bins, device atomic_float* pooled,
    constant uint& size, constant uint& count, constant float& area,
    uint index [[thread_position_in_grid]]) {
    if(index >= count * 16) return;
    uint row = index / 16, channel = index % 16;
    float sum = 0.0f;
    for(uint input = 0; input < 16; ++input) {
        for(uint k = 0; k < 9; ++k) {
            long neighbor = neighbors[row * 9 + k];
            sum += delta[neighbor * 16 + input] * weight[(channel * 16 + input) * 9 + k];
        }
    }
    float base = anchors[(groups[row] * size + cells[row]) * 16 + channel];
    float difference = (tanh(base + sum) - tanh(base)) / area;
    atomic_fetch_add_explicit(&pooled[bins[row] * 16 + channel], difference, memory_order_relaxed);
}

// Exact compact-index variants; address products remain64-bit.
// Inputs12..27 are object embeddings;28..32 are visible map flags.
kernel void sparse_first_i32(device const float* delta, device const int* neighbors,
    device const int* groups, device const int* cells, device const float* anchors,
    device const float* weight, device float* out,
    constant uint& size, constant uint& count, uint index [[thread_position_in_grid]]) {
    if(index >= count * 16) return;
    uint row = index / 16, channel = index % 16;
    float sum = 0.0f;
    for(uint input = 0; input < 21; ++input) {
        for(uint k = 0; k < 9; ++k) {
            long neighbor = neighbors[row * 9 + k];
            sum += delta[neighbor * 21 + input] * weight[(channel * 33 + input + 12) * 9 + k];
        }
    }
    float base = anchors[(long(groups[row]) * size + cells[row]) * 16 + channel];
    out[(row + 1) * 16 + channel] = tanh(base + sum) - tanh(base);
}

kernel void sparse_second_i32(device const float* delta, device const int* neighbors,
    device const int* groups, device const int* cells, device const float* anchors,
    device const float* weight, device const int* bins, device atomic_float* pooled,
    constant uint& size, constant uint& count, constant float& area,
    uint index [[thread_position_in_grid]]) {
    if(index >= count * 16) return;
    uint row = index / 16, channel = index % 16;
    float sum = 0.0f;
    for(uint input = 0; input < 16; ++input) {
        for(uint k = 0; k < 9; ++k) {
            long neighbor = neighbors[row * 9 + k];
            sum += delta[neighbor * 16 + input] * weight[(channel * 16 + input) * 9 + k];
        }
    }
    float base = anchors[(long(groups[row]) * size + cells[row]) * 16 + channel];
    float difference = (tanh(base + sum) - tanh(base)) / area;
    atomic_fetch_add_explicit(&pooled[long(bins[row]) * 16 + channel], difference, memory_order_relaxed);
}
