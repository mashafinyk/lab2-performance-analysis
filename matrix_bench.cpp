#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>

constexpr size_t N = 16777216; // 16M елементів (~64 МБ на один вектор float)
constexpr int DEFAULT_ITERS = 30;

// Функція 1: добре векторизується компілятором у SIMD/AVX2 інструкції
__attribute__((noinline))
float saxpy_accumulate(const float* __restrict x, const float* __restrict y, float alpha, size_t n) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    for (size_t i = 0; i < n; i += 4) {
        sum0 += alpha * x[i] + y[i];
        sum1 += alpha * x[i + 1] + y[i + 1];
        sum2 += alpha * x[i + 2] + y[i + 2];
        sum3 += alpha * x[i + 3] + y[i + 3];
    }
    return (sum0 + sum1) + (sum2 + sum3);
}

// Функція 2: цілочисельне перемішування (навантаження на ALU та кеш)
__attribute__((noinline))
uint64_t hash_reduce(const uint32_t* data, size_t n) {
    uint64_t acc = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = data[i];
        v ^= v >> 16;
        v *= 0x7feb352dU;
        v ^= v >> 15;
        acc += v ^ (i & 31U);
    }
    return acc;
}

__attribute__((noinline))
double run_pipeline(const std::vector<float>& x, const std::vector<float>& y,
    const std::vector<uint32_t>& ids, int iters) {
    double total = 0.0;
    for (int step = 0; step < iters; ++step) {
        float s = saxpy_accumulate(x.data(), y.data(), 1.25f, x.size());
        uint64_t h = hash_reduce(ids.data(), ids.size());
        total += static_cast<double>(s) + static_cast<double>(h & 0xFFFFU);
    }
    return total;
}

int main(int argc, char* argv[]) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : DEFAULT_ITERS;

    std::vector<float> x(N);
    std::vector<float> y(N);
    std::vector<uint32_t> ids(N);

    for (size_t i = 0; i < N; ++i) {
        x[i] = static_cast<float>((i % 103) - 51) * 0.01f;
        y[i] = static_cast<float>((i % 89) - 44) * 0.02f;
        ids[i] = static_cast<uint32_t>(i * 2654435761U + 17U);
    }

    double result = run_pipeline(x, y, ids, iters);
    std::cout << "Result: " << result << "\n";
    return 0;
}