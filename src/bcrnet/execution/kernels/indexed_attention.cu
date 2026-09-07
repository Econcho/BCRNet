// Forward-only CUDA backend, compiled by NVRTC. No PyTorch/CUDA headers are required.
// One block computes one (window, head, query). KV is read directly from the bank.
#if BCR_HALF
typedef unsigned short scalar_t;
__device__ __forceinline__ float load_scalar(scalar_t x) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(x));
    return value;
}
__device__ __forceinline__ scalar_t store_scalar(float x) {
    scalar_t value;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(value) : "f"(x));
    return value;
}
#else
typedef float scalar_t;
__device__ __forceinline__ float load_scalar(scalar_t x) { return x; }
__device__ __forceinline__ scalar_t store_scalar(float x) { return x; }
#endif

extern "C" __global__ void indexed_attention(
    const scalar_t* query, const scalar_t* kv, const long long* inverse,
    const unsigned char* valid, const scalar_t* bias, scalar_t* output,
    int heads, int nq, int nm, int dim, int bank_rows, float scale
) {
    const int tid = threadIdx.x;
    const int qrow = blockIdx.x % nq;
    const int head = (blockIdx.x / nq) % heads;
    const int window = blockIdx.x / (nq * heads);
    const long long qbase = ((long long)window * heads * nq + head * nq + qrow) * dim;
    const float neg_inf = -__int_as_float(0x7f800000);
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = shared + nm;
    float* qcache = reduction + blockDim.x;
    for (int d = tid; d < dim; d += blockDim.x) qcache[d] = load_scalar(query[qbase + d]);
    __syncthreads();
    float local_max = neg_inf;
    for (int j = tid; j < nm; j += blockDim.x) {
        const long long id = inverse[(long long)window * nm + j];
        float score = neg_inf;
        if (valid[(long long)window * nm + j] && id >= 0 && id < bank_rows) {
            const long long kbase = (id * 2 * heads + head) * dim;
            float dot = 0;
            for (int d = 0; d < dim; ++d) dot += qcache[d] * load_scalar(kv[kbase + d]);
            score = dot * scale + load_scalar(bias[((long long)head * nq + qrow) * nm + j]);
        }
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }
    reduction[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    float maximum = reduction[0];
    if (maximum == neg_inf) maximum = 0; // Defensive: supported model always provides a valid dummy.
    float local_sum = 0;
    for (int j = tid; j < nm; j += blockDim.x) {
        const float probability = expf(scores[j] - maximum);
        scores[j] = probability;
        local_sum += probability;
    }
    reduction[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float denominator = fmaxf(reduction[0], 1e-30f);
    __syncthreads();
    // Parallelize weighted sums over both channels and keys.
    const int groups = blockDim.x / dim;
    const int group = tid / dim;
    const int d = tid % dim;
    float partial = 0;
    if (group < groups) {
        for (int j = group; j < nm; j += groups) {
            const long long id = inverse[(long long)window * nm + j];
            if (scores[j] != 0 && id >= 0 && id < bank_rows) {
                const long long vbase = (id * 2 * heads + heads + head) * dim;
                partial += scores[j] * load_scalar(kv[vbase + d]);
            }
        }
    }
    reduction[tid] = partial;
    __syncthreads();
    if (tid < dim) {
        float value = 0;
        for (int g = 0; g < groups; ++g) value += reduction[g * dim + tid];
        output[qbase + tid] = store_scalar(value / denominator);
    }
}
