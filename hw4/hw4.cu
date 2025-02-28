#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include <cuda.h>
#include <vector>
#include <iostream>
#include <fstream>

#define Br 32
#define Tile1 64*64
#define Tile2 64*64

__device__ float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

void pprint_float_array(const float* array, int size) {
    for (int i = 0; i < 20; ++i) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));
    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    fwrite(O, sizeof(float), B * N * d, file);
    free(Q);
    free(K);
    free(V);
    free(O);
    fclose(file);
}

__device__ void QKDotAndScalar(float *out, float *q, float *k, int tx, const int br, const int bc, float scalar, const int d) {
    if(tx < br) {
        for (int j = 0; j < bc; j++) {
            out[tx * bc + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[tx * bc + j] += q[tx * d + t] * k[j * d + t];
            }
            out[tx * bc + j] *= scalar;
        }
    }
}

__device__ void RowMax(float *out, float *in, int tx, const int br, const int bc) {
    if(tx < br) {
        float max_val = in[tx * bc];
        for (int j = 0; j < bc; j++) {
            max_val = _max(max_val, in[tx * bc + j]);
        }
        out[tx] = max_val;
    }
    
}

__device__ void MinusMaxAndExp(float *out, float *in, float *mx, int tx, const int br, const int bc) {
    if(tx < br) {
        for (int j = 0; j < bc; j++) {
            out[tx * bc + j] = __expf(in[tx * bc + j] - mx[tx]);
        }
    }
}

__device__ void RowSum(float *out, float *in, int tx, const int br, const int bc) {
    if(tx < br) {
        out[tx] = 0.0F;
        for (int j = 0; j < bc; j++) {
            out[tx] += in[tx * bc + j];
        }
    }
    
}

__device__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, 
                            const int br, const int bc, int tx, const int d) {
    __shared__ float mi_new[Br];
    __shared__ float li_new[Br];

    if(tx < br) {
        mi_new[tx] = _max(mi[tx], mij[tx]);
        li_new[tx] = __expf(mi[tx] - mi_new[tx]) * li[tx] + __expf(mij[tx] - mi_new[tx]) * lij[tx];
    }
    __syncthreads();

    if(tx < br) {
        for (int j = 0; j < d; j++) {
            float pv = 0.0;
            for (int t = 0; t < bc; t++) {
                pv += pij[tx * bc + t] * vj[t * d + j];
            }
            oi[tx * d + j] = (li[tx] * __expf(mi[tx] - mi_new[tx]) * oi[tx * d + j] + 
                             __expf(mij[tx] - mi_new[tx]) * pv) / li_new[tx];
        }
        
        mi[tx] = mi_new[tx];
        li[tx] = li_new[tx];
    }
    __syncthreads();
}

__global__ void flash_attention(float *q, float *k, float *v, float *o, const int d, const int N, 
                                         const float scale) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int batch = blockIdx.z;
    
    int global_row = bx * blockDim.x + tx;
    int batch_offset = batch * N * d;
    
    if (global_row >= N) return;

    const int TILE_SIZE = Br/2;
    __shared__ float tile_k[TILE_SIZE * 64];
    __shared__ float tile_v[TILE_SIZE * 64];
    __shared__ float qi[Br * 64];
    __shared__ float oi[Br * 64];
    __shared__ float li[Br];
    __shared__ float mi[Br];
    __shared__ float sij[Br * TILE_SIZE];
    __shared__ float pij[Br * TILE_SIZE];
    
    // 初始化
    if (tx < Br) {
        mi[tx] = FLT_MIN;
        li[tx] = 0.0f;
        for (int x = 0; x < d; x++) {
            qi[tx * d + x] = q[batch_offset + global_row * d + x];
            oi[tx * d + x] = 0.0f;  // 初始化輸出
        }
    }
    __syncthreads();

    // 分塊處理
    for (int j = 0; j < N; j += TILE_SIZE) {
        int remaining = min(TILE_SIZE, N - j);
        
        // 載入K和V
        if (tx < TILE_SIZE && (j + tx) < N) {
            for (int x = 0; x < d; x++) {
                tile_k[tx * d + x] = k[batch_offset + (j + tx) * d + x];
                tile_v[tx * d + x] = v[batch_offset + (j + tx) * d + x];
            }
        }
        __syncthreads();

        if (global_row < N) {
            // 計算S_ij (QK^T)
            float max_val = FLT_MIN;
            for (int t = 0; t < remaining; t++) {
                float dot = 0.0f;
                for (int x = 0; x < d; x++) {
                    dot += qi[tx * d + x] * tile_k[t * d + x];
                }
                sij[tx * TILE_SIZE + t] = dot * scale;
                max_val = _max(max_val, sij[tx * TILE_SIZE + t]);
            }
            
            // 計算當前tile的最大值
            float mi_new = _max(mi[tx], max_val);
            float li_new = li[tx] * __expf(mi[tx] - mi_new);
            
            // 計算softmax分母並更新注意力權重
            for (int t = 0; t < remaining; t++) {
                pij[tx * TILE_SIZE + t] = __expf(sij[tx * TILE_SIZE + t] - mi_new);
                li_new += pij[tx * TILE_SIZE + t];
            }
            
            // 更新輸出
            float scale_old = __expf(mi[tx] - mi_new) * li[tx] / li_new;
            float scale_new = 1.0f / li_new;
            
            for (int x = 0; x < d; x++) {
                float new_val = 0.0f;
                for (int t = 0; t < remaining; t++) {
                    new_val += pij[tx * TILE_SIZE + t] * tile_v[t * d + x];
                }
                oi[tx * d + x] = oi[tx * d + x] * scale_old + new_val * scale_new;
            }
            
            mi[tx] = mi_new;
            li[tx] = li_new;
        }
        __syncthreads();
    }

    // 寫回結果
    if (tx < Br && global_row < N) {
        for (int x = 0; x < d; x++) {
            o[batch_offset + global_row * d + x] = oi[tx * d + x];
        }
        // int lm_idx = batch * N + global_row;
        // l_batch[lm_idx] = li[tx];
        // m_batch[lm_idx] = mi[tx];
    }
}
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    
    input(argv[1]);

    double start, end;
    start = getTimeStamp();
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc((void**)&d_Q, B * N * d * sizeof(float));
    cudaMalloc((void**)&d_K, B * N * d * sizeof(float));
    cudaMalloc((void**)&d_V, B * N * d * sizeof(float));
    cudaMalloc((void**)&d_O, B * N * d * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_Q, Q, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, B * N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, B * N * d * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and initialize l and m arrays
    // float *host_l = new float[B*N];
    // float *host_m = new float[B*N];
    // std::fill_n(host_l, B*N, 0.0f);
    // std::fill_n(host_m, B*N, FLT_MIN);
    
    // float *d_l, *d_m;
    // cudaMalloc((void**)&d_l, B*N * sizeof(float));
    // cudaMalloc((void**)&d_m, B*N * sizeof(float));
    // cudaMemcpy(d_l, host_l, B*N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_m, host_m, B*N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    dim3 grid_dim((N + Br - 1)/Br, 1, B);  // Blocks for N dimension and batches
    dim3 block_dim(Br);  // Threads per block
    
    const float softmax_scale = 1.0 / sqrt(d);
    
    flash_attention<<<grid_dim, block_dim>>>(d_Q, d_K, d_V, d_O, d, N, softmax_scale);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(O, d_O, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);
    
    pprint_float_array(O, 10);
    output(argv[2]);

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    // cudaFree(d_l);
    // cudaFree(d_m);
    // delete[] host_l;
    // delete[] host_m;

    return 0;
}