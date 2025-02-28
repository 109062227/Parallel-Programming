#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include <time.h>

//======================
#define DEV_NO 0
cudaDeviceProp prop;
int n, m, ori_n;
const int B = 64;//block_size
const int HALF_B = 32;
int *Dist = NULL;
const int INF = ((1 << 30) - 1);




void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    ori_n = n;
    if(n%B != 0) n += (B - (n%B));

    Dist = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i*n+j] = 0;
            } else {
                Dist[i*n+j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*n + pair[1]] = pair[2];
    }
    printf("n: %d, m: %d\n", n, m);
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < ori_n; ++i) {
        for (int j = 0; j < ori_n; ++j) {
            if (Dist[i*n+j] >= INF) Dist[i*n+j] = INF;
        }
        fwrite(Dist + i*n, sizeof(int), ori_n, outfile);
    }
    
    fclose(outfile);
}

//int ceil(int a, int b) { return (a + b - 1) / b; }


__global__ void Block_FW_phase1(int n, int r, int *global_D)
{
    __shared__ int shared_D[B][B];

    const int x = threadIdx.x;    
    const int y = threadIdx.y;    
    const int block_offset = B * r; 

   
    const int global_row1 = block_offset + y;
    const int global_row2 = block_offset + y + HALF_B;
    const int global_col1 = block_offset + x;
    const int global_col2 = block_offset + x + HALF_B;

    
    shared_D[y][x] = global_D[global_row1 * n + global_col1];
    shared_D[y + HALF_B][x] = global_D[global_row2 * n + global_col1];
    shared_D[y][x + HALF_B] = global_D[global_row1 * n + global_col2];
    shared_D[y + HALF_B][x + HALF_B] = global_D[global_row2 * n + global_col2];
    __syncthreads();

    
    for (int k = 0; k < B; k++) {
        int candidate1 = shared_D[y][k] + shared_D[k][x];
        int candidate2 = shared_D[y + HALF_B][k] + shared_D[k][x];
        int candidate3 = shared_D[y][k] + shared_D[k][x + HALF_B];
        int candidate4 = shared_D[y + HALF_B][k] + shared_D[k][x + HALF_B];

        shared_D[y][x] = min(shared_D[y][x], candidate1);
        shared_D[y + HALF_B][x] = min(shared_D[y + HALF_B][x], candidate2);
        shared_D[y][x + HALF_B] = min(shared_D[y][x + HALF_B], candidate3);
        shared_D[y + HALF_B][x + HALF_B] = min(shared_D[y + HALF_B][x + HALF_B], candidate4);
        __syncthreads();
    }

    
    global_D[global_row1 * n + global_col1] = shared_D[y][x];
    global_D[global_row2 * n + global_col1] = shared_D[y + HALF_B][x];
    global_D[global_row1 * n + global_col2] = shared_D[y][x + HALF_B];
    global_D[global_row2 * n + global_col2] = shared_D[y + HALF_B][x + HALF_B];

}
__global__ void Block_FW_phase2(int n, int r, int *global_D)
{
    
    const int i = blockIdx.x;
    if (i == r) return; // skip if i == r

    __shared__ int shared_D[B][B];
    __shared__ int row[B][B];
    __shared__ int col[B][B];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int block_offset = B * r;

    // 計算列和行的全局索引
    const int col_offset = i * B * n + r * B;
    const int row_offset = r * B * n + i * B;

    const int diagonal_offset = block_offset * (n + 1);

    // col[][] for col fixed,
    col[y][x] = global_D[col_offset + y * n + x];
    col[y + HALF_B][x] = global_D[col_offset + (y + HALF_B) * n + x];
    col[y][x + HALF_B] = global_D[col_offset + y * n + (x + HALF_B)];
    col[y + HALF_B][x + HALF_B] = global_D[col_offset + (y + HALF_B) * n + (x + HALF_B)];

    row[y][x] = global_D[row_offset + y * n + x];
    row[y + HALF_B][x] = global_D[row_offset + (y + HALF_B) * n + x];
    row[y][x + HALF_B] = global_D[row_offset + y * n + (x + HALF_B)];
    row[y + HALF_B][x + HALF_B] = global_D[row_offset + (y + HALF_B) * n + (x + HALF_B)];

    
    shared_D[y][x] = global_D[diagonal_offset + y * n + x];
    shared_D[y + HALF_B][x] = global_D[diagonal_offset + (y + HALF_B) * n + x];
    shared_D[y][x + HALF_B] = global_D[diagonal_offset + y * n + (x + HALF_B)];
    shared_D[y + HALF_B][x + HALF_B] = global_D[diagonal_offset + (y + HALF_B) * n + (x + HALF_B)];

    __syncthreads();

    
    #pragma unroll 32
    for (int k = 0; k < B; k++) {
        int candidate_col1 = col[y][k] + shared_D[k][x];
        int candidate_col2 = col[y + HALF_B][k] + shared_D[k][x];
        int candidate_col3 = col[y][k] + shared_D[k][x + HALF_B];
        int candidate_col4 = col[y + HALF_B][k] + shared_D[k][x + HALF_B];

        col[y][x] = min(col[y][x], candidate_col1);
        col[y + HALF_B][x] = min(col[y + HALF_B][x], candidate_col2);
        col[y][x + HALF_B] = min(col[y][x + HALF_B], candidate_col3);
        col[y + HALF_B][x + HALF_B] = min(col[y + HALF_B][x + HALF_B], candidate_col4);

        int candidate_row1 = shared_D[y][k] + row[k][x];
        int candidate_row2 = shared_D[y + HALF_B][k] + row[k][x];
        int candidate_row3 = shared_D[y][k] + row[k][x + HALF_B];
        int candidate_row4 = shared_D[y + HALF_B][k] + row[k][x + HALF_B];

        row[y][x] = min(row[y][x], candidate_row1);
        row[y + HALF_B][x] = min(row[y + HALF_B][x], candidate_row2);
        row[y][x + HALF_B] = min(row[y][x + HALF_B], candidate_row3);
        row[y + HALF_B][x + HALF_B] = min(row[y + HALF_B][x + HALF_B], candidate_row4);

        
    }

    
    global_D[col_offset + y * n + x] = col[y][x];
    global_D[col_offset + (y + HALF_B) * n + x] = col[y + HALF_B][x];
    global_D[col_offset + y * n + (x + HALF_B)] = col[y][x + HALF_B];
    global_D[col_offset + (y + HALF_B) * n + (x + HALF_B)] = col[y + HALF_B][x + HALF_B];

    global_D[row_offset + y * n + x] = row[y][x];
    global_D[row_offset + (y + HALF_B) * n + x] = row[y + HALF_B][x];
    global_D[row_offset + y * n + (x + HALF_B)] = row[y][x + HALF_B];
    global_D[row_offset + (y + HALF_B) * n + (x + HALF_B)] = row[y + HALF_B][x + HALF_B];


}
__global__ void Block_FW_phase3(int n, int r, int *global_D, int yOffset)
{

    const int j = blockIdx.x;
    const int i = blockIdx.y + yOffset;
    if(i == r && j == r) return;

    __shared__ int shared_D[B][B];
    __shared__ int row[B][B];
    __shared__ int col[B][B];

    const int x = threadIdx.x;
    const int y = threadIdx.y;

    // Simplify the indexing
    const int base_i = i * B * n + r * B;
    const int base_r = r * B * n + j * B;

    // Load columns (from D[i, r] into col)
    col[y][x] = global_D[base_i + y * n + x];
    col[y + HALF_B][x] = global_D[base_i + (y + HALF_B) * n + x];
    col[y][x + HALF_B] = global_D[base_i + y * n + (x + HALF_B)];
    col[y + HALF_B][x + HALF_B] = global_D[base_i + (y + HALF_B) * n + (x + HALF_B)];

    // Load rows (from D[r, j] into row)
    row[y][x] = global_D[base_r + y * n + x];
    row[y + HALF_B][x] = global_D[base_r + (y + HALF_B) * n + x];
    row[y][x + HALF_B] = global_D[base_r + y * n + (x + HALF_B)];
    row[y + HALF_B][x + HALF_B] = global_D[base_r + (y + HALF_B) * n + (x + HALF_B)];

    // Load shared D values (for shared distance matrix)
    const int base_shared = i * B * n + j * B;  // This is the correct base offset for shared D
    shared_D[y][x] = global_D[base_shared + y * n + x];
    shared_D[y + HALF_B][x] = global_D[base_shared + (y + HALF_B) * n + x];
    shared_D[y][x + HALF_B] = global_D[base_shared + y * n + (x + HALF_B)];
    shared_D[y + HALF_B][x + HALF_B] = global_D[base_shared + (y + HALF_B) * n + (x + HALF_B)];

    // Synchronize threads to ensure that shared memory is fully loaded
    __syncthreads();

    // Perform the relaxation step (Floyd-Warshall update) using shared memory
    #pragma unroll 32
    for(int k = 0; k < B; k++)
    {
        int candidate1 = col[y][k] + row[k][x];
        int candidate2 = col[y + HALF_B][k] + row[k][x];
        int candidate3 = col[y][k] + row[k][x + HALF_B];
        int candidate4 = col[y + HALF_B][k] + row[k][x + HALF_B];

        shared_D[y][x] = min(shared_D[y][x], candidate1);
        shared_D[y + HALF_B][x] = min(shared_D[y + HALF_B][x], candidate2);
        shared_D[y][x + HALF_B] = min(shared_D[y][x + HALF_B], candidate3);
        shared_D[y + HALF_B][x + HALF_B] = min(shared_D[y + HALF_B][x + HALF_B], candidate4);
    }

    // Store the results back into the global memory (distance matrix)
    global_D[base_shared + y * n + x] = shared_D[y][x];
    global_D[base_shared + (y + HALF_B) * n + x] = shared_D[y + HALF_B][x];
    global_D[base_shared + y * n + (x + HALF_B)] = shared_D[y][x + HALF_B];
    global_D[base_shared + (y + HALF_B) * n + (x + HALF_B)] = shared_D[y + HALF_B][x + HALF_B];
    
}


int main(int argc, char* argv[]) {
    

    input(argv[1]);
    //int B = 512;
    clock_t s = clock();

    //cudaGetDeviceProperties(&prop, DEV_NO);
    //printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d", prop.maxThreasPerBlock, prop.sharedMemPerBlock);

    //block_FW(B);
    
        int *host_D, *dev_D[2];
        unsigned long matrix = sizeof(int) * n * n;
        cudaHostRegister(Dist, matrix, cudaHostRegisterDefault);
        //cudaMalloc(&dev_D, matrix);
        //cudaMemcpy(dev_D, Dist, matrix, cudaMemcpyHostToDevice);

        int block_num = (n + B - 1) / B;//ceil(n, B)
        dim3 grid_dim1(1, 1);
        
        //dim3 grid_dim3(block_num, block_num);
        dim3 block_dim(HALF_B, HALF_B);//padding to multiple of 32
        
        
    #pragma omp parallel num_threads(2)
    {
        const int threadId = omp_get_thread_num();
        cudaSetDevice(threadId);

        cudaMalloc(&dev_D[threadId], matrix);

		int numblock_perThread = block_num / 2;
        const int yOffset = threadId == 1 ? numblock_perThread: 0;

        // If num of blocks is uneven, we make the second GPU to add "1"
        if(threadId == 1 && (block_num % 2) != 0) numblock_perThread += 1;

        dim3 grid_dim3(block_num, numblock_perThread);
        
        cudaMemcpy(dev_D[threadId] + yOffset*B*n, Dist + yOffset*B*n, B * n * sizeof(int) * numblock_perThread, cudaMemcpyHostToDevice);
        for(int i = 0; i < block_num; i++)
        {
            // Every thread has its own yOffset
            
            if ((i >= yOffset) && (i < (yOffset + numblock_perThread))) {
                cudaMemcpy(Dist + i * B * n, dev_D[threadId] + i * B * n, B * n * sizeof(int), cudaMemcpyDeviceToHost);
            }
            #pragma omp barrier

            cudaMemcpy(dev_D[threadId] + i * B * n, Dist + i * B * n, B * n * sizeof(int), cudaMemcpyHostToDevice);
            
            Block_FW_phase1<<<grid_dim1, block_dim>>>(n, i, dev_D[threadId]);
            Block_FW_phase2<<<block_num, block_dim>>>(n, i, dev_D[threadId]);
            Block_FW_phase3<<<grid_dim3, block_dim>>>(n, i, dev_D[threadId], yOffset);
            
            
        }

        cudaMemcpy(Dist + yOffset*B*n, dev_D[threadId] + yOffset*B*n, B * n * sizeof(int) * numblock_perThread, cudaMemcpyDeviceToHost);
        #pragma omp barrier
    }
    cudaFree(dev_D[0]);
    cudaFree(dev_D[1]);

    output(argv[2]);
    clock_t e = clock();
    double time = (double)(e - s) / CLOCKS_PER_SEC;
    printf("time: %f\n", time);
    return 0;
}