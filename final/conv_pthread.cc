#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <iomanip> //for fixed precision
#include <time.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <immintrin.h>
using namespace std;

int w, h;

// Global variables
int height, width, k;         // Matrix dimensions and kernel size
vector<vector<float>> Dist;   // Input matrix
vector<vector<float>> Mask;   // Kernel (Mask)
vector<vector<float>> Result; // Output matrix

// 全域變數
vector<float> *Dist_ptr;
vector<float> *Mask_ptr;
vector<float> *Result_ptr;


queue<int> task_queue;
mutex task_queue_mutex;
condition_variable task_queue_cv;
atomic<int> completed_tasks(0);

// 計算線程函數
// void worker_thread(int thread_id, int num_threads, int num_elements) {
//     clock_t s = clock();

//     // 每個執行緒處理自己的任務範圍
//     int start = thread_id * num_elements / num_threads;
//     int end = (thread_id + 1) * num_elements / num_threads;

//     int boundary = end - (end % 8);
    
//     for (int task_index = start; task_index < boundary; task_index += 8) {
//         int row = task_index / width;
//         int col = task_index % width;

//         // float sum = 0.0f;
//         // for (int ki = 0; ki < k; ++ki) {
//         //     for (int kj = 0; kj < k; ++kj) {
//         //         sum += (*Dist_ptr)[(row + ki) * width + (col + kj)] * (*Mask_ptr)[ki * k + kj];
//         //     }
//         // }

//         // (*Result_ptr)[task_index] = sum;
//         __m256 sum = _mm256_setzero_ps();

//             for (int ki = 0; ki < k; ++ki)
//             {
//                 for (int kj = 0; kj < k; ++kj)
//                 {
//                     __m256 pixel = _mm256_loadu_ps(&(*Dist_ptr)[(row + ki) * width + (col + kj)]);
//                     __m256 mask = _mm256_set1_ps((*Mask_ptr)[ki * k + kj]);
//                     sum = _mm256_add_ps(sum, _mm256_mul_ps(pixel, mask));
//                 }
//             }

//             _mm256_storeu_ps(&(*Result_ptr)[task_index], sum);
//     }
//     for (int task_index = boundary; task_index < end; task_index ++)
//     {
//         int row = task_index / width;
//         int col = task_index % width;

//         float sum = 0.0f;
//         for (int ki = 0; ki < k; ++ki) {
//             for (int kj = 0; kj < k; ++kj) {
//                 sum += (*Dist_ptr)[(row + ki) * width + (col + kj)] * (*Mask_ptr)[ki * k + kj];
//             }
//         }
//     }

//     clock_t e = clock();
//     double time_spent = (double)(e - s) / CLOCKS_PER_SEC;
//     printf("Time took to worker_thread %d is %f seconds\n", thread_id, time_spent);
// }

// void ParallelConvolution(vector<float> &Dist, vector<float> &Mask, vector<float> &Result, int num_threads) {
//     int result_height = height;
//     int result_width = width;
//     int num_elements = result_height * result_width;

//     // 初始化全域變數
//     Dist_ptr = &Dist;
//     Mask_ptr = &Mask;
//     Result_ptr = &Result;
//     Result.resize(num_elements, 0.0f);

//     // 創建計算線程
//     vector<thread> threads;
//     for (int i = 0; i < num_threads; ++i) {
//         threads.emplace_back(worker_thread, i, num_threads, num_elements);
//     }

//     // 等待所有計算線程完成
//     for (auto &th : threads) {
//         th.join();
//     }
// }
void worker_thread(int thread_id) {
    clock_t s = clock();
    while (true) {
        int task_index;
        {
            // 從工作佇列中取出任務
            unique_lock<mutex> lock(task_queue_mutex);
            task_queue_cv.wait(lock, [] { return !task_queue.empty(); });

            task_index = task_queue.front();
            task_queue.pop();
        }

        // 終止信號檢查
        if (task_index == -1) {
            break;
        }

        // 計算行與列
        int row = task_index / width;
        int col = task_index % width;

        // 執行卷積計算
        float sum = 0.0f;
        for (int ki = 0; ki < k; ++ki) {
            for (int kj = 0; kj < k; ++kj) {
                //int image_row = row + ki - k / 2;
                //int image_col = col + kj - k / 2;

                // 確保不超出影像邊界
                //if (image_row >= 0 && image_row < height && image_col >= 0 && image_col < width) {
                    sum += (*Dist_ptr)[(row + ki) * width + (col + kj)] * (*Mask_ptr)[ki * k + kj];
                    
                //}
            }
        }

        // 儲存結果
        (*Result_ptr)[task_index] = sum;

        // 更新完成任務計數
        completed_tasks.fetch_add(1);
    }
    clock_t e = clock();
    double time_spent = (double)(e - s) / CLOCKS_PER_SEC;
    printf("Time took to worker_thread%d is %f seconds\n\n", thread_id, time_spent);
}

// 管理線程函數
void manager_thread(int num_elements, int num_threads) {
    // 將初始任務分配給工作執行緒
    {
        lock_guard<mutex> lock(task_queue_mutex);
        for (int i = 0; i < num_elements; ++i) {
            task_queue.push(i);
        }

        // 將終止信號加入佇列
        for (int i = 0; i < num_threads; ++i) {
            task_queue.push(-1);
        }
    }

    // 通知所有工作執行緒
    task_queue_cv.notify_all();
}

void ParallelConvolution(vector<float> &Dist, vector<float> &Mask, vector<float> &Result, int num_threads) {
    int result_height = height;
    int result_width = width;
    int num_elements = result_height * result_width;

    // 初始化全域變數
    Dist_ptr = &Dist;
    Mask_ptr = &Mask;
    Result_ptr = &Result;
    Result.resize(num_elements, 0.0f);

    // 創建計算線程
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker_thread, i);
    }

    // 創建管理線程
    thread manager(manager_thread, num_elements, num_threads);

    // 等待管理線程完成
    manager.join();

    // 等待所有計算線程完成
    for (auto &th : threads) {
        th.join();
    }
}
void input(const char *infile, vector<float> &Dist, vector<float> &Mask)
{
    FILE *file = fopen(infile, "rb"); // Open file in binary mode
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read dimensions (h, w, k)
    fread(&h, sizeof(int), 1, file);
    fread(&w, sizeof(int), 1, file);
    fread(&k, sizeof(int), 1, file);

    width = w - k + 1;
    height = h - k + 1;

    // Allocate memory for Dist and Mask as float*
    Dist.resize(h * w, 0.0f);
    Mask.resize(k * k, 0.0f);

    // Read flattened Dist and Mask
    fread(Dist.data(), sizeof(float), h * w, file);
    fread(Mask.data(), sizeof(float), k * k, file);
    // cout << "Mask matrix (" << k << "x" << k << "):" << endl;
    // for (int i = 0; i < k; i++)
    // {
    //     for (int j = 0; j < k; j++)
    //     {
    //         cout << fixed << setprecision(4) << Mask[i * k + j] << ' ';
    //     }
    //     cout << endl;
    // }
    fclose(file);
}

void output(const char *outFileName, const vector<float> &Result)
{
    // Debugging output to console
    cout << "Result matrix (" << height << "x" << width << "):" << endl;
    // for (int i = 0; i < height; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         cout << fixed << setprecision(4) << Result[i * width + j] << ' ';
    //     }
    //     cout << endl;
    // }
    // Open the binary file for writing
    FILE *outfile = fopen(outFileName, "wb");

    if (!outfile)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write the dimensions (height and width) to the file
    fwrite(&height, sizeof(int), 1, outfile);
    fwrite(&width, sizeof(int), 1, outfile);

    // Write the matrix data row by row

    // Write the matrix data as a flat array (one row after another)
    fwrite(Result.data(), sizeof(float), height * width, outfile);

    fclose(outfile);
    cout << "Output written to file: " << outFileName << endl;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cerr << "Usage: ./conv <input_file> <output_file>" << endl;
        return EXIT_FAILURE;
    }

    vector<float> Dist, Mask;

    vector<float> Result; // 使用一維 vector 存儲所有結果
    // float *Dist, *Mask;
    // float *Result = new float[height * width];

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);
    // Input data
    double time_spent, io_time, o_time;
    clock_t start = clock();
    input(argv[1], Dist, Mask);
    clock_t end = clock();
    io_time = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("Time took to input is %f seconds\n\n", time_spent);
    
    // Perform Convolution
    start = clock();
    ParallelConvolution(Dist, Mask, Result, ncpus);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to compute matrix A of dimensions  on CPU is %f seconds\n\n", time_spent);

    // Output results
    // Output results
    start = clock();
    output(argv[2], Result);
    end = clock();
    o_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to io is %f seconds\n\n", o_time + io_time);

    // return;
}