#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <iomanip> //for fixed precision
#include <omp.h>
#include <mpi.h>
#include <math.h>
using namespace std;

int w, h, k;
int width = 0, height = 0, idx = 0;
// odd procs send, event procs receive
double comm_time;
clock_t s_mpi, e_mpi;
void ParallelConvolution(std::vector<float> &Dist, std::vector<float> &Mask, std::vector<float> &Result, int rank, int size)
{
    int result_height = height;
    int result_width = width;
    int num_elements = result_height * result_width;

    if (rank == 0)
    {
        Result.resize(num_elements, 0.0f);

        // 發送初始任務給所有工作進程
        for (int worker = 1; worker < size; ++worker)
        {
            if (worker - 1 < num_elements)
            {
                int task = worker - 1;
                s_mpi = clock();
                MPI_Send(&task, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                e_mpi = clock();
                comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;
            }
            else
            {
                int terminate_signal = -1;
                s_mpi = clock();
                MPI_Send(&terminate_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                e_mpi = clock();
                comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;
            }
        }

        int completed_tasks = 0;
        while (completed_tasks < num_elements)
        {
            MPI_Status status;
            int task_index;
            float result;

            // 接收結果
            s_mpi = clock();
            MPI_Recv(&task_index, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&result, 1, MPI_FLOAT, status.MPI_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            e_mpi = clock();
            comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;

            Result[task_index] = result;
            completed_tasks++;

            // 如果還有未處理的任務，發送新任務
            if (completed_tasks + size - 1 <= num_elements)
            {
                int new_task = completed_tasks + size - 1 - 1;
                s_mpi = clock();
                MPI_Send(&new_task, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                e_mpi = clock();
                comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;
            }
            else
            {
                int terminate_signal = -1;
                s_mpi = clock();
                MPI_Send(&terminate_signal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                e_mpi = clock();
                comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;
            }
        }
    }
    else
    {
        while (true)
        {
            int task_index;
            MPI_Status status;

            // 接收任務
            s_mpi = clock();
            MPI_Recv(&task_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            e_mpi = clock();
            comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;

            // 檢查是否為終止信號
            if (task_index == -1)
            {
                break;
            }

            // 執行卷積計算
            std::vector<float> local_Dist(k * k, 0.0f);

            int row = task_index / result_width;
            int col = task_index % result_width;

            for (int ki = 0; ki < k; ++ki)
            {
                for (int kj = 0; kj < k; ++kj)
                {
                    local_Dist[ki * k + kj] = Dist[(row + ki) * width + (col + kj)];
                }
            }

            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                for (int kj = 0; kj < k; ++kj)
                {
                    sum += local_Dist[ki * k + kj] * Mask[ki * k + kj];
                }
            }

            // 發送結果回 Rank 0
            s_mpi = clock();
            MPI_Send(&task_index, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&sum, 1, MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
            e_mpi = clock();
            comm_time += (double)(e_mpi - s_mpi) / CLOCKS_PER_SEC;
        }
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

    // cout << "Dist matrix: \n";
    // for (int i = 0; i < h; ++i)
    // {
    //     for (int j = 0; j < w; ++j)
    //     {
    //         cout << fixed << setprecision(4) << Dist[i * w + j] << ' ';
    //     }
    //     cout << endl;
    // }
    // cout << "Mask matrix: \n";
    // for (int i = 0; i < k; ++i)
    // {
    //     for (int j = 0; j < k; ++j)
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
    // if (ferror(outfile))
    // {
    //     perror("Error closing file");
    // }
    cout << "Output written to file: " << outFileName << endl;
}
void Convolution(vector<float> &Dist, vector<float> &Mask, vector<float> &Result)
{
    int i, j, ki, kj;
    float sum;
    Result.resize(height * width, 0.0f);
    // 遍歷每個像素
    for (i = 0; i < height; ++i)
    {
        for (j = 0; j < width; ++j)
        {
            sum = 0.0f;

            // 在當前像素位置進行卷積運算
            for (ki = 0; ki < k; ++ki)
            {
                for (kj = 0; kj < k; ++kj)
                {
                    // 進行加權和計算
                    // printf("Dist[%d]: %f, Mask[%d]: %f\n", (i + ki) * width + (j + kj), Dist[(i + ki) * width + (j + kj)], ki * k + kj, Mask[ki * k + kj]);
                    sum += Dist[(i + ki) * width + (j + kj)] * Mask[ki * k + kj];
                }
            }

            // 儲存卷積結果
            Result[i * width + j] = sum;
        }
    }
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
    //clock_t start, end;
    //double time_spent;
    // float *Dist, *Mask;
    // float *Result = new float[height * width];

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);
    // Input data
    double time_spent, io_time;
    clock_t start = clock();
    input(argv[1], Dist, Mask);
    clock_t end = clock();
    
    io_time += (double)(end - start) / CLOCKS_PER_SEC;
    //printf("Time took to input is %f seconds\n\n", time_spent);
    Result.resize(height * width, 0.0f);

    // Perform Convolution
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start = clock();

    if(size > 1)
        ParallelConvolution(Dist, Mask, Result, rank, size);
    else 
        Convolution(Dist, Mask, Result);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    if (rank == 0)
        printf("Time took to compute matrix A of dimensions  on CPU is %f seconds\n\n", time_spent - comm_time);

    // Output results
    // Output results
    // printf("rank %d reached barrier\n", rank);
    // MPI_Barrier(MPI_COMM_WORLD);
    // printf("rank %d passed barrier\n", rank);
    if (rank == 0)
    {
        start = clock();
        output(argv[2], Result);
        end = clock();
        io_time += (double)(end - start) / CLOCKS_PER_SEC;
        printf("Time took to io is %f seconds\n\n", io_time);
        printf("Time took to communication is %f seconds\n\n", comm_time);
    }

    MPI_Finalize();
    // return;
}