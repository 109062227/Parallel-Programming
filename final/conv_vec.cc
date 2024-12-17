#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <iomanip> //for fixed precision
#include <omp.h>
#include <immintrin.h>

using namespace std;

int w, h, k;
int width = 0, height = 0;

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

    fclose(file);
}

void output(const char *outFileName, const vector<float> &Result)
{
    // Debugging output to console
    cout << "Result matrix (" << height << "x" << width << "):" << endl;

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

void Convolution(vector<float> &Dist, vector<float> &Mask, vector<float> &Result)
{
    int i, j, ki, kj;
    
    float sum;
    Result.resize(height * width, 0.0f);
    int in_offset, j_offset;
    // 遍歷每個像素
    // omp_set_num_threads(6);

    // #pragma omp parallel for schedule(dynamic)
    int boundary = width - (width % 8);

    for (int i = 0; i < height; ++i)
    {
        int j;
        // 處理對齊部分
        for (j = 0; j < boundary; j += 8)
        {
            __m256 sum = _mm256_setzero_ps();

            for (int ki = 0; ki < k; ++ki)
            {
                for (int kj = 0; kj < k; ++kj)
                {
                    __m256 pixel = _mm256_loadu_ps(&Dist[(i + ki) * width + j + kj]);
                    __m256 mask = _mm256_set1_ps(Mask[ki * k + kj]);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(pixel, mask));
                }
            }

            _mm256_storeu_ps(&Result[i * width + j], sum);
        }

        // 處理剩餘部分
        for (j = boundary; j < width; ++j)
        {
            float sum = 0.0f;

            for (int ki = 0; ki < k; ++ki)
            {
                for (int kj = 0; kj < k; ++kj)
                {
                    sum += Dist[(i + ki) * width + j + kj] * Mask[ki * k + kj];
                }
            }

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
    // float *Dist, *Mask;
    // float *Result = new float[height * width];

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);
    // Input data
    clock_t start = clock();
    input(argv[1], Dist, Mask);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to input is %f seconds\n\n", time_spent);

    // Perform Convolution
    start = clock();
    Convolution(Dist, Mask, Result);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to compute matrix A of dimensions  on CPU is %f seconds\n\n", time_spent);

    // Output results
    // Output results
    start = clock();
    output(argv[2], Result);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to output is %f seconds\n\n", time_spent);

    // return;
}
