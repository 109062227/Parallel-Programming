#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <vector>
#include <iomanip> //for fixed precision
#include <time.h>
#include <cmath>
#include <cassert>
using namespace std;

int w, h, k;
int width, height;
// 展平矩陣的函數，將 2D 子矩陣轉換為一維向量
std::vector<float> conv_row(const std::vector<std::vector<float>> &mat, int startRow, int startCol, int kernelSize)
{
    std::vector<float> row;
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            row.push_back(mat[startRow + i][startCol + j]);
        }
    }
    return row;
}

// 將矩陣展平為提取矩陣（im2col 操作）
std::vector<std::vector<float>> unfold(const std::vector<std::vector<float>> &mat, int kernelSize)
{
    int n = mat.size();    // 輸入矩陣的行數
    int m = mat[0].size(); // 輸入矩陣的列數

    int outputRows = n - kernelSize + 1;
    int outputCols = m - kernelSize + 1;
    int kernelElements = kernelSize * kernelSize;

    // 創建展平後的矩陣
    std::vector<std::vector<float>> unfolded(outputRows * outputCols, std::vector<float>(kernelElements));

    int row = 0;
    for (int i = 0; i < outputRows; ++i)
    {
        for (int j = 0; j < outputCols; ++j)
        {
            unfolded[row] = conv_row(mat, i, j, kernelSize);
            ++row;
        }
    }
    return unfolded;
}

// 計算矩陣乘法
std::vector<float> matmul(const std::vector<std::vector<float>> &A, const std::vector<float> &B)
{

    assert(A[0].size() == B.size()); // 確保列數相符
    std::vector<float> result(A.size(), 0.0f);

    for (size_t i = 0; i < A.size(); ++i)
    {
        for (size_t j = 0; j < B.size(); ++j)
        {
            result[i] += A[i][j] * B[j];
        }
    }
    return result;
}
void input(const char *infile, vector<vector<float>> &Dist, vector<vector<float>> &Mask)
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

    // Resize 2D vectors
    Dist.resize(h, vector<float>(w));
    Mask.resize(k, vector<float>(k));

    // Allocate a 1D array for binary reading
    vector<float> flatDist(h * w);
    vector<float> flatMask(k * k);

    // Read flattened Dist and Mask
    fread(flatDist.data(), sizeof(float), h * w, file);
    fread(flatMask.data(), sizeof(float), k * k, file);

    // Map 1D array to 2D vector for Dist
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            Dist[i][j] = flatDist[i * w + j];
        }
    }

    // Map 1D array to 2D vector for Mask
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            Mask[i][j] = flatMask[i * k + j];
        }
    }
    // print check
    // for (int i = 0; i < h; ++i)
    // {
    //     for (int j = 0; j < w; ++j)
    //     {
    //         cout << Dist[i][j] << ' ';
    //     }
    //     cout << endl;
    // }
    // for (int i = 0; i < k; ++i)
    // {
    //     for (int j = 0; j < k; ++j)
    //     {
    //         cout << Mask[i][j] << ' ';
    //     }
    //     cout << endl;
    // }

    fclose(file);
}
void output(const char *outFileName, const vector<vector<float>> &Result)
{
    // Debugging output to console
    cout << "Result matrix (" << height << "x" << width << "):" << endl;
    // for (int i = 0; i < height; i++)
    // {
    //     for (int j = 0; j < width; j++)
    //     {
    //         cout << fixed << setprecision(4) << Result[i][j] << ' ';
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
    for (int i = 0; i < height; ++i)
    {
        fwrite(Result[i].data(), sizeof(float), width, outfile);
    }

    fclose(outfile);
    cout << "Output written to file: " << outFileName << endl;
}

void Convolution(const vector<vector<float>> &Dist, const vector<vector<float>> &Mask, vector<vector<float>> &Result)
{
    int kernelSize = Mask.size(); // 卷積核的大小
    int outputRows = Dist.size() - kernelSize + 1;
    int outputCols = Dist[0].size() - kernelSize + 1;
    // 展平輸入矩陣
    clock_t start = clock();
    std::vector<std::vector<float>> X_unfold = unfold(Dist, kernelSize);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time to unflold is %f seconds\n\n", time_spent);
    // 展平卷積核
    std::vector<float> W_flat;
    for (const auto &row : Mask)
    {
        for (float val : row)
        {
            W_flat.push_back(val);
        }
    }

    // 通過矩陣乘法計算卷積
    start = clock();
    std::vector<float> Y_flat = matmul(X_unfold, W_flat);
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to matmul is %f seconds\n\n", time_spent);

    // 重塑結果為輸出矩陣

    std::vector<std::vector<float>> Y(outputRows, std::vector<float>(outputCols));
    Result.resize(height, vector<float>(width, 0));

    for (int i = 0; i < outputRows; ++i)
    {
        for (int j = 0; j < outputCols; ++j)
        {

            Result[i][j] = Y_flat[i * outputCols + j];
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

    vector<vector<float>> Dist, Mask, Result;

    // Input data
    input(argv[1], Dist, Mask);

    // Perform Convolution
    clock_t start = clock();
    Convolution(Dist, Mask, Result);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time took to compute matrix A of dimensions  on CPU is %f seconds\n\n", time_spent);

    // Output results
    output(argv[2], Result);

    return EXIT_SUCCESS;
}
