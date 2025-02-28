#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <iostream>
using namespace std;
// void compute_boundary(int *cnt, int chunk, int arr_size, int start)
// {
//     // if (start >= arr_size)
//     // {
//     //     return 0;
//     // }
//     // else
//     // {
//     //     return min(int(chunk), arr_size - start);
//     // }

//     cnt[0] = (start >= arr_size) ? 0 : min(int(chunk), arr_size - start);                   // data cnt
//     cnt[1] = (start + chunk >= arr_size) ? 0 : min(int(chunk), arr_size - (start + chunk)); // right cnt
//     cnt[2] = (start >= arr_size + chunk) ? 0 : min(int(chunk), arr_size - (start - chunk)); // left cnt
// }

inline void merge_and_sort(float *data, float *buffer, float *merge, bool is_small, int *cnt)
{

    int i, j, k;
    if (is_small)
    {
        i = j = k = 0;
        while (k < cnt[0])
        {
            // if(j >= cnt[1]){
            //     merge[k++] = data[i++];
            // }
            // else{
            //     merge[k++] = data[i]<buffer[j]?data[i++]:buffer[j++];
            // }
            // Compare j with cnt[1] to decide whether to take from data or buffer
            int take_from_data = (j >= cnt[1]) || (data[i] < buffer[j]);

            // Use min/max to select the value to merge
            merge[k++] = take_from_data ? data[i++] : buffer[j++];
        }
    }
    else
    {
        i = k = cnt[0] - 1;
        j = cnt[2] - 1;
        while (k >= 0)
        {
            merge[k--] = data[i] > buffer[j] ? data[i--] : buffer[j--];
        }
    }
    for (i = 0; i < cnt[0]; i++)
    {
        data[i] = merge[i];
    }
}
// bool check_edge(int chunk, int start, int arr_size, int *cnt)
// {
//     if (cnt[0] != 0)
//     {
//         if (start + chunk >= arr_size)
//         {
//             return true;
//         }
//     }
//     return false;
// }

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr_size = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    int chunk = ceil(arr_size / double(size));
    int start = rank * chunk;
    int cnt[3] = {0, 0, 0};
    //compute_boundary(cnt, chunk, arr_size, start);
    cnt[0] = (start >= arr_size) ? 0 : min(chunk, arr_size - start);                   // data cnt
    cnt[1] = (start + chunk >= arr_size) ? 0 : min(chunk, arr_size - (start + chunk)); // right cnt
    cnt[2] = (start >= arr_size + chunk) ? 0 : min(chunk, arr_size - (start - chunk)); // left cnt
    int edge = cnt[0] - 1;
    int size_edge = size - 1;
    int right_rank = rank + 1;
    int left_rank = rank - 1;
    // int data_cnt = compute_boundary(rank, chunk, arr_size, start);
    // int right_cnt = compute_boundary(rank + 1, chunk, arr_size, start + chunk);
    // int left_cnt = compute_boundary(rank - 1, chunk, arr_size, start - chunk);
    int sorted = 0;
    int odd_sorted = 0;
    int even_sorted = 0;
    bool phase = true;
    // bool edge = check_edge(chunk, start, arr_size, cnt);

    MPI_File input_file, output_file;
    float *data = new float[cnt[0]];
    float *buffer = new float[chunk];
    float *merge = new float[cnt[0]];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    MPI_File_read_at(input_file, sizeof(float) * start, data, cnt[0], MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    //printf("rank %d got float: %f\n", rank, data[0]);

    // sort partition i
    if (cnt[0] > 0)
        boost::sort::spreadsort::float_sort(data, data + cnt[0]);

    /*perform odd-even sort*/
    int r = rank & 1;
    for (int i = 0; i <= size; i++)
    {
        sorted = 0;

        // on left handside
        if (r != phase)
        {
            if (rank != size_edge && cnt[1] != 0 && cnt[0] != 0)
            {
                MPI_Sendrecv(&data[edge], 1, MPI_FLOAT, right_rank, 0, buffer, 1, MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[edge] > buffer[0])
                {
                    sorted = 0;
                    MPI_Sendrecv(data, cnt[0], MPI_FLOAT, right_rank, 0, buffer, cnt[1], MPI_FLOAT, right_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_and_sort(data, buffer, merge, true, cnt);
                    // merge_left_side(data, buffer, merge, cnt);
                }
                else
                    sorted = 1;
            }
        }
        else // on right handside
        {
            if (rank != 0 && cnt[2] != 0 && cnt[0] != 0)
            {
                MPI_Sendrecv(&data[0], 1, MPI_FLOAT, left_rank, 0, buffer, 1, MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (data[0] < buffer[0])
                {
                    sorted = 0;
                    MPI_Sendrecv(data, cnt[0], MPI_FLOAT, left_rank, 0, buffer, cnt[2], MPI_FLOAT, left_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_and_sort(data, buffer, merge, false, cnt);
                    // merge_right_side(data, buffer, merge, cnt);
                }
                else
                    sorted = 1;
            }
        }

        odd_sorted = 0;
        even_sorted = 0;
        if (i >= 20)
        {
            if (!phase)
            {
                MPI_Allreduce(&sorted, &odd_sorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            else
            {
                MPI_Allreduce(&sorted, &even_sorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            if (odd_sorted + even_sorted == size)
                break;
        }
        phase = !phase;
    }
    /*endddd*/

    

    if (cnt[0] != 0)
        MPI_File_write_at(output_file, sizeof(float) * start, data, cnt[0], MPI_FLOAT, MPI_STATUS_IGNORE);

    
    MPI_File_close(&output_file);

    delete[] data;
    delete[] buffer;
    delete[] merge;

    MPI_Finalize();

    return 0;
}