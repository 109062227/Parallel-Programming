#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>
#include <algorithm>

omp_lock_t lock;
int mpi_rank, mpi_size, ncpus;
int boundary, boundary1, boundary2;
int iters, width, height, row = 0, chunk;
double left, right, lower, upper, x_rlw, y_ulh;
int *image;
union AVX
{
    double arr[8];
    __m512d sum;
};
void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int bot_y = height - 1 - y;
            int idx = (bot_y % mpi_size) * chunk + (bot_y / mpi_size);
            int p = buffer[idx * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
int get_row()
{
    int ret = -1;
    if (row < height)
    {
        ret = row;
        row += mpi_size;
    }
    return ret;

    // static int row = mpi_rank * chunk;
    // if(row >= (mpi_rank+1)*chunk) return -1;
    // return row++;
}

int main(int argc, char **argv)
{

    /*yu---start*/
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    /*yu---end*/

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    y_ulh = (upper - lower) / height;
    x_rlw = (right - left) / width;
    row = mpi_rank;
    chunk = ceil(height / (double)mpi_size);

    /* allocate memory for image */
    image = (int *)malloc(width * chunk * sizeof(int));
    assert(image);

    int *total_image;
    if (mpi_rank == 0)
    {
        total_image = (int *)malloc(mpi_size * width * chunk * sizeof(int));
    }

    /*yu---start*/
    ncpus = CPU_COUNT(&cpu_set);
    omp_init_lock(&lock);

    boundary1 = width % 16;
    boundary2 = width - boundary1;
    boundary = (boundary1 == 0) ? width : boundary2;

#pragma omp parallel num_threads(ncpus) shared(image, mpi_rank)
    {
        // for (int j = 0; j < height; ++j) {

        int j;
        while (true)
        {
            omp_set_lock(&lock);
            j = get_row();
            if (j == -1)
            {
                omp_unset_lock(&lock);
                break;
            }
            omp_unset_lock(&lock);
            /* mandelbrot set */
            double y0 = j * y_ulh + lower;

            __m512d y0_sum = _mm512_set1_pd(y0);

            /*yu--add*/
            int index = (j / mpi_size) * width;
            for (int i = 0; i < boundary; i += 16)
            {
                AVX x0a, x0b, length_squared_a, length_squared_b, xa, xb, ya, yb;
                int repeat_a[8], repeat_b[8];

                for (int k = 0; k < 8; k++)
                {
                    // 第一組 (前 8 個元素)
                    x0a.arr[k] = (i + k) * x_rlw + left;
                    length_squared_a.arr[k] = 0;
                    xa.arr[k] = 0;
                    ya.arr[k] = 0;
                    repeat_a[k] = 0;

                    // 第二組 (後 8 個元素)
                    x0b.arr[k] = (i + 8 + k) * x_rlw + left;
                    length_squared_b.arr[k] = 0;
                    xb.arr[k] = 0;
                    yb.arr[k] = 0;
                    repeat_b[k] = 0;
                }

                int repeats = 0;

                while (repeats < iters)
                {
                    // 第一組
                    __m512d temp_a = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(xa.sum, xa.sum), _mm512_mul_pd(ya.sum, ya.sum)), x0a.sum);
                    ya.sum = _mm512_add_pd(_mm512_mul_pd(_mm512_add_pd(ya.sum, ya.sum), xa.sum), y0_sum);
                    xa.sum = temp_a;
                    length_squared_a.sum = _mm512_add_pd(_mm512_mul_pd(xa.sum, xa.sum), _mm512_mul_pd(ya.sum, ya.sum));

                    // 第二組
                    __m512d temp_b = _mm512_add_pd(_mm512_sub_pd(_mm512_mul_pd(xb.sum, xb.sum), _mm512_mul_pd(yb.sum, yb.sum)), x0b.sum);
                    yb.sum = _mm512_add_pd(_mm512_mul_pd(_mm512_add_pd(yb.sum, yb.sum), xb.sum), y0_sum);
                    xb.sum = temp_b;
                    length_squared_b.sum = _mm512_add_pd(_mm512_mul_pd(xb.sum, xb.sum), _mm512_mul_pd(yb.sum, yb.sum));

                    for (int k = 0; k < 8; k++)
                    {
                        if (length_squared_a.arr[k] >= 4.0 && repeat_a[k] == 0)
                        {
                            repeat_a[k] = repeats + 1;
                        }
                        if (length_squared_b.arr[k] >= 4.0 && repeat_b[k] == 0)
                        {
                            repeat_b[k] = repeats + 1;
                        }
                    }

                    ++repeats;

                    bool complete_a = std::all_of(repeat_a, repeat_a + 8, [](int r)
                                                  { return r != 0; });
                    bool complete_b = std::all_of(repeat_b, repeat_b + 8, [](int r)
                                                  { return r != 0; });

                    if (complete_a && complete_b)
                        break;
                }

                for (int k = 0; k < 8; k++)
                {
                    image[index + i + k] = repeat_a[k];
                    image[index + i + 8 + k] = repeat_b[k];
                }
                // image[(j / mpi_size) * width + i] = repeat1;
                // image[(j / mpi_size) * width + i + 1] = repeat2;
                // image[(j / mpi_size) * width + i + 2] = repeat3;
                // image[(j / mpi_size) * width + i + 3] = repeat4;
            }
            if (boundary == boundary2) // odd
            {
                for (int i = boundary; i < width; ++i)
                {
                    double x0 = i * x_rlw + left; // Calculate x0 for remaining pixels
                    int repeats = 0;
                    double x = 0, y = 0, length_squared = 0;

                    while (repeats < iters && length_squared < 4)
                    {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[index + i] = repeats; // Store the result
                }
            }

            //}
        }
    }
    /*send result back to root */
    MPI_Gather(image, width * chunk, MPI_INT, total_image, width * chunk, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    /*yu---end*/
    /* mandelbrot set */

    /* draw and cleanup */
    if (mpi_rank == 0)
    {
        write_png(filename, iters, width, height, total_image);
    }

    free(image);
    if (mpi_rank == 0)
    {
        free(total_image);
    }
    omp_destroy_lock(&lock);
}