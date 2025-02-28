#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	MPI_Init(&argc, &argv);
	int mpi_rank, mpi_size, omp_threads, omp_thread;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	unsigned long long batch = ceil(r / mpi_size);
	unsigned long long chunk = 5000;
	unsigned long long r_squraed = r * r;
	unsigned long long start = mpi_rank * batch;
	unsigned long long boundary = (mpi_rank == mpi_size - 1) ? r : (mpi_rank + 1) * batch;

#pragma omp parallel shared(chunk, r_squraed, k) reduction(+ : pixels)
	{

		// printf("Hello  rank %2d/%2d, thread %2d/%2d\n", mpi_rank, mpi_size,
		// 	   omp_thread, omp_threads);
		//unsigned long long y = 0;

#pragma omp for schedule(dynamic, chunk) nowait

		for (unsigned long long x = start; x < boundary; x += 2)
		{
            unsigned long long y = 0;
			if ((x + 1) < boundary)
			{
				unsigned long long arr[2] = {0, 0};
#pragma GCC ivdep // vectorization
				for (int i = 0; i < 2; i++)
				{

					arr[i] += ceil(sqrtl(r_squraed - (x + i) * (x + i)));
				}
				for (int i = 0; i < 2; i++)
				{

					y += arr[i];
				}
			}
			else
			{
				y += ceil(sqrtl(r_squraed - x * x));
			}
            pixels += y;
		}
		
		pixels %= k;
	}

	unsigned long long total_sum;
	MPI_Reduce(&pixels, &total_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (mpi_rank == 0)
	{
		printf("%llu\n", (4 * total_sum) % k);
	}
	MPI_Finalize();
}
