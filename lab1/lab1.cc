#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
int main(int argc, char **argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size); // #process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// printf("rank: %d      size: %d\n", rank, size);
	if (argc != 3)
	{
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;

	unsigned long long total_pixels = 0;
	unsigned long long r_square = r * r;
	unsigned long long tmp_pixels = 0;

	for (unsigned long long x = rank; x < r; x += size)
	{
		unsigned long long y = ceil(sqrtl(r_square - x * x));
		tmp_pixels += y;

		// reduce computation
		// if (x % 1000 == 0)
		// {
		// 	tmp_pixels %= k;
		// }
	}
	pixels = tmp_pixels % k;
	MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); // plus all pixels value and put into sum
																							   // MPI_COMM_WORLD : all process join this

	if (rank == 0)
	{
		printf("%llu\n", (4 * total_pixels) % k);
	}

	MPI_Finalize();
}
