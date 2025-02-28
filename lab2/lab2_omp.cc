#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <stdlib.h>
#include <pthread.h>
int main(int argc, char **argv)
{
	if (argc != 3)
	{
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long chunk = 5000;
	unsigned long long r_squraed = r * r;

#pragma omp parallel shared(chunk, r_squraed, k) reduction(+ : pixels)
	{

#pragma omp for schedule(dynamic, chunk) nowait
		for (unsigned long long x = 0; x < r; x += 2)
		{
			unsigned long long y[2] = {0, 0};
			// unsigned long long y = 0;
			if ((x + 1) < r)
			{
				// unsigned long long arr[2] = {0, 0};
#pragma GCC ivdep // vectorization
				for (int i = 0; i < 2; i++)
				{
					y[i] += ceil(sqrtl(r_squraed - (x + i) * (x + i)));
				}
			}
			else
			{
				y[0] = ceil(sqrtl(r_squraed - x * x));
			}
			pixels += (y[0] + y[1]);
		}
		pixels %= k;
		// for(int i=0; i<2; i++){

		// 	pixels += y[i];
		// 	pixels %= k;
		// }
	}
	// for (unsigned long long x = 0; x < r; x++) {
	// 	unsigned long long y = ceil(sqrtl(r*r - x*x));
	// 	pixels += y;
	// 	pixels %= k;
	// }
	printf("%llu\n", (4 * pixels) % k);
}
