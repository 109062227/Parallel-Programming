#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <algorithm>
#include <immintrin.h> // include Intel Intrinsics
#include <iostream>
using namespace std;
struct s
{
	int threadid;
	unsigned long long *r;
	unsigned long long *k;
	unsigned long long *pixels;
	unsigned long long *batch_size;
	pthread_mutex_t *mutex;
};
void* collect_pixel(void* arg) {
	struct s *ss = (struct s *)arg;
	unsigned long long r_squraed = *(ss->r) * (*(ss->r));
	unsigned long long y = 0;
	unsigned long long boundary = min(*(ss->r), (ss->threadid+1)*(*(ss->batch_size)));
	unsigned long long start = (ss->threadid)*(*(ss->batch_size));

	if(start < *(ss->r)) {
    
		
		for (unsigned long long x = start; x < boundary; x += 2) {
			if((x+1) < boundary){
				unsigned long long arr[2];
				#pragma GCC ivdep//vectorization
				for(int i = 0; i < 2; i++){
					arr[i] += ceil(sqrtl(r_squraed - (x+i) * (x+i)));
				}
				for (int i = 0; i < 2; i++) {
					y += arr[i];
				}

				
			}
			else {
				y += ceil(sqrtl(r_squraed - x*x));
				
			}
			
		}
	}
	else pthread_exit(NULL);
	
	
	pthread_mutex_lock (ss->mutex);
	*(ss->pixels) += y;
	*(ss->pixels) %= *(ss->k);
	pthread_mutex_unlock (ss->mutex);
	pthread_exit(NULL);

}
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	//cpu_set_t cpuset;
	//sched_getaffinity(0, sizeof(cpuset), &cpuset);
	//unsigned long long ncpus = CPU_COUNT(&cpuset);

	/*yu--add*/
	//get #cpu per process
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	unsigned long long *batch_size = new unsigned long long (ceil(r / double(ncpus)));
	struct s **s = new struct s*[ncpus];
    //int num_threads = atoi(argv[1]);
    pthread_t threads[ncpus];
	pthread_mutex_t mutex;
    //int rc;
    int ID[ncpus];
    int t;
    for (t = 0; t < ncpus; t++) {
		s[t] = new struct s;
		s[t] -> r = &r;
		s[t] -> k = &k;
		s[t] -> pixels = &pixels;
		s[t] -> batch_size = batch_size;
		s[t] -> threadid = t;
		s[t] -> mutex = &mutex;
		pthread_create(&threads[t], NULL, collect_pixel, (void*)s[t]);

        // ID[t] = t;
        // printf("In main: creating thread %d\n", t);
        // rc = pthread_create(&threads[t], NULL, collect_pixel, (void*)&ID[t]);
        // if (rc) {
        //     printf("ERROR; return code from pthread_create() is %d\n", rc);
        //     exit(-1);
        // }
    }
	//waiting all threads done
	for(t = 0; t < ncpus; t++){
		pthread_join(threads[t], NULL);
	} 
	printf("%llu\n", (4 * pixels) % k);
	#pragma GCC ivdep
	for(t = 0; t < ncpus; t++){
		delete s[t];
	}
	delete [] s;
	delete batch_size;
	pthread_exit(NULL);
	/*yu--end*/

	
	//printf("%llu\n", (4 * pixels) % k);
}
