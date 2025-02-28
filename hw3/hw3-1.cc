#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <bits/stdc++.h>
#include <pthread.h>
#define INF 1073741823
using namespace std;

int n = 0, m = 0, chunk, ncpus;
vector<vector<int>> Dist;
pthread_barrier_t barrier;

void* cal(void *arg){
  int id = *((int*)arg);

  int start = id*chunk;
  int end = min((id + 1) * chunk, n);
  for(int k=0; k<n; k++){
    for(int i=start; i<end; i++){
      if(Dist[i][k] == INF || i == k) continue;
      for(int j = 0; j < n; j++){
        if(Dist[i][k] + Dist[k][j] < Dist[i][j]){
          Dist[i][j] = Dist[i][k] + Dist[k][j];
        }
      }
    }
    pthread_barrier_wait(&barrier);
  }
  return NULL;
}

int main(int argc, char** argv) {
    
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);
    pthread_barrier_init(&barrier, NULL, (unsigned)ncpus);

    ifstream f(argv[1]);
    f.seekg(0, ios_base::end);
    f.seekg(0, ios_base::beg);
    f.read((char*)&n, sizeof n);
    f.read((char*)&m, sizeof m);

    chunk = ceil(n / double(ncpus));

    vector<int> row;
    row.assign(n, INF);
    Dist.assign(n, row);

    for(int j=0; j<n; j++){
      Dist[j][j] = 0;
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        f.read((char*)pair, sizeof pair);
        Dist[pair[0]][pair[1]] = pair[2];
    }

    pthread_t thread[ncpus];
    int threadid[ncpus];

    for(int i=0; i<ncpus; i++){
      threadid[i] = i;
      pthread_create(&thread[i], NULL, cal, &threadid[i]);
    }

    for(int i=0; i<ncpus; i++){
      pthread_join(thread[i], NULL);
    }
    ofstream fout(argv[2]);
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        fout.write((char*)(&Dist[i][j]), sizeof(int));
      }
    }
    
    fout.close();
    pthread_exit(NULL);

    return 0;
}


