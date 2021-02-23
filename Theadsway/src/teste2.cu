/*
//algortimo p-twdtw
#include <stdio.h>
#define min2(x,y) (x < y ? x : y)


__global__ void loop()
{

  int num_rows = 4;
  int num_cols = 4;
  int tid = threadIdx.x;


 for (int si = 0; si < num_rows; si++) {
      if (tid <= min2(si, num_cols - 1)){
    	  printf("si%d\n", si);
          int i = si - tid;
          int j = tid;
          printf("A tid %d calcula o elemento i %d e j %d\n" , tid, i ,j);
      }
      __syncthreads();
  }


  for (int sj = num_cols - 2; sj >= 0; sj--) {
	  if (tid <= min2(sj, num_rows - 1)) {
		  printf("sj%d\n", sj);
		  int i = num_rows - tid - 1;
		  int j = num_cols - (sj - tid) - 1;
		  printf("A tid %d calcula o elemento i %d e j %d\n" , tid, i ,j);
	  }
	  __syncthreads();
   }


}

int main()
{

  loop<<<1, 4>>>();
  cudaDeviceSynchronize();
}
*/


/*
//
t0: j >= 0 e j < 2
t1: j >= 2 e j < 4
t2: j >= 4 e j < 6

O i só muda quando o j chega no final.(importante)

base = (tid * num_per_thread) + si * num_cols;
---------------------------------------------------------
t0:
id.x = 0
j >= base & j < (base + num_per_thread)

index >= base && index < (base + num_per_thread)

i = 0
index_min = base = 0 * 2 + 0 * 6 = 0
index_max = base + num_per_thread = 0 + 2 = 2
index >= 0 && index < 2

i = 1
index_min = base = 0 * 2 + 1 * 6 = 6
index_max = base + num_per_thread = 6 + 2 = 8
index >= 6 && index < 8
---------------------------------------------------------
t1:
id.x = 1
j >= base & j < (base + num_per_thread)

i = 0
index_min = base = 1 * 2 + 0 * 6 = 2
index_max = base + num_threads = 4
index >= 2 && index < 4

i = 1
index_min = base = 1 * 2 + 1 * 6 = 8
index_max = base + num_threads = 10
index >= 8 && index < 10
---------------------------------------------------------
t2:
id.x = 2
j >= base & j < (base + num_per_thread)

i = 0
index_min = base = 2 * 2 + 0 * 6 = 4
index_max = base + num_threads = 6
index >= 4 && index < 6

i = 1
index_min = base = 2 * 2 + 1 * 6 = 10
index_max = base + num_threads = 12
index >= 10 && index < 12
---------------------------------------------------------

j >= base & j < (base + num_per_thread) só pra entender a faixa de coluna de cada thread


for (i = 0; i < num_rows; i++) {
	base = (id.x * num_per_thread) + i * num_cols

	if (diagonal_superior) {
		for (index = base; index < base + num_per_threads; index++) {
			calcula custo acumulado(index)
		}
	}

	if (diagonal_inferior) {
		for (index = base; index < base + num_per_threads; index++) {
			calcula custo acumulado(index)
		}
	}
}*/


#include <stdio.h>
#define min2(x,y) (x < y ? x : y)
#define max2(x,y) (x > y ? x : y)

__global__ void loop(int num_threads)
{

	int num_cols = 6;
	int window  = (num_cols / num_threads); //window num_per_thread
	int num_rows = 6;
	int tid = threadIdx.x;


	for (int si = 0; si < num_rows; si++) {
		int base = (tid * window) + (si - tid) * num_cols;
	    int aux = tid * (window - 1);

		if (tid <= min2(si, num_cols-1)) {
			printf("quantidade de janelas no passo %d\n", si);
			int auxi = si -tid;
			for (int index = base; index < base + window; index++) {

				  //printf("A tid %d calcula o elemento %d\n" , tid, index);

				  int i = si - tid + auxi;
			      int j = tid + aux;
			      aux = aux + 1;

			    printf("A tid %d calcula o elemento i %d e j %d\n" , tid, i ,j);


			}
		}
		__syncthreads();
	}


	 int si = num_rows -1;
	 int aux = 0;
	 for (int sj = (num_cols/window) - 2; sj >= 0; sj--) {
		 int base = (tid * window) + ((si - tid) * num_cols) + window + aux;
		 int auxj = 0;
		 aux = aux + window;
		 if (tid <= min2(sj, num_rows - 1)) {
			 printf("quantidade de janelas no passo %d\n", sj);
			  for (int index = base; index < base + window; index++) {
				  //printf("A tid %d calcula o elemento %d\n" , tid, index);
				  int i = num_rows - tid - 1;
				  int j = num_cols - (window * sj) - window + auxj + tid*window;
				  auxj = auxj + 1;
				  printf("A tid %d calcula o elemento i %d e j %d\n" , tid, i ,j);

		  }

	  }
	  __syncthreads();
   }



}

int main()
{
  int num_threads = 3;
  loop<<<1, num_threads>>>(num_threads);
  cudaDeviceSynchronize();
}



