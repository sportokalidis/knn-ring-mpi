#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include <mpi.h>


#define PROS 4

typedef struct knnresult{
  int * nidx;
  double * ndist;
  int m;
  int k;
} knnresult;

typedef struct distIdx{
  double distance;
  int index;
} distIdx;

distIdx qselect(double *tArray, int *index, int len, int k) {
	#	define SWAP(a, b) { temp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = temp; }
  #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
  distIdx c;
	int i, st;
	double temp;
  int tmp;
	// double * tArray = (double * ) malloc(len * sizeof(double));
	// for(int i=0; i<len; i++){
	// 	tArray[i] = v[i];
	// }
  // double * tIdx = (double * ) malloc(len * sizeof(double));
  // for(int i=0; i<len; i++){
  //   tArray[i] = v[i];
  // }
	for (st = i = 0; i < len - 1; i++) {
		if (tArray[i] > tArray[len-1]) continue;
		SWAP(i, st);
    SWAPINDEX(i,st);
		st++;
	}
	SWAP(len-1, st);
  SWAPINDEX(len-1,st);
  if (k == st){
    c.distance = tArray[st];
    c.index = index[st];
    return c;
  }
  else if (k < st) qselect(tArray, index,st, k);
  else qselect(tArray + st, index + st, len - st, k - st);

	// return k == st	? tArray[st] : st > k	? qselect(tArray, st, k) : qselect(tArray + st, len - st, k - st);
}

// void * qselect(double *tArray,int *index, int len, int k) {
// 	#	define SWAP(a, b) { tmp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = tmp; }
//   #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
// 	int i, st;
// 	double tmp;
// 	// double * tArray = (double * ) malloc(len * sizeof(double));
// 	// for(int i=0; i<len; i++){
// 	// 	tArray[i] = v[i];
// 	// }
// 	for (st = i = 0; i < len - 1; i++) {
// 		if (tArray[i] > tArray[len-1]) continue;
// 		SWAP(i, st);
//     SWAPINDEX(i,st);
// 		st++;
// 	}
// 	SWAP(len-1, st);
//   SWAPINDEX(len-1,st);
//   if(k < st){
//     qselect(tArray, index,st, k);
//   }
//   else if(k > st){
//     qselect(tArray + st, index + st, len - st, k - st);
//   }
//   return NULL;
// 	//return k == st	? tArray[st] : st > k	? qselect(tArray, st, k) : qselect(tArray + st, len - st, k - st);
// }

knnresult kNN(double * X , double * Y , int n , int m , int d , int k){

  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;

  distIdx p;

  //X: n * d
  //Y: m * d
  double * distance;
  int *indeces;
  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  int counter = 0;

  distance = (double *) malloc((n*m)*sizeof(double));

  indeces= (int*)malloc(m * n  *sizeof(int));

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++)
    *(indeces+i*n+j)=j;
  }

  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);

  double * xRow = (double *) calloc(n,sizeof(double));
  double * yRow = (double *) calloc(m,sizeof(double));

  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++){
      xRow[i] += (*(X+i*d+j)) * (*(X+i*d+j));
    }
  }
  for(int i=0; i<m; i++){
    for(int j=0; j<d; j++){
      yRow[i] += (*(Y+i*d+j)) * (*(Y+i*d+j));
    }
  }

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(distance + i*m + j) += xRow[i] + yRow[j];
      *(distance + i*m + j) = sqrt( *(distance + i*m + j) );
    }
  }
  free(xRow);
  free(yRow);

  // calculate transpose matrix
  double * transD = (double *) malloc(m*d*sizeof(double));
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i ) = *(distance + i*m + j );
    }
  }
  // distance = transD then delete transD
  for(int i=0; i<n*m; i++) {
    *(distance+i) = *(transD+i);
  }
  free(transD);

  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(distance+i*n+j);
      *(tempIdx+j)= *(indeces+i*n+j);
    }

    for(int j=0; j<k; j++){
      p = qselect(temp,tempIdx,n,j);
      *(final+i*k+j) = p.distance;
      *(finalIdx+i*k+j) = p.index;
    }
  }


  double * transD1 = (double *) malloc(m*k*sizeof(double));
  int * transD2 = (int *) malloc(m*k*sizeof(int));
  for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      *(transD1 + j*m + i ) = *(final + i*k + j );
      *(transD2 + j*m + i ) = *(finalIdx + i*k + j );
    }
  }

  result.ndist = transD1;
  result.nidx = transD2;

  return result;
}


// double *distance(double * X, double * Y, int n, int m, int d) {
//
//   double *result = (double *)malloc(n*m*sizeof(double));
//
//   for(int i=0; i<m; i++) {
//     for(int j=0; j<n; j++) {
//       for(int k=0; k<d; k++) {
//         *(result + j*m + i) += (*(X + j*d + k) - *(Y + i*d + k)) * (*(X + j*d + k) - *(Y + i*d + k));
//       }
//     }
//   }
//   return result;
// }

double * generatePoints(int n, int d){

  srand(time(NULL));
  double * res = (double *) malloc(n * d * sizeof(double));

  for(int i=0; i<n; i++){
    for(int j=0; j<d; j++)
      *(res+i*d+j) = (double)rand()/ RAND_MAX;
  }
  return res;
}

/* #define VERBOSE */

static char * STR_CORRECT_WRONG[] = {"WRONG", "CORRECT"};

// =================
// === UTILITIES ===
// =================

double dist(double *X, double *Y, int i, int j, int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist += ( X[i*d+l] - Y[j*d+l] ) * ( X[i*d+l] - Y[j*d+l] );
  }

  return sqrt(dist);
}

// kNNresult distrAllkNN(double *X , int n , int d , int k ){
// int * idx =(int *)malloc(*PROS*n*k*sizeof(int));
// double * dist = (double *) malloc(PROS*n * k * sizeof(double))
// knnresult result = kNN(X,X,n,n,k,d);
// result.m=n;
// result.k=k;
// idx = result.nidx;
// dist = result.ndist;
// int numtasks , taskid ;
// double *buffer = (double *) malloc(n * d * sizeof(double));
// buffer = X;
// //MPI_Init(&argc , &argv);
// //mpi init prin apo thn klisi ths sunarthshsh
// MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
// MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
// for(int i=0; i<PROS-1; i++){
// MPI_Send(buffer , n*d , MPI_DOUBLE, taskid + 1 , 0 , MPI_COMM_WORLD );
// Mpi_Recv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
// result = kNN(X,buffer , n ,n , k ,d );
// for(int l=((i+1)*n*k; l<((i+2)*n*k); l++){
//   for(int j=0; j<d; j++){
//     *(dist + l*d + j) = *(result.ndist + l*d + j);
//     *(idx + l*d + j) =  *(result.nidx + l*d + j) ;
//   }
// }
// }
// distIdx p;
// double * final = (double *) malloc(m*k * sizeof(double));
// int * finalIdx = (int *) malloc (m * k * sizeof(int));
// double * temp = (double *) malloc(n * sizeof(double));
// int * tempIdx = (int *) malloc (n * sizeof(int));
// for(int i=0; i<m; i++){
//   for(int j=0; j<n; j++){
//     *(temp+j) = *(distance+i*n+j);
//     *(tempIdx+j)= *(indeces+i*n+j);
//   }
//   for(int j=0; j<k; j++){
//     p = qselect(temp,tempIdx,n,j);
//     *(final+i*k+j) = p.distance;
//     *(finalIdx+i*k+j) = p.index;
//   }
// }
// result.ndist=final;
// result.nidx =finalIdx;
//
// return result;
//
// }

// ==================
// === VALIDATION ===
// ==================

//! kNN validator
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResult( knnresult knnres, double * corpus, double * query,
                    int n, int m, int d, int k ) {

  /* loop through all query points */
  for (int j = 0; j < m; j++ ){

    /* max distance so far (equal to kth neighbor after nested loop) */
    double maxDist = -1;

    /* mark all distances as not computed */
    int * visited = (int *) calloc( n, sizeof(int) );

    /* go over reported k nearest neighbors */
    for (int i = 0; i < k; i++ ){

      /* keep list of visited neighbors */
      visited[ knnres.nidx[i*m + j] ] = 1;

      /* get distance to stored index */
      double distxy = dist( corpus, query, knnres.nidx[i*m + j], j, d, n, m );

      /* make sure reported distance is correct */
      if ( abs( knnres.ndist[i*m + j] - distxy ) > 1e-8 ) return 0;

      /* distances should be non-decreasing */
      if ( knnres.ndist[i*m + j] < maxDist ) return 0;

      /* update max neighbor distance */
      maxDist = knnres.ndist[i*m + j];

    } /* for (k) -- reported nearest neighbors */

    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        /* get distance to unvisited vertex */
        double distxy = dist( corpus, query, i, j, d, n, m );

        /* point cannot be closer than kth distance */
        if ( distxy < maxDist ) return 0;

      } /* if (!visited[i]) */

    } /* for (i) -- unvisited notes */

    /* deallocate memory */
    free( visited );

  } /* for (j) -- query points */

  /* return */
  return 1;

}




int main()
{

  int n=7;                    // corpus
  int m=6;                    // query
  int d=3;                      // dimensions
  int k=4;                     // # neighbors
  int i,j;

  double  * corpus = (double * ) malloc( n*d * sizeof(double) );
  double  * query  = (double * ) malloc( m*d * sizeof(double) );

  for (int i=0;i<n*d;i++)
    corpus[i]= ((double)(rand()%100))/50;

  for (int i=0;i<m*d;i++)
    query[i]= ((double)(rand()%100))/50;

  knnresult knnres = kNN( corpus, query, n,m,d,k);

  printf ("\n Matrix Corpus: \n");
for (i=0; i<n; i++) {
  for (j=0; j<d; j++) {
    printf ("%10.2lf", *(corpus+j+i*d));
  }
  printf ("\n");
}

printf ("\n Matrix Query: \n");
for (i=0; i<m; i++) {
  for (j=0; j<d; j++) {
    printf ("%10.2lf", *(query+i*d+j));
  }
  printf ("\n");
}
//
// printf("\n\n");
printf ("\n Matrix DISTANCE: \n");
for (i=0; i<k; i++) {
  for (j=0; j<m; j++) {
    printf ("%lf    ", *(knnres.ndist+j+i*m));
  }
  printf ("\n");
}

printf ("\n Matrix INDECES: \n");
for (i=0; i<k; i++) {
  for (j=0; j<m; j++) {
    printf ("%10.2d", *(knnres.nidx+j+i*m));
  }
  printf ("\n");
}

  int isValid = validateResult( knnres, corpus, query, n, m, d, k );

  printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValid]);

  free( corpus );
  free( query );

  return 0;

}
