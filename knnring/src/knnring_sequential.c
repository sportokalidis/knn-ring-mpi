#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"


void * qselect(double *tArray,int *index, int len, int k) {
	#	define SWAP(a, b) { tmp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
	int i, st;
	double tmp;
	// double * tArray = (double * ) malloc(len * sizeof(double));
	// for(int i=0; i<len; i++){
	// 	tArray[i] = v[i];
	// }
	for (st = i = 0; i < len - 1; i++) {
		if (tArray[i] > tArray[len-1]) continue;
		SWAP(i, st);
    SWAPINDEX(i,st);
		st++;
	}
	SWAP(len-1, st);
  SWAPINDEX(len-1,st);
  if(k < st){
    qselect(tArray, index,st, k);
  }
  else if(k > st){
    qselect(tArray + st, index + st, len - st, k - st);
  }
  return NULL;
	//return k == st	? tArray[st] : st > k	? qselect(tArray, st, k) : qselect(tArray + st, len - st, k - st);
}

knnresult kNN(double * X , double * Y , int n , int m, int d , int k){

  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;

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
  //free(xRow);
  //free(yRow);

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
  //free(transD);

  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(distance+i*n+j);
      *(tempIdx+j)=*(indeces+i*n+j);
    }


    // for(int j=0; j<k; j++){
    //   *(final+i*k+j) = qselect(temp, n, j);
    // }
    qselect(temp,tempIdx, n, k);
    for(int j=0; j<k; j++){
      *(final+i*k+j) = temp[j];
      *(finalIdx+i*k+j) = tempIdx[j];
    }
  }

  result.ndist = final;
  result.nidx = finalIdx;

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

// int main (int argc , char *argv[]) {
//
//   int n=3,m=2,k=2,d=2,i,j;
//   int counter=0;
//   knnresult ofi;
//
//   double * A = (double *) malloc(n*d*sizeof(double));
//   double * B = (double *) malloc(m*d*sizeof(double));
//   double * C = (double *) malloc(m*n*sizeof(double));
//
//   for(i=0; i<n; i++){
//     for(j=0; j<d; j++){
//       *(A+i*d+j) = (double)(rand()%10);
//     }
//   }
//
//   counter=0;
//
//   for(i=0; i<m; i++){
//     for(j=0; j<d; j++){
//       *(B+i*d+j) = (double)(rand()%10);
//     }
//   }
//   ofi = kNN(A , B ,  n ,  d , k ,  m);
//
//   printf ("\n Matrix A: \n");
//   for (i=0; i<n; i++) {
//     for (j=0; j<d; j++) {
//       printf ("%10.2lf", *(A+j+i*d));
//     }
//     printf ("\n");
//   }
//
//   printf ("\n Matrix B: \n");
//   for (i=0; i<m; i++) {
//     for (j=0; j<d; j++) {
//       printf ("%10.2lf", *(B+i*d+j));
//     }
//     printf ("\n");
//   }
//   //
//   // printf("\n\n");
//   printf ("\n Matrix DISTANCE: \n");
//   for (i=0; i<m; i++) {
//     for (j=0; j<k; j++) {
//       printf ("%10.2lf", *(ofi.ndist+j+i*k));
//     }
//     printf ("\n");
//   }
//
//   printf ("\n Matrix INDECES: \n");
//   for (i=0; i<m; i++) {
//     for (j=0; j<k; j++) {
//       printf ("%10.2d", *(ofi.nidx+j+i*k));
//     }
//     printf ("\n");
//   }
//
//   printf("\n\n");
//
//   free(A);
//   free(B);
//   free(C);
//
//   return 0;
// }
