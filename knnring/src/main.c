#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"

int main (int argc , char *argv[]) {

  int n=3,m=2,k=2,d=2,i,j;
  int counter=0;
  knnresult ofi;

  double * A = (double *) malloc(n*d*sizeof(double));
  double * B = (double *) malloc(m*d*sizeof(double));
  double * C = (double *) malloc(m*n*sizeof(double));

  for(i=0; i<n; i++){
    for(j=0; j<d; j++){
      *(A+i*d+j) = (double)(rand()%10);
    }
  }

  counter=0;

  for(i=0; i<m; i++){
    for(j=0; j<d; j++){
      *(B+i*d+j) = (double)(rand()%10);
    }
  }
  ofi = kNN(A , B ,  n, m, d, k);

  printf ("\n Matrix A: \n");
  for (i=0; i<n; i++) {
    for (j=0; j<d; j++) {
      printf ("%10.2lf", *(A+j+i*d));
    }
    printf ("\n");
  }

  printf ("\n Matrix B: \n");
  for (i=0; i<m; i++) {
    for (j=0; j<d; j++) {
      printf ("%10.2lf", *(B+i*d+j));
    }
    printf ("\n");
  }
  //
  // printf("\n\n");
  printf ("\n Matrix DISTANCE: \n");
  for (i=0; i<m; i++) {
    for (j=0; j<k; j++) {
      printf ("%10.2lf", *(ofi.ndist+j+i*k));
    }
    printf ("\n");
  }

  printf ("\n Matrix INDECES: \n");
  for (i=0; i<m; i++) {
    for (j=0; j<k; j++) {
      printf ("%10.2d", *(ofi.nidx+j+i*k));
    }
    printf ("\n");
  }

  printf("\n\n");

  free(A);
  free(B);
  free(C);

  return 0;
}
