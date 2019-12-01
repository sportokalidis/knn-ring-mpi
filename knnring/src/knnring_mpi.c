/*
* file:   knnring_mpi.c
* 
*
* authors: Charalabos Papadakis, Portokalidis Stavros (9334)
* emails: , stavport@ece.auth.gr
* date:   2019-12-01
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cblas.h"
#include "knnring.h"
#include <mpi.h>
#include "utilities.h"


knnresult kNN(double * X , double * Y , int n , int m , int d , int k);


knnresult distrAllkNN(double * X , int n , int d , int k ) {

  int numtasks , taskid ;
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));

  knnresult result ;
  knnresult tempResult  ;

  result.m=n;
  result.k=k;
  idx = result.nidx;
  dist = result.ndist;

  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));
  double *y = (double *)malloc(n*k*sizeof(double));
  int *yidx = (int *)malloc(n*k*sizeof(int));
  myElements = X;
  int counter= 2;
  int p1, p2, p3;
  int offset , newOff ;


  switch(taskid%2){
    case 0:
      MPI_Recv(otherElements , n*d , MPI_DOUBLE, (numtasks+taskid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      result = kNN(myElements,myElements,n,n,d,k);
      MPI_Send(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
      tempResult = kNN(otherElements , myElements , n , n , d ,k);
      offset = (numtasks+taskid-1)%numtasks;
      newOff = (numtasks+offset-1)%numtasks;
      result = updateResult( result, tempResult, offset, newOff);

      while(counter<numtasks){
        MPI_Recv(buffer , n*d , MPI_DOUBLE, (numtasks+taskid- 1)%numtasks , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
        swapElement(&otherElements, &buffer);
        //otherElements = buffer;
        newOff = (numtasks + newOff-1)%numtasks;
        tempResult = kNN( otherElements , myElements,  n ,n , d ,k );
        result = updateResult( result, tempResult, 0, newOff);
        counter++;
      }
      break;

    case 1:
      MPI_Send(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
      result = kNN(myElements,myElements,n,n,d,k);
      MPI_Recv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      tempResult = kNN(otherElements , myElements,  n , n , d ,k);
      offset = (numtasks+taskid-1)%numtasks;
      newOff = (numtasks+offset-1)%numtasks;
      result = updateResult( result, tempResult, offset, newOff);

      while(counter<numtasks){
          MPI_Send(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD );
          MPI_Recv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
          newOff = (numtasks + newOff-1)%numtasks;
          result = updateResult( result, tempResult, 0, newOff);
          counter++;
      }
      break;
  }

  double localMin=result.ndist[1];
  double localMax=result.ndist[0];
  for(int i=0; i <n*k; i++){
    if(result.ndist[i]>localMax){
      localMax = result.ndist[i];
    }
    if(result.ndist[i]<localMin && result.ndist[i]!=0){
      localMin = result.ndist[i];
    }
  }


  double globalMin;
  double globalMax;

  MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  printf("AT process  %d MAX : %lf, MIN : %lf  \n " ,taskid , globalMax , globalMin );


  free(y);
  free(yidx);

  return result;

}

knnresult kNN(double * X , double * Y , int n , int m , int d , int k) {

  knnresult result;
  result.k = k;
  result.m = m;
  result.nidx = NULL;
  result.ndist = NULL;
  int taskid, numtasks;
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);


  double * distance;
  int *indeces;
  double alpha=-2.0, beta=0.0;
  int lda=d, ldb=d, ldc=m, i, j;
  int counter = 0;

  distance = (double *) calloc(n*m,sizeof(double));
  double * transD = (double *)malloc(m*n*sizeof(double));
  indeces= (int*)malloc(m * n  *sizeof(int));
  if(distance == NULL){
    printf("distance exei thema");

  }

  if(indeces ==NULL ){
    printf("indeces exei thema ");

  }

  if(transD==NULL){
    printf("transd exei thema \n");

  }
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);



  for(int i=0; i<n; i++){
    double SumX =  SumRow(X, d, i);
    for(int j=0; j<m; j++){
      double SumY =  SumRow(Y, d, j);
      *(distance + i*m + j) += SumX + SumY;
      if(*(distance + i*m + j) < 0.00000001){
        *(distance + i*m + j) = 0;
      }
      else{
        *(distance + i*m + j) = sqrt( *(distance + i*m + j) );
      }
    }
  }

  // calculate transpose matrix

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i )   = *(distance + i*m + j );
    }
  }
  free(distance);

  double * final = (double *) malloc(m*k * sizeof(double));
  int * finalIdx = (int *) malloc (m * k * sizeof(int));
  double * temp = (double *) malloc(n * sizeof(double));
  int * tempIdx = (int *) malloc (n * sizeof(int));
  if(final==NULL){
    printf(" FINAL  EXEI THEMA");

  }
  if(finalIdx==NULL){
    printf(" finalidx  EXEI THEMA");

  }
  if(temp==NULL){
    printf(" temp EXEI THEMA");

  }
  if(tempIdx==NULL){
    printf(" tempidx  EXEI THEMA");

  }
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      *(temp+j) = *(transD+i*n+j);
      *(tempIdx+j)= *(indeces+i*n+j);
    }
    qselect(temp,tempIdx,n,k);
    quicksort(temp, tempIdx,0,k);
    for(int j=0; j<k; j++){
      *(final+i*k+j) = temp[j];
      *(finalIdx+i*k+j) = tempIdx[j];
    }
  }
  result.ndist = final;
  result.nidx = finalIdx;

  return result;
}
