/*
* file:   knnring_mpi_asyc.c
* Iplemantation of knnring asychronous verision
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
  MPI_Request request[3];
  MPI_Status status;


  int *idx =(int *)malloc(n*k*sizeof(int));
  double * dist = (double *) malloc(n * k * sizeof(double));
  double *buffer = (double *) malloc(n * d * sizeof(double));
  double *myElements = (double *) malloc(n * d * sizeof(double));
  double *otherElements = (double *) malloc(n * d * sizeof(double));
   if(idx==NULL){
     printf("IDX THEMA ");

   }
   if(dist==NULL){
     printf("DIST THEMA");

   }
   if(buffer==NULL){
     printf("BUFFER THEMA");

   }
   if(myElements==NULL){
     printf("MyElements THEM");

   }
   if(otherElements==NULL){
     printf("OTHER ELEMENTS THEMA");

   }


  knnresult result ;
  knnresult tempResult  ;

  result.m=n;
  result.k=k;
  idx = result.nidx;
  dist = result.ndist;


  myElements = X;

  int counter= 2;
  int p1, p2, p3;
  int newOff , offset;

  switch(taskid%2){
    case 0:
      MPI_Isend(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[0] );
      MPI_Irecv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
      result = kNN(myElements,myElements,n,n,d,k);
      offset = (numtasks+taskid-1)%numtasks;
      newOff = (numtasks + offset-1)%numtasks;
      MPI_Wait(&request[1],&status);
      while(counter<numtasks){
        MPI_Isend(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[2] );
        MPI_Irecv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);
          tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

          if(counter == 2 ){
          result = updateResult( result, tempResult, offset, newOff);
          }
          else{
            newOff = (numtasks + newOff-1)%numtasks;
            result = updateResult( result, tempResult, 0, newOff);
          }
        MPI_Wait(&request[1],&status);
        MPI_Wait(&request[2],&status);
        swapElement(&otherElements,&buffer);
          counter++;
      }

      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
      newOff = (numtasks + newOff-1)%numtasks;
      result = updateResult( result, tempResult, 0, newOff);
      break;

    case 1:
      MPI_Isend(myElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[0] );
      MPI_Irecv(otherElements , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);

      result = kNN(myElements,myElements,n,n,d,k);
      offset = (numtasks+taskid-1)%numtasks;
      newOff = (numtasks + offset-1)%numtasks;
      MPI_Wait(&request[1],&status);

      while(counter<numtasks){
        MPI_Isend(otherElements , n*d , MPI_DOUBLE, (taskid + 1)%numtasks , 0 , MPI_COMM_WORLD , &request[2] );
        MPI_Irecv(buffer , n*d , MPI_DOUBLE, taskid - 1 , 0 , MPI_COMM_WORLD, &request[1]);

        tempResult = kNN(otherElements ,  myElements, n , n , d ,k );

        if(counter == 2 ){
          result = updateResult( result, tempResult, offset, newOff);
        }
        else{
          newOff = (numtasks + newOff-1)%numtasks;
          result = updateResult( result, tempResult, 0, newOff);
        }

        MPI_Wait(&request[1],&status);
        MPI_Wait(&request[2],&status);
        swapElement(&otherElements,&buffer);
        counter++;
      }

      tempResult = kNN(otherElements ,  myElements, n , n , d ,k );
      newOff = (numtasks + newOff-1)%numtasks;
      result = updateResult( result, tempResult, 0, newOff);
      break;

  }

  MPI_Barrier(MPI_COMM_WORLD);

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

  distance = (double *) malloc((n*m)*sizeof(double));
  if(distance == NULL){
    printf("distance exei thema");

  }

  indeces= (int*)malloc(m * n  *sizeof(int));
  if(indeces ==NULL ){
    printf("indeces exei thema ");

  }
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++) {
      *(indeces+i*n+j)=j;
    }
  }

  cblas_dgemm(CblasRowMajor , CblasNoTrans , CblasTrans , n, m , d , alpha , X , lda , Y , ldb , beta, distance , ldc);

  double * xRow = (double *) calloc(n,sizeof(double));
  double * yRow = (double *) calloc(m,sizeof(double));
  double * transD = (double *) malloc(m*n*sizeof(double));

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
      if(*(distance + i*m + j) < 0.00000001){
        *(distance + i*m + j) = 0;
      }
      else{
        *(distance + i*m + j) = sqrt( *(distance + i*m + j) );
      }
    }
  }

  // calculate transpose matrix
  if(transD==NULL){
    printf("transd exei thema");

  }
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      *(transD + j*n + i ) = *(distance + i*m + j );
    }
  }

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
