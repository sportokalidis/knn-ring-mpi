/*
* file:   utilities.c
* Implementation of project's functions 
*
* authors: Charalabos Papadakis, Portokalidis Stavros (9334)
* emails: , stavport@ece.auth.gr
* date:   2019-12-01
*/


#include <stdio.h>
#include <stdlib.h>
#include "knnring.h"
#include <math.h>


void swapElement(double **one, double  **two){
	double  *temp = *one;
	*one = *two;
	*two = temp;
}

double SumRow(double *array, int numOfColumns, int row) {
  double result=0;

  for(int j=0; j<numOfColumns; j++){
    result += pow(*(array+row*numOfColumns+j),2);
  }

  return result;
}


void qselect(double *tArray,int *index, int len, int k) {
	#	define SWAP(a, b) { tmp = tArray[a]; tArray[a] = tArray[b]; tArray[b] = tmp; }
  #	define SWAPINDEX(a, b) { tmp = index[a]; index[a] = index[b]; index[b] = tmp; }
	int i, st;
	double tmp;

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
  if (k == st){
    return ;
  }
  return ;
}



void quicksort(double *array, int *idx, int first, int last){
   int i, j, pivot;
   double  temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(array[i]<=array[pivot]&&i<last)
            i++;
         while(array[j]>array[pivot])
            j--;
         if(i<j){
            temp=array[i];
            array[i]=array[j];
            array[j]=temp;

            temp=idx[i];
            idx[i]=idx[j];
            idx[j]=temp;
         }
      }

      temp=array[pivot];
      array[pivot]=array[j];
      array[j]=temp;

      temp=idx[pivot];
      idx[pivot]=idx[j];
      idx[j]=temp;

      quicksort(array,idx,first,j-1);
      quicksort(array,idx,j+1,last);

   }
}


knnresult updateResult(knnresult result,knnresult tempResult,int offset,int newOff){
  double *y = (double *)malloc(result.m*result.k*sizeof(double));
  int *yidx = (int *)malloc(result.m*result.k*sizeof(int));

  if(y==NULL){
    printf("Y EXEI THEMA");

  }
  if(yidx==NULL){
    printf("YIDX EXEI THEMA");

  }
  int p1 , p2 , p3;
  for(int i=0; i<result.m; i++){
    p1=0, p2=0, p3=0;
    while (p3<result.k) {
        if (*(result.ndist + i*result.k + p1) < *(tempResult.ndist + i*result.k+ p2)){
          *(y+i*result.k+p3) = *(result.ndist+ i*result.k+p1);
          *(yidx+i*result.k+p3) = *(result.nidx+i*result.k+p1) + offset*result.m;
          p3++;
          p1++;
        }
        else{
          *(y+i*result.k+p3) = *(tempResult.ndist+i*result.k+p2);
          *(yidx+i*result.k+p3) = *(tempResult.nidx+i*result.k+p2) + newOff*result.m  ;
          p3++;
          p2++;
        }
    }
  }
  for(int i=0; i<result.m; i++){
    for(int j = 0 ; j <result.k ; j++){
      *(result.ndist+i*result.k+j) = *(y+i*result.k+j);
      *(result.nidx+i*result.k+j)= *(yidx+i*result.k+j);
    }
  }

  return result;
}
