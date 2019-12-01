#ifndef UTILITIES_H
#define UTILITIES_H



void swapElement(double **one, double  **two);


double SumRow(double *array, int numOfColumns, int row);


void qselect(double *tArray,int *index, int len, int k);


void quicksort(double *array, int *idx, int first, int last);


knnresult updateResult(knnresult result,knnresult tempResult,int offset,int newOff);



#endif
