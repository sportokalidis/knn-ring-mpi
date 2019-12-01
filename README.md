# Knn-Ring-MPI

#### Run Commands
###### Make all libs and executables
```shell
$ make all 
```
###### Make only libs
```shell
$ make lib
```

###### Run Sequential
```shell
$ ./src/knnring_sequential
```

###### Run Synchronous and Asynchronous
```shell
$ mpirun -np <number of processes> ./src/knnring_mpi_syc
$ mpirun -np <number of processes> ./src/knnring_mpi_asyc
```
