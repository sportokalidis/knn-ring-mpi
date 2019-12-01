# Knn-Ring-MPI
---

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
$ mpirun -np <number of processes> ./src/knnring_mpi_synchronous
$ mpirun -np <number of processes> ./src/knnring_mpi_asynchronous
```
#### Dependencies
In order to run the project, you need to download and install OpenBlas library. See this [link](https://www.openblas.net)

---
#### Authors
**Papadakis Charalabos** <br/> 
**Portokalidis Stavros** <br/>

---
