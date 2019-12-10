CC = gcc
MPICC = mpicc
CFLAGS = -Wall -O3 -std=c99 -g
SDIR=./src
LDIR=./lib
IDIR=./inc
EXE = $(SDIR)/knnring_sequential $(SDIR)/knnring_mpi $(SDIR)/knnring_mpi_synchronous  $(SDIR)/knnring_mpi_asynchronous
LIBS = $(LDIR)/knnring_sequential.a $(LDIR)/knnring_mpi.a $(LDIR)/knnring_mpi_synchronous.a $(LDIR)/knnring_mpi_asynchronous.a
MAIN = tester
LDFLAGS = -lopenblas -lm
INCLUDES = -I$(IDIR) -L/opt/OpenBLAS/lib/

all: $(EXE)
lib: $(LIBS)



$(SDIR)/knnring_sequential: $(SDIR)/$(MAIN).c $(LDIR)/knnring_sequential.a
	$(CC) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(SDIR)/knnring_mpi: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi.a
	$(MPICC) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(SDIR)/knnring_mpi_synchronous: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi_synchronous.a
	$(MPICC) $(INCLUDES) -o $@ $^ $(LDFLAGS)

$(SDIR)/knnring_mpi_asynchronous: $(SDIR)/$(MAIN)_mpi.c $(LDIR)/knnring_mpi_asynchronous.a
	$(MPICC) $(INCLUDES) -o $@ $^ $(LDFLAGS)




$(LDIR)/%.a: $(SDIR)/%.o $(SDIR)/utilities.o
	ar rcs $@ $^




$(SDIR)/knnring_sequential.o: $(SDIR)/knnring_sequential.c
	$(CC) $(INCLUDES) $(CFLAGS) -o $@ -c $<

$(SDIR)/knnring_mpi.o: $(SDIR)/knnring_mpi.c
	$(MPICC) $(INCLUDES) $(CFLAGS) -o $@ -c $<

$(SDIR)/knnring_mpi_synchronous.o: $(SDIR)/knnring_mpi_synchronous.c
	$(MPICC) $(INCLUDES) $(CFLAGS) -o $@ -c $<

$(SDIR)/knnring_mpi_asynchronous.o: $(SDIR)/knnring_mpi_asynchronous.c
	$(MPICC) $(INCLUDES) $(CFLAGS) -o $@ -c $<

$(SDIR)/utilities.o: $(SDIR)/utilities.c
	$(MPICC) $(INCLUDES) $(CFLAGS) -o $@ -c $<



clean:
	rm -f $(SDIR)/*.o $(EXE) $(LIBS)
