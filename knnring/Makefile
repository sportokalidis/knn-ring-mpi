CC = gcc
LOADER = gfortran
CFLAGS = -O3 -DADD_ -std=c99
SDIR=./src
LDIR=./lib
IDIR=./inc
EXE = $(SDIR)/knnring_sequential
CLIBS = $(LDIR)/cblas_LINUX.a $(LDIR)/libblas.a
LIBS = $(LDIR)/knnring_sequential.a
INCLUDES = $(IDIR)/knnring.h
MAIN = tester


all: $(EXE)
lib: $(LIBS)



$(SDIR)/knnring_sequential: $(SDIR)/$(MAIN).c $(LDIR)/knnring_sequential.a
	$(CC) -I$(IDIR) -L$(LDIR) -o $@ $^ -lopenblas -lm



$(LDIR)/knnring_sequential.a: $(SDIR)/knnring_sequential.o
	ar rcs $@ $<



$(SDIR)/knnring_sequential.o:	$(SDIR)/knnring_sequential.c
	$(CC) -g -I$(IDIR) $(CFLAGS) -L$(LDIR) -o $@ -c $< -lm



clean:
	rm -f $(SDIR)/*.o $(EXE) $(LIBS)
