CC = gcc
LOADER = gfortran
CFLAGS = -O3 -DADD_
SDIR=./src
LDIR=./lib
IDIR=./inc
EXE = $(SDIR)/knnring_sequential
CLIBS = $(LDIR)/cblas_LINUX.a $(LDIR)/libblas.a
LIBS = $(LDIR)/knnring_sequential.a
INCLUDES = inc/knnring.h
MAIN = main


all: $(EXE)
lib: $(LIBS)



$(SDIR)/knnring_sequential: $(LDIR)/knnring_sequential.a $(SDIR)/$(MAIN).c $(SDIR)/knnring_sequential.c
	$(LOADER) -I$(IDIR) -o $@ $^ $(CLIBS) -lm



$(LDIR)/knnring_sequential.a: $(SDIR)/knnring_sequential.o
	ar rcs $@ $< $(INCLUDES)



$(SDIR)/knnring_sequential.o:	$(SDIR)/knnring_sequential.c
	$(CC) -g $(CFLAGS) -I$(IDIR) -o $@ -c $<



clean:
	rm -f $(SDIR)/*.o $(EXE) $(LIBS) 
