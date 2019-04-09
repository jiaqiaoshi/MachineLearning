BIN=pca

CXX=g++
TARNAME=pca
SHELL=/bin/sh

# optimization flags
CXXFLAGS=-O3 -Wall
# debugging flags
CXXFLAGS=-g -Wall
LIBS = -lm

SRCS=\
pca.cpp\
mat.cpp\
rand.cpp

HDRS=\
mat.h\
rand.h

OBJS=\
mat.o\
rand.o

EXTRAS=\
randf.cpp\
randmt.cpp

$(BIN): $(OBJS) $(BIN).o
	$(CXX)  $(CFLAGS) $(OBJS) $(BIN).o $(LIBS) -o $(BIN)

clean:
	/bin/rm -f *.o a.out
	
tar:
	tar -cvf $(TARNAME).tar makefile $(EXAMPLES) $(SRCS) $(HDRS) $(EXTRAS)
	ls -l $(TARNAME).tar
