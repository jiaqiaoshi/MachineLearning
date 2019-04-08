BIN=nn

CXX=g++
TARNAME=nn
SHELL=/bin/sh

# optimization flags
CXXFLAGS=-O3 -Wall
# debugging flags
CXXFLAGS=-g -Wall
LIBS=

SRCS=\
nnoneof.cpp\
nn.cpp\
mat.cpp\
randf.cpp

HDRS=\
mat.h\
rand.h

OBJS=\
mat.o\
rand.o

EXTRAS=\
rand.cpp\
randmt.cpp

$(BIN): $(OBJS) $(BIN).o
	$(CXX)  $(CFLAGS) $(OBJS) $(BIN).o $(LIBS) -o $(BIN)

$(BIN)oneof: $(OBJS) $(BIN)oneof.o
	$(CXX)  $(CFLAGS) $(OBJS) $(BIN)oneof.o $(LIBS) -o $(BIN)oneof

clean:
	/bin/rm -f *.o a.out
