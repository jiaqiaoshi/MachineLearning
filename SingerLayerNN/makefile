BIN=nn

# what you want to name your tar/zip file:
TARNAME=nn
CXX=g++

CXXFLAGS=-O3 -Wall   # optimize
CXXFLAGS=-g -Wall    # debug
LIBS = -lm

EXAMPLES=

EXTRAS=\
rand.cpp\
randmt.cpp

SRCS=\
$(BIN).cpp\
mat.cpp\
randf.cpp

HDRS=\
rand.h\
mat.h

OBJS=\
$(BIN).o\
mat.o\
randf.o

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $(BIN)

$(BIN)oneof: $(OBJS) $(BIN)oneof.o
	$(CXX)  $(CFLAGS) $(OBJS) $(BIN)oneof.o $(LIBS) -o $(BIN)oneof

clean:
	/bin/rm -f *.o $(BIN)*.tar *~ core gmon.out a.out

tar:
	tar -cvf $(TARNAME).tar makefile $(EXAMPLES) $(SRCS) $(HDRS) $(EXTRAS)
	ls -l $(TARNAME).tar

zip:
	zip $(TARNAME).zip makefile $(EXAMPLES) $(SRCS) $(HDRS) $(EXTRAS)
	ls -l $(TARNAME).zip
