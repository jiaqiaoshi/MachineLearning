#include <iostream>
#include <stdio.h>
#include "mat.h"
#include "rand.h"

int main (){

	int inputs;
	scanf("%d",&inputs);
    
	Matrix temp; // to read matrix
	temp.read();  // the row/col number will be first read
	//x.print();
	int rows=temp.numRows();
	int cols=temp.numCols();  // including inputs/features collums and outputs/target collums
    int outputs = cols-inputs;  //seperate target collums

    Matrix X(temp.numRows(), inputs+1,"inputs"); //sample data-features
    X.constant(0); // initial to o
    //X.print();
    //temp.extract(0,0,0,inputs).print(), this is the input part of matrix temp
    X.insert(temp.extract(0,0,0,inputs),0,0); //insert/cover the input part in X
    //s.print();

    Matrix bias(rows,1,"00");
	bias.constant(-1);
	//bias.print();
	X.insert(bias,0,inputs); //insert bias into the last collums of X
	//X.print(); // X now is complete training inputs
  
	Matrix T(rows,outputs,"targets"); // target matrix
	T.constant(0); // initial to o
	T.insert(temp.extract(0,inputs,0,0),0,0); //insert/cover the output part of x in T 

	Matrix Y(rows,outputs,"my_results"); // results by out algorithm

	Matrix W(inputs+1,outputs,"weights");  // weight matrix
	initRand(); //init required before use!!!
	W.constant(randUnit()); //initial bwtween [0,1)
	//W.print();


	int looping=0; // number of looping
	double eta=0.15; // learning rate
	Matrix _T;

	// training 
	do{
		Y=X.dot(W); //dot product

		for (int i=0;i<rows;i++){
			for(int j=0; j<outputs;j++){
				if (Y.get(i,j)>=0.5)
					Y.set(i,j,1);
				else
					Y.set(i,j,0);
			}
		}
		if (!Y.equal(T)){
			_T=T; // as don't want to modify origal Target Matrix
			W.add(X.Tdot(_T.sub(Y)).scalarMul(eta));
		}
		else
			break;
		looping++;
	}while(looping<30000);//2000-10000-15000-20000
	//Y.print(); //print matrix and its name
	//W.print();

	//testing
	printf("BEGIN TESTING\n");
	temp.read(); // new matrix will be read in x,  also the row/col number will be read first
	rows=temp.numRows(); // maybe different with above, but number of inputs and
	//cols=x.numCols(); now the cols = inputs
    //x.print();

	X.insert(temp.extract(0,0,0,inputs),0,0);
	X.insert(bias,0,inputs);
    //s.print();
    
	Y=X.dot(W);
	for (int i=0;i<rows;i++){
		for(int j=0; j<outputs;j++){
			if (Y.get(i,j)>0.5)
				Y.set(i,j,1);
			else
				Y.set(i,j,0);
		}
	}
	//Y.print();
	for (int i=0;i<rows;i++){
		for(int j=0; j<inputs;j++){
			printf("%10.2lf", temp.get(i,j));
		}
		for(int k=0;k<outputs;k++){
			printf("%10.2lf", Y.get(i,k));
		}
			
		printf("\n");
	}

	return 0;
}
