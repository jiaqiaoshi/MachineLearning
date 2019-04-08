#include <iostream>
#include <stdio.h>
#include <cmath>
#include "mat.h"
#include "rand.h"

double sigmoid (double x){
    	return 1.0/(1.0+exp(-1*x));
}

int main(){
	initRand(); // to initiate the two weight matrixs V and W

	int step, stride, hidden; // #step is the number of steps using a stride of #stride.
	scanf("%d", &step);  // The first #step values are the input and the last value is the expected output
    scanf("%d", &stride);
    scanf("%d", &hidden);

    Matrix Temp("Temp"); // store all following inputs and targets
    Temp.read();  // eg. 100 * 1 for testsine
    //Temp.normalizeCols();
    Matrix minMax = Temp.normalizeCols();
    //minMax.print();
    //Temp.print(); 
 
    // Matrix seriesSampleCol(int col, int numsteps, int stride)
    Matrix Real("Time_Series_Matrix");
    Real = Temp.seriesSampleCol(0,step,stride); // the Real matrix with structured inputs/features and targets (eg. testsine 98 * 3)
    //Real.print(); //task 1- print the normalized time series matrix you will train on
    Real.printfmt("Sampled Normalized Input:");

    int rows=Real.numRows(); // get the number of samples/data points

    Matrix X(rows, step+1, "Inputs"); // store only the inputs + bias
    X.constant(-1);
    X.insert(Real.extract(0,0,0,step),0,0);
    //X.print();

    Matrix T(rows,1,"Targets");
    T.constant(0);
    T.insert(Real.extract(0,step,0,0),0,0);
    //T.print();

    Matrix Y(rows,1,"My_Results");
    Matrix Nsamples(rows,1);
    Nsamples.constant(rows);
    
    Matrix V(step+1, hidden, "First_Weights"); // 3 * 4
    V.rand(-0.5,0.5);
    Matrix W(hidden+1,1, "Second_Weights"); // 4 * 1
    W.rand(-0.5,0.5);

    Matrix Hidden(rows, hidden, "Hidden without bias");
    Matrix Hidden_b(rows, hidden+1, "Hidden with bias");
    Hidden_b.constant(-1);

    Matrix W_delta("W_delta");
    Matrix Hb_delta("Hb_delta");
    Matrix H_delta("H_delta");

    Matrix Y1,Hidden_b1,temp_wdelta;

    Matrix update_W (hidden+1,1,"update_W");
    update_W.constant(0);

    Matrix update_V (step+1, hidden,"update_V");
    update_V.constant(0);
    

    // training
    double eta=0.15;
    for(int i=0; i<2000; i++){
    	// (* FORWARD *)
    	Hidden = (X.dot(V)).map(sigmoid); // still need to apply sigmiod function in the hidden layer

    	Hidden_b.insert(Hidden,0,0);
    	Y = Hidden_b.dot(W); // Remove the sigmoid function in the FINAL layer
    	//Y.print();
    	//   (* backprop phase *)
        Y1 = Y;
        // the first delta back to the hidden layer is just the simple difference between your prediction and the target. 
        W_delta = (Y1.sub(T)).div(Nsamples);  // This gives a simple signed value into the backprop, dy = (y - t) ******* dy = (y - t) y (1 - y);
        //W_delta.print();
        // the second dalta
        Hidden_b1 = Hidden_b;
        Hidden_b1.scalarPreSub(1); // (1 - hb)
        temp_wdelta=W_delta.dotT(W); //(dy . Transpose[w]);
        Hb_delta=(Hidden_b1.mul(Hidden_b)).mul(temp_wdelta); //dhb = hb (1 - hb) (dy . Transpose[w]);
        //Hb_delta.print();
        //(* UPDATE *)
        double momentum=0.9; //W & V updated not just by the newly computed but add in momentum factor times the last update you did
        // updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
        update_W = ((Hidden_b.Tdot(W_delta)).scalarMul(eta)).add(update_W.scalarMul(momentum));
        //update_W.print();
        W.sub(update_W);     //w -= eta*(Transpose[hb] . dy); // self.weights1 -= updatew1

        //updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
        H_delta=Hb_delta.extract(0,0,rows,hidden); // dh = Map[Drop[#, -1] &, dhb];
        update_V = ((X.Tdot(H_delta)).scalarMul(eta)).add(update_V.scalarMul(momentum));
        //update_V.print();
        V.sub(update_V);    //v -= eta*(Transpose[xb] . dh);  // self.weights2 -= updatew2
    }
    //V.print();
    //W.print();
    
   // testing
    Hidden = (X.dot(V)).map(sigmoid);
    Hidden_b.insert(Hidden,0,0);
    Y = Hidden_b.dot(W); 
    //Y.print();

  /*
    Matrix Y_T("Predicted_Target");
    Y_T = Y.joinRight(T); // not change Y and T
    Y_T.print(); // before unnormalize
   */
    //minMax.extract(0,step,0,0).print();
    Matrix T_back = T;
    T_back.unnormalizeCols(minMax); //undo the normalization done by normalizeCols using the minMax matrix
   
    Matrix Y_back = Y;
    Y_back.unnormalizeCols(minMax);
    
    Matrix Y_T("Predicted_Target");
    Y_T=Y_back.joinRight(T_back); // not change Y and T
    //Y_T.print(); // task 2-print a 2 column matrix with your prediction followed by the target. Don't forget to unnormalize.
    Y_T.printfmt("Est. and Target");

    Matrix Temp_Y = Y_back;
    double sum = Temp_Y.dist2(T_back);
    printf("Dist: %8.4lf ", sum);
    printf("\n");

	return 0;
}
