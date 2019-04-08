#include <iostream>
#include <stdio.h>
#include <cmath>
#include "mat.h"
#include "rand.h"

double sigmoid (double x){
    	return 1.0/(1.0+exp(-1*x));
    }

//double step(double x) {return (x<0.5 ? 0.0 : 1.0);}
int main(){
	initRand();
	int features, hiddenodes, sorts;
    scanf("%d", &features);
    scanf("%d", &hiddenodes);
    scanf("%d", &sorts);
    Matrix Temp; // store all inputs/features and following targets/outputs
    Temp.read();
    //Temp.print();
    Temp.normalizeCols(); // scale your input to be between 0 and 1 in each column
    //Temp.print();
    int rows=Temp.numRows(); // number of samples
    int total_cols=Temp.numCols(); // total of collums (inputs + targets)
    int targets=total_cols-features;

    Matrix X(rows, features+1, "Trainging_Inputs"); // store only the inputs + bias
    X.constant(-1);
    X.insert(Temp.extract(0,0,0,features),0,0);
    //X.print();

    Matrix V(features+1, hiddenodes, "First_Weights");
    V.rand(-0.2,0.2);
    Matrix W(hiddenodes+1,targets, " Second_Weights");
    W.rand(-0.2,0.2);

    Matrix Hidden(rows, hiddenodes, "Hidden without bias");
    Matrix Hidden_bias(rows, hiddenodes+1, "Hidden with bias");
    Hidden_bias.constant(-1);

    Matrix T(rows,targets,"Targets");
    Matrix Y(rows,targets,"My_Results");
    T.constant(0);
    T.insert(Temp.extract(0,features,0,0),0,0);

    Matrix W_delta,Hb_delta;
    Matrix tempy1,tempy2,temphb,temp_w,H_delta;

    // training
    double eta=0.15;
    for(int i=0; i<20000; i++){
    	// (* FORWARD *)
    	Hidden=(X.dot(V)).map(sigmoid);
    	Hidden_bias.insert(Hidden,0,0);
    	Y=(Hidden_bias.dot(W)).map(sigmoid);
        //   (* BACKWARD *)
        tempy1=Y;
        tempy1.sub(T);
        tempy2=Y;
        tempy2.scalarPreSub(1);
        W_delta=(tempy1.mul(Y)).mul(tempy2);   //dy = (y - t) y (1 - y);
        
        temphb=Hidden_bias;     
        temphb.scalarPreSub(1);
        temp_w=W_delta.dotT(W);
        Hb_delta=(temphb.mul(Hidden_bias)).mul(temp_w); //dhb = hb (1 - hb) (dy . Transpose[w]);
        //(* UPDATE *)
         W.sub((Hidden_bias.Tdot(W_delta)).scalarMul(eta));     //w -= eta*(Transpose[hb] . dy);
         
         H_delta=Hb_delta.extract(0,0,rows,hiddenodes); // dh = Map[Drop[#, -1] &, dhb];
         V.sub((X.Tdot(H_delta)).scalarMul(eta));    //v -= eta*(Transpose[xb] . dh);
    }
    // testing
    Matrix NX;
    NX.read();
    //NX.print();
    int Nrows=NX.numRows(); // newIrisData2----such as :30
    int Ntotal_cols=NX.numCols(); //such as 5
    int Noutputs=Ntotal_cols-features; // then : 5-4=1

    Matrix temp_NX=NX;
    NX.normalizeCols(); 
    //NX.print();

    Matrix NT(Nrows,Noutputs,"Test_Targets");
    NT.constant(0);
    NT.insert(temp_NX.extract(0,features,0,0),0,0);
   // NT.print();

    Matrix X_bias(Nrows,features+1, "Test_Inputs");
    X_bias.constant(-1);
    X_bias.insert(NX.extract(0,0,0,features),0,0);
    //X_bias.print();

    Matrix H(Nrows, hiddenodes, "Test_Hidden without bias");
    H=(X_bias.dot(V)).map(sigmoid);

    Matrix Hb(Nrows, hiddenodes+1, "Test_Hidden with bias");
    Hb.constant(-1);
    Hb.insert(H,0,0);

    Matrix NY(Nrows,Noutputs,"Test_Outputs");
    NY=(Hb.dot(W)).map(sigmoid);
    //Y_Pre.map(step);
    printf("Target\n");
    for(int i=0; i< Nrows; i++){
    	NT.writeLine(i);
    	printf("\n");
    }
 
    double m=0;
    double index;

    Matrix Pre(Nrows,1,"Predicted_Label");
    Pre.constant(0);

    printf("Predicted\n");
    for (int i=0; i < Nrows; i++){
        m = -100000;
        for(int j=0; j<targets; j++){
            if (NY.get(i,j)>m){
                m=NY.get(i,j);
                index=j;
            }
        }
        Pre.set(i,0,index);
        Pre.writeLine(i);
        printf("\n");
        //printf("%8.4f\n",index);
     }
    
    Matrix CM(sorts,sorts," Confusion_Matrix");
    CM.constant(0);
    for(int r=0; r<Nrows;r++){
        const int c=0;
        if(NT.get(r,c)==0 && Pre.get(r,c)==0) CM.inc(0,0);
        else if(NT.get(r,c)==1 && Pre.get(r,c)==1) CM.inc(1,1);
        else if(NT.get(r,c)==2 && Pre.get(r,c)==2) CM.inc(2,2);
        else if(NT.get(r,c)==0 && Pre.get(r,c)==1) CM.inc(1,0);
        else if(NT.get(r,c)==0 && Pre.get(r,c)==2) CM.inc(2,0);  //Columns represent actual values and rows predicted values.
        else if(NT.get(r,c)==1 && Pre.get(r,c)==0) CM.inc(0,1);
        else if(NT.get(r,c)==1 && Pre.get(r,c)==2) CM.inc(2,1);
        else if(NT.get(r,c)==2 && Pre.get(r,c)==0) CM.inc(0,2);
        else if(NT.get(r,c)==2 && Pre.get(r,c)==1) CM.inc(1,2);
    }

    printf("Confusion Matrix\n");
    for(int i=0; i< sorts; i++){
        CM.writeLine(i);
         printf("\n");
    }
    return 0;
}