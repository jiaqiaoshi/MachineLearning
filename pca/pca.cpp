#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "mat.h"
#include "rand.h"
using namespace std;

int main(int argc, char* argv[]){
    
	int k = atoi(argv[1]); //  command line argument, the number of eigenvectors you want to keep eg. 10
	//cout<<k<<endl;
	// step 1: get the original picture data matric
	bool isColor;
    Matrix Pic_org;
	Pic_org.readImagePixmap("","Image",isColor);

	if(k<0){
		Pic_org=Pic_org.transpose(); //If command line argument is a negative integer then transpose the matrix before the doing the PCA
	}

    Pic_org.printSize(); //task 1--(size of Pic: 540 X 1464)

	//cout<<boolalpha<<"The input is color pic : "<<isColor<< endl; // isColor will be True if input img file is ppm format
	
    // step 2: center the columns of data about the mean of each column
    Matrix Mean("Mean");
    Mean=Pic_org.meanVec();
    Mean.printSize(); // tesk 2--(size of Mean: 1 X 1464)
    /*
    Matrix Stddev("Stddev");
    Stddev=Pic_org.stddevVec();*/

    Matrix Pic_centered;
    Pic_centered=Pic_org;
    Pic_centered.subRowVector(Mean); // subtract Mean row matrix to each row
    //Pic_centered.divRowVector(Stddev); // divide Stddev row matrix to each row------to be confirmed
    // step 3: compute the eigenvalues and eigenvectors
    // get the covariance matrix first
    Matrix Cov("Covariance_Matrix");
    Cov=Pic_centered.cov();
    
    Matrix Eigenvec("EigenVectors");
    Eigenvec = Cov;
    Matrix Eigenval("EigenValues");
    // Destroys self by replacing self with EIGENVECTORS in rows.
    // Returns a new matrix (a row vector) with the EIGENVALUES in it.
    // Eigenvalues and vectors returned sorted from largest magnitude to smallest
    Eigenval=Eigenvec.eigenSystem(); // already normalized & sorted
    Eigenvec.printSize(); // tesk 3
    Eigenval.printSize(); // tesk 4

    // step 4- Encode/translate/Compress the centered pic data
    // 4-1)first need to take the k largest from eigenvalues and eigenvectors
    Matrix K_Eigenvec;
    K_Eigenvec=Eigenvec.extract(0,0,abs(k),0); 
    // 4-2) reduce the dimension using the reduced set of eigenvectors X_new=X*K_Eigenvec.transpose
    Matrix Pic_compressed("Encoded");
    Pic_compressed=Pic_centered.dotT(K_Eigenvec);
    Pic_compressed.printSize(); //tesk 5

    // step 5-Decode/Recover pic data from compressed one
    Matrix Pic_back("Decoded");
    // first: rotate the data back by multiplying the reduced eigenvector matrix
    Pic_back=Pic_compressed.dot(K_Eigenvec);
    // second: move the data back from where it was centered to its old position
   // Pic_back.mulRowVector(Stddev).addRowVector(Mean);
    Pic_back.addRowVector(Mean);
    Pic_back.printSize(); //tesk 6

    // step 6-compute the disctance between the original pic data and the recovered pic data
    double dist;
    dist= Pic_org.dist2(Pic_back); // get the matrix distance
    dist /= (Pic_org.numCols() * Pic_org.numRows());// get the average per pixel distance, so divide the matrix distance by the total number of elements/pixels in the data matrix
    printf("Per Pixel Dist^2: %g\n",dist);

    // step 7- write the recovered pic to either a ppm or pgm file called "z.ppm" or "z.pgm"
    if(k<0){
    	Pic_back=Pic_back.transpose();
    }

    if (isColor == true){
    	Pic_back.writeImagePpm("z.ppm","");
    }
    else{
    	Pic_back.writeImagePgm("z.pgm","");
    }

	return 0;
}
