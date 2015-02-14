/*
 * =====================================================================================
 *
 *       Filename:  ReadFile.hpp
 *
 *    Description:  Head for all classes used in convolution assignment
 *
 *        Version:  1.0
 *        Created:  02/12/2015 07:39:17 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikhil Sharma (ns), nsharma9@ncsu.edu
 *   Organization:  NCSU
 *
 * =====================================================================================
 */
#ifndef READFILE_H
#define READFILE_H

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>

using namespace std;
/*
 * =====================================================================================
 *        Class:  ReadFile
 *  Description:  Read input file and create the kernel and image matrices
 * =====================================================================================
 */
class ReadFile {
	private:
		float** a;
		float** h;
		ifstream infile;
		stringstream stream;
		int size_ax, size_ay, size_hx, size_hy;
		
		ReadFile() {}
		void read();
		void getMatrices();
		float** makeArray(int, int);

	public:
		~ReadFile();
		ReadFile(char*);
		void printArrays();
		void getImageSize(int&, int&);
		void getKernelSize(int&, int&);
		float** getImage(){return a;}
		float** getKernel(){return h;}
		void printArr(float**, int, int);
};

#endif
