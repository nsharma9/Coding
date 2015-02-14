/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  02/12/2015 07:51:10 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikhil Sharma (ns), nsharma9@ncsu.edu
 *   Organization:  NCSU
 *
 * =====================================================================================
 */
#include"ReadFile.hpp"
#include"convolution.hpp"

int main(int argc, char* argv[])
{
	int a_x, a_y, h_x, h_y;
	Mat image, kernel, output; 
	ReadFile* rd = new ReadFile(argv[1]);
	rd->printArrays();
	rd->getImageSize(image.size_x, image.size_y);
	rd->getKernelSize(kernel.size_x, kernel.size_y);
	image.mat = rd->getImage();
	kernel.mat = rd->getKernel();
	convolve2D_CPU(image, kernel, output);
	cout << "A_size: x=" << image.size_x << ", y=" << image.size_y << endl;
	cout << "H_size: x=" << kernel.size_x << ", y=" << kernel.size_y << endl;
	cout << "output:" << endl;
	rd->printArr(output.mat, output.size_x, output.size_y);
	delete rd;
	return 0;
}