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
	Mat image, kernel, output, output_GPU; 
	ReadFile* rd = new ReadFile(argv[1]);
	rd->printArrays();
	rd->getImageSize(image.size_x, image.size_y);
	rd->getKernelSize(kernel.size_x, kernel.size_y);
	image.mat = rd->getImage();
	kernel.mat = rd->getKernel();
	convolve2D_CPU(image, kernel, output);
	convolve2D_GPU(image, kernel, output_GPU);
	cout << "A_size: x=" << image.size_x << ", y=" << image.size_y << endl;
	cout << "H_size: x=" << kernel.size_x << ", y=" << kernel.size_y << endl;
	cout << "Output:" << endl;
	rd->printArr(output.mat, output.size_x, output.size_y);
	cout << "O_size: x=" << output.size_x << ", y=" << output.size_y << endl;
	cout << "Output_GPU:" << endl;
	rd->printArr(output_GPU.mat, output_GPU.size_x, output_GPU.size_y);
	cout << "O_GPU_size: x=" << output_GPU.size_x << ", y=" << output_GPU.size_y << endl;
	delete rd;
	return 0;
}
