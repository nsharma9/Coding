/*
 * =====================================================================================
 *
 *       Filename:  convolution.hpp
 *
 *    Description:  convolution function declarations
 *
 *        Version:  1.0
 *        Created:  02/12/2015 10:15:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikhil Sharma (ns), nsharma9@ncsu.edu
 *   Organization:  NCSU
 *
 * =====================================================================================
 */
#include<cuda.h>
#include<cuda_runtime.h>

struct _Mat {
	float** mat;
	int size_x;
	int size_y;
};
typedef _Mat Mat;

void convolve2D_CPU(const Mat&, const Mat&, Mat&);
void convolve2D_GPU(const Mat&, const Mat&, Mat&);
