/*
 * =====================================================================================
 *
 *       Filename:  convolution_GPU.cu
 *
 *    Description: Host and device code for CUDA 
 *
 *        Version:  1.0
 *        Created:  02/14/2015 03:56:01 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Nikhil Sharma (ns), nsharma9@ncsu.edu
 *   Organization:  NCSU
 *
 * =====================================================================================
 */
#include "ReadFile.hpp"
#include "convolution.hpp"
#include <stdio.h>
//Kernel Code
//
__global__ void conv2(float *d_image, int i_size_x, int i_size_y, float *d_kernel, int k_size_x, int k_size_y, float *d_output, int o_size_x, int o_size_y, size_t pitch_id, size_t pitch_kd)
{
	int o_x = blockDim.x * blockIdx.x + threadIdx.x;
	int o_y = blockDim.y * blockIdx.y + threadIdx.y;

	int index = pitch_id/sizeof(float);
	if(o_x < i_size_x && o_y < i_size_y)
	{
		d_output[o_x * index + o_y] = d_image[o_x * index + o_y] + 5;
	}
/*	float sum = 0.0f;
	for(int k_x=0; k_x < k_size_x; k_x++) {
		for(int k_y=0; k_y < k_size_y; k_y++) {
			int t_x = o_x - k_x;
			int t_y = o_y - k_y;
			if(!(t_x < 0 || t_x >= i_size_x || t_y < 0 || t_y >= i_size_y)) {
				sum+=d_image[t_x * i_size_y + t_y] * d_kernel[k_x * k_size_y + k_y];
			}
		}
	}
	d_output[threadIdx.x] = sum;*/
}	

void printMatrix(float *mat, int x, int y)
{
	for(int i = 0; i < x; i++) {
		for(int j = 0; j < y; j++)
			cout << mat[i * y + j] << "\t";
		cout << endl;
	}
}

//Host Code (xx.cpp and xx.cu):
//
void convolve2D_GPU(const Mat& image, const Mat& kernel, Mat& output)
{
	//	Initialize/acquire device (GPU)
	//dim3 numBlocks(image.size_x/kernel.size_x, image.size_y/kernel.size_y);
	//dim3 threadsPerBlock(kernel.size_x, kernel.size_y);
	output.size_x = image.size_x + kernel.size_x - 1;
	output.size_y = image.size_y + kernel.size_y - 1;
	dim3 numBlocks(output.size_x/16, output.size_y/16);
	dim3 threadsPerBlock(16, 16);

	float *h_output, *d_output, *h_image, *d_image, *h_kernel, *d_kernel;
	size_t i_width = image.size_y * sizeof(float);
	size_t i_height = image.size_x;
	size_t k_width = kernel.size_y * sizeof(float);
	size_t k_height = kernel.size_x;
	size_t o_width = output.size_y * sizeof(float);
	size_t o_height = output.size_x;

	h_image = new float[image.size_x * image.size_y];
	for (int i = 0; i < image.size_x; ++i)
		for (int j=0; j < image.size_y; ++j)
			h_image[i*image.size_y+j] = image.mat[i][j];			
	
	cout << "\nImage check" << endl;
	printMatrix(h_image, image.size_x, image.size_y);

	h_kernel = new float[kernel.size_x * kernel.size_y];
	for (int i = 0; i < kernel.size_x; ++i)
		for (int j=0; j < kernel.size_y; ++j)
			h_kernel[i*kernel.size_y+j] = kernel.mat[i][j];	

	cout << "\nkernel check" << endl;
	printMatrix(h_kernel, kernel.size_x, kernel.size_y);

	h_output = new float[output.size_x * output.size_y];

	size_t pitch_ih = image.size_y * sizeof(float);
	size_t pitch_kh = kernel.size_y * sizeof(float);

	//	Allocate memory on GPU
	size_t pitch_id;
	size_t pitch_kd;
	cudaMallocPitch(&d_image, &pitch_id, i_width, i_height);
	cudaMallocPitch(&d_kernel, &pitch_kd, k_width, k_height);
	cudaMalloc(&d_output, o_width * o_height);

	//	Copy data from host to GPU
	cudaMemcpy2D(d_image, pitch_id, h_image, pitch_ih, i_width, i_height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_kernel, pitch_kd, h_kernel, pitch_kh, k_width, k_height, cudaMemcpyHostToDevice);

	//	Execute kernel on GPU
	conv2<<<numBlocks, threadsPerBlock>>>(d_image, image.size_x, image.size_y , d_kernel, kernel.size_x, kernel.size_y, d_output, output.size_x, output.size_y, pitch_id, pitch_kd);

	//	Copy data from GPU to host
	cudaMemcpy(h_output, d_output, o_width * o_height, cudaMemcpyDeviceToHost);	

	float** out = new float*[output.size_x]; 
	for(int i=0; i < output.size_x; i++)
		out[i] = new float[output.size_y];

	for (int i = 0; i < image.size_x; ++i)
		for (int j=0; j < image.size_y; ++j)
			out[i][j] = h_output[i*image.size_y+j];
	output.mat = out;	

	cout << "\nh_output check" << endl;
	printMatrix(h_output, output.size_x, output.size_y);

	//	Deallocate memory on GPU
	delete[] h_image;
	delete[] h_kernel;
	delete[] h_output;
	cudaFree(d_image);
	cudaFree(d_kernel);
	cudaFree(d_output);	
}
