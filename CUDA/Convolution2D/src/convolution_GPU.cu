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
#include "convolution.hpp"

//Kernel Code
//
__global__ void conv2(float *d_image, int i_x, int i_y, float *d_kernel, int k_x, int k_y, float *d_output, int o_x, int o_y, size_t pitch_id, size_t pitch_kd)
{
		
}

//Host Code (xx.cpp and xx.cu):
//
void convolve2D_GPU(const Mat& image, const Mat& kernel, Mat& output)
{
	//	Initialize/acquire device (GPU)
	output.size_x = image.size_x + kernel.size_x - 1;
	output.size_y = image.size_y + kernel.size_y - 1;

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

	h_kernel = new float[kernel.size_x * kernel.size_y];
	for (int i = 0; i < kernel.size_x; ++i)
		for (int j=0; j < kernel.size_y; ++j)
			h_kernel[i*kernel.size_y+j] = kernel.mat[i][j];	

	h_output = new float[output.size_x * output.size_y];	

	//	Allocate memory on GPU
	size_t pitch_id;
	size_t pitch_kd;
	cudaMallocPitch(&d_image, &pitch_id, i_width, i_height);
	cudaMallocPitch(&d_kernel, &pitch_kd, k_width, k_height);
	cudaMalloc(&d_output, o_width * o_height);

	//	Copy data from host to GPU
	cudaMemcpy2D(d_image, pitch_id, h_image, i_width, i_height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_kernel, pitch_kd, h_kernel, k_width, k_height, cudaMemcpyHostToDevice);

	//	Execute kernel on GPU
	conv2<<<>>>(d_image, image.size_x, image.size_y , d_kernel, kernel.size_x, kernel.size_y, d_output, output.size_x, output.size_y, pitch_id, pitch_kd);
	cudaDeviceSynchronize();

	//	Copy data from GPU to host
	cudaMemcpy(h_output, d_output, o_width * o_height, cudaMemcpyDeviceToHost);	

	float** out = new float*[output.size_x]; 
	for(int i=0; i < output.size_x; i++)
		out[i] = new float[output.size_y];

	for (int i = 0; i < image.size_x; ++i)
		for (int j=0; j < image.size_y; ++j)
			out[i][j] = h_output[i*image.size_y+j];
	output.mat = out;	

	//	Deallocate memory on GPU
	delete[] h_image;
	delete[] h_kernel;
	delete[] h_output;
	cudaFree(d_image);
	cudaFree(d_kernel);
	cudaFree(d_output);	
}
