#include "convolution.hpp"

void convolve2D_CPU(const Mat& image, const Mat& kernel, Mat& output) 
{
	// set size of output matrix
	output.size_x = image.size_x + kernel.size_x - 1;
	output.size_y = image.size_y + kernel.size_y - 1;
	
	// allocate memory to matrix
	float** out = new float*[output.size_x]; 
	for(int i=0; i < output.size_x; i++)
		out[i] = new float[output.size_y];

	//convolution
	for(int o_x=0; o_x < output.size_x; o_x++) {
		for(int o_y=0; o_y < output.size_y; o_y++)
		{
			int sum=0;
			for(int k_x=0; k_x < kernel.size_x; k_x++) {
				for (int k_y=0; k_y < kernel.size_y; k_y++)
				{
					int t_x = o_x-k_x;
					int t_y = o_y-k_y;
					if(!(t_x < 0 || t_x >= image.size_x || t_y < 0 || t_y >= image.size_y)) {
						sum+=image.mat[t_x][t_y]*kernel.mat[k_x][k_y];
					}
				}
			}
			out[o_x][o_y] = sum;
		}	
	}
	output.mat = out;
}
