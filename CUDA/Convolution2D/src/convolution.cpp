#include "convolution.hpp"

void convolve2D_CPU(const Mat& image, const Mat& kernel, Mat& output) 
{
	float** out = new float*[image.size_x];
	for(int i=0; i < image.size_x; i++)
		out[i] = new float[image.size_y];

	// for each row and each column of image
	// sum the multiple of kernel and image surrounding
	int k_center_x = kernel.size_x >> 1;
	int k_center_y = kernel.size_y >> 1;	

	for(int i_x=0; i_x < image.size_x; i_x++) {
		for(int i_y=0; i_y < image.size_y; i_y++)
		{
			int sum=0;
			for(int k_x=0; k_x < kernel.size_x; k_x++) {
				for (int k_y=0; k_y < kernel.size_y; k_y++)
				{
					int t_x = i_x-k_center_x+k_x;
					int t_y = i_y-k_center_y+k_y;
					if(!(t_x < 0 || t_x >= image.size_x || t_y < 0 || t_y >= image.size_y)) {
//						cout << image.mat[t_x][t_y] << " * " << kernel.mat[k_x][k_y] << " + ";
						sum+=image.mat[t_x][t_y]*kernel.mat[k_x][k_y];
					}
				}
			}
//			cout << endl;
			out[i_x][i_y] = sum;
		}	
	}
	output.mat = out;
	output.size_x = image.size_x;
	output.size_y = image.size_y;
}
