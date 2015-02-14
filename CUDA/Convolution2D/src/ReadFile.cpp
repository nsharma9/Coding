#include "ReadFile.hpp"

ReadFile::ReadFile(char * _infile) 
{
	infile.open(_infile);
	read();
}

ReadFile::~ReadFile() 
{
	infile.close();
	for (int i = 0; i < size_ax; i++)
		delete [] a[i];
	delete [] a;
	for (int i = 0; i < size_hx; i++)
		delete [] h[i];
	delete [] h;
	a = NULL;
	h = NULL;
}

void ReadFile::read() 
{
	int ax=0,ay=0,hx=0,hy=0;
	string line;
	getMatrices();
	infile.clear();
	infile.seekg(0);
	stream.clear();
	if(infile.is_open()){
		float i;
		while(getline(infile,line) && !line.empty())
		{
			stream << line;
			while (stream >> i)
				a[ax][ay++] = i;
			ax++; ay = 0;
			stream.clear();
		}
		while(getline(infile,line))
		{
			stream << line;
			while (stream >> i)
				h[hx][hy++] = i;
			hx++; hy = 0;
			stream.clear();
		}
	}
}

void ReadFile::getMatrices() 
{
	string line;
	int flag_a=0, flag_h=0;
	if(infile.is_open()){
		infile.clear();
		infile.seekg(0);
		stream.clear();
		while(getline(infile,line) && !line.empty())
		{
			float i;
			size_ax++;
			stream << line;
			if(!flag_a) {
				flag_a=1;
				while (stream >> i)
					size_ay++;
			}
		}
		stream.clear();
		while(getline(infile,line))
		{
			float i;
			size_hx++;
			stream << line;
			if(!flag_h) {
				flag_h=1;
				while (stream >> i)
					size_hy++;
			}
		}
	}
	a = makeArray(size_ax, size_ay);
	h = makeArray(size_hx, size_hy);
}

float * * ReadFile::makeArray(int row, int col) 
{
	float** arr = new float*[row];
	for(int i=0; i < row; i++)
		arr[i] = new float[col];
	return arr;
}

void ReadFile::printArr(float **arr , int row, int col) 
{
	for(int i=0; i < row; i++)
	{
		for(int j=0; j < col; j++)
			cout << arr[i][j] << "\t";
		cout << endl;
	}	
}

void ReadFile::printArrays() 
{
	cout << "Image:" << endl;
	printArr(a, size_ax, size_ay);
	cout << "Kernel:" << endl;
	printArr(h, size_hx, size_hy);
}

void ReadFile::getImageSize(int& x, int& y) 
{
	x = size_ax;
	y = size_ay;	
}

void ReadFile::getKernelSize(int& x, int& y) 
{
	x = size_hx;
	y = size_hy;
}

