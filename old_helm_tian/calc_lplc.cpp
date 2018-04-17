#include "lplc.h"
#include <iostream>
#include <fstream>
#include <string>

//int calc_lplc(char *file, double *slong, double *slati, double *grdt, double *azi, double *lplc, int *nsta, double tension, double blc_size)

int main() {
	std::string fname = "travel_time_SS.phase.c.txt";
	double *slong = new double[50000];
	double *slati = new double[50000];
	double *grdt = new double[50000];
	double *theta = new double[50000];
	double *lplc = new double[50000];
	int nsta;
	calc_lplc(fname.c_str(), slong, slati, grdt, theta, lplc, &nsta, 0.2, 10);
	std::ofstream fout(fname+"_lplc");
	for(int i=0; i<nsta; i++)
		fout<<slong[i]<<" "<<slati[i]<<" "<<lplc[i]<<" "<<grdt[i]<<" "<<theta[i]<<"\n";
	return 0;
}
