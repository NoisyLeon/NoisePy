#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
using namespace std;

int sort ( double *datax, double *datay1, double *datay2, double *datay3, double *datay4, double *datay5, double *datay6, double *datay7, double *datay8, int ndat )
{
  int i, j, ii;
  double datamin, dataxx, datayy;

  for(i=0;i<ndat;i++) {
     datamin=datax[i]; ii=i;
     for(j=i+1;j<ndat;j++)
        if(datamin>datax[j]){
           datamin=datax[j];
           ii=j;
        }
     if(ii==i) continue;
     dataxx=datax[i];
     datax[i]=datax[ii];
     datax[ii]=dataxx;
     datayy=datay1[i];
     datay1[i]=datay1[ii];
     datay1[ii]=datayy;
     datayy=datay2[i];
     datay2[i]=datay2[ii];
     datay2[ii]=datayy;
     datayy=datay3[i];
     datay3[i]=datay3[ii];
     datay3[ii]=datayy;
     datayy=datay4[i];
     datay4[i]=datay4[ii];
     datay4[ii]=datayy;
     datayy=datay5[i];
     datay5[i]=datay5[ii];
     datay5[ii]=datayy;
     datayy=datay6[i];
     datay6[i]=datay6[ii];
     datay6[ii]=datayy;
     datayy=datay7[i];
     datay7[i]=datay7[ii];
     datay7[ii]=datayy;
     datayy=datay8[i];
     datay8[i]=datay8[ii];
     datay8[ii]=datayy;
  }

  return 1;
}

int sort2 ( double *datax, double *datay1, int ndat )
{
  int i, j, ii;
  double datamin, dataxx, datayy;

  for(i=0;i<ndat;i++) {
     datamin=datax[i]; ii=i;
     for(j=i+1;j<ndat;j++)
        if(datamin>datax[j]){
           datamin=datax[j];
           ii=j;
        }
     if(ii==i) continue;
     dataxx=datax[i];
     datax[i]=datax[ii];
     datax[ii]=dataxx;
     datayy=datay1[i];
     datay1[i]=datay1[ii];
     datay1[ii]=datayy;
  }

  return 1;
}

int fit_line ( double *dataxo, double *datayo, int idatao, int indv, double *slpout, double *itcptout, double *sdout )
{
  int i,j,ii,ndat;
  double datx[idatao], daty[idatao], std;

  ndat=0;
  for(i=0;i<idatao;i++) {
     if(datayo[i]==-1) continue;
     datx[ndat] = dataxo[i];
     daty[ndat] = datayo[i];
     ndat++;
  }

  double a, b, X=0, Y=0, X2=0, Y2=0, XY=0;
   for(i=0;i<ndat;i++) {
      X += datx[i];
      Y += daty[i];
      X2 += pow(datx[i],2);
      Y2 += pow(daty[i],2);
      XY += datx[i]*daty[i];
   }
   if(indv==0) {
      a = (ndat*XY-X*Y)/(ndat*X2-X*X);
      b = (-X*XY+X2*Y)/(ndat*X2-X*X);
   }
   else if(indv==1) {
      a=(ndat*Y2-Y*Y)/(ndat*XY-X*Y);
      b=(Y*XY-Y2*X)/(ndat*XY-X*Y);
   }
   else {
      cout<<"Line_fit: Wrong input for indep var, stopped!"<<endl;
      exit(0);
   }
   std=0;
   for(i=0;i<ndat;i++) std += pow(daty[i]-a*datx[i]-b,2);
   std = sqrt(std/(ndat-1));
   if(indv==1) std /= a*a;

  *slpout=a; *itcptout=b; *sdout=std;
  return 1;
}
