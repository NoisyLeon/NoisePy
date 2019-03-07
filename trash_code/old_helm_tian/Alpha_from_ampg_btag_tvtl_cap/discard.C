#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
using namespace std;

int discard_by_std (double *dat, double *weit, int ndat, double std_coef, int *count, double *weito, double *avgo, double *stdo)
{
   int i;
   double avg, std, v1, v2;

   avg=0; v1=0; v2=0;
   for(i=0;i<ndat;i++) {
      avg += dat[i]*weit[i];
      v1 += weit[i];
      v2 += pow(weit[i],2);
   }
   avg /= v1; 
   std=0;
   for(i=0;i<ndat;i++) std += pow(dat[i]-avg,2)*weit[i];
   std = sqrt(std*v1/(v1*v1-v2));
   for(i=0;i<ndat;i++) if(fabs(dat[i]-avg)>std_coef*1.5*std) dat[i]=-1;

   avg=0; v1=0; v2=0;
   for(i=0;i<ndat;i++) {
      if(dat[i]==-1) continue;
      avg += dat[i]*weit[i];
      v1 += weit[i];
      v2 += pow(weit[i],2);
   }
   avg /= v1;
   std=0;
   for(i=0;i<ndat;i++) {
      if(dat[i]==-1) continue;
      std += pow(dat[i]-avg,2)*weit[i];
   }
   std = sqrt(std*v1/(v1*v1-v2));
   for(i=0;i<ndat;i++) if(fabs(dat[i]-avg)>std_coef*std) dat[i]=-1;

   *count = 0;
   *avgo = 0; v1 = 0; v2 = 0;
   for(i=0;i<ndat;i++) {
      if(dat[i]==-1) continue;
      *avgo += dat[i]*weit[i];
      v1 += weit[i];
      v2 += pow(weit[i],2);
      *count = *count + 1;
   }
   *weito = v1;
   *avgo = *avgo/v1;
   *stdo=0;
   for(i=0;i<ndat;i++) {
      if(dat[i]==-1) continue;
      *stdo = *stdo + pow(dat[i]-avg,2)*weit[i];
   }
   *stdo = sqrt(*stdo*v1/(v1*v1-v2));

   return 1;
}
