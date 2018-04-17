#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;


int calc_dist(double lati1, double long1, double lati2, double long2, double *dist)
{
  int i,flag=0;
  double latio1,longo1,dlt_lati,dlt_long;
  double Ra,Rb,R,Rx,f;
  double U1,U2;
  double Ds,alpha2;
  double ctr_angl;
  double pi;
  double cv,cv1,cv2,cv3,cv4,cv5,cvC,numda1;
  double numda;
  double mius, cvA, cvB, deltacv;

  pi=4.0*atan(1.0);
  Ra = 6378.137; Rb = 6356.7523142;
  f = 1/298.257223563;
begin:
  long1=long1-(int)floor(long1)/360*360;
  if(long1<0) long1+=360;
  long2=long2-(int)floor(long2)/360*360;
  if(long2<0) long2+=360;
  if(lati1==-lati2 && fabs(long1-long2)==180){
  *dist=20003.917357;
  return 1;
  }
  dlt_lati=fabs(lati2-lati1);
  dlt_long=long2-long1;

if (dlt_long > 180.000)  dlt_long = 360.000000 - dlt_long;
if (dlt_long < -180.000) dlt_long = 360.000 - fabs(dlt_long);
dlt_long = fabs(dlt_long);

U1 = atan((1-f)*tan(lati1/180*pi));
U2 = atan((1-f)*tan(lati2/180*pi));
dlt_long = dlt_long*pi/180;

numda = dlt_long;
numda1 = numda;
i=0;
do {
i++;
numda = numda1;
cv1 =  sqrt( (cos(U2)*sin(numda))*(cos(U2)*sin(numda))+ (cos(U1)*sin(U2)-sin(U1)*cos(U2)*cos(numda))*(cos(U1)*sin(U2)-sin(U1)*cos(U2)*cos(numda)) );
cv2 = sin(U1)*sin(U2)+ cos(U1)*cos(U2)*cos(numda);
cv = atan2(cv1,cv2);
if(cv==0)cv+=0.00000000001;
cv3 = cos(U1)*cos(U2)*sin(numda)/sin(cv);
cv4 = 1 - cv3*cv3;
if(cv4==0)cv4+=0.00000000001;
cv5 = cos(cv) - 2*sin(U1)*sin(U2)/cv4;
cvC = f/16*cv4*(4 + f*(4 - 3*cv4));
numda1 = dlt_long + (1-cvC)*f*cv3*(cv + cvC*cv1*(cv5 + cvC*cv2*(-1 +2*cv5*cv5)));
if(i>50){
flag=1;
latio1=lati1*pi/180.;
Rx=sqrt((pow(Ra*Ra*cos(latio1),2)+pow(Rb*Rb*sin(latio1),2))/(pow(Ra*cos(latio1),2)+pow(Rb*sin(latio1),2)));
Rx=pi*(Rx+Ra)/2.;
latio1=lati1;
longo1=long1;
lati1=lati2;
long1=long2;
lati2=-latio1;
long2=longo1+180;
goto begin;
}
} while (fabs(numda - numda1) > 1e-10);
mius = cv4*(Ra*Ra - Rb*Rb)/(Rb*Rb);
cvA = 1+mius/16384*(4096 + mius*(-768 + mius*(320 - 175*mius)));
cvB = mius/1024*(256+ mius*(-128 + mius*(74 - 47*mius)));
deltacv = cvB*cv1*(cv5 +cvB/4*(cv2*(-1 + 2*cv5*cv5)-cvB/6*cv5*(-3+4*cv1*cv1)*(-3+4*cv5*cv5) ));
*dist = Rb * cvA *(cv - deltacv);

if(flag==1){
alpha2=atan2(cos(U1)*sin(dlt_long), -sin(U1)*cos(U2) + cos(U1)*sin(U2)*cos(dlt_long))*180/pi;
if( fabs(long2-long1)>180 ) { alpha2 = 360-alpha2;}
if( long2 < long1 ) { alpha2 = 360-alpha2; }
if(alpha2>180) alpha2=360-alpha2;
alpha2=fabs(90-alpha2);
Ds=Rx*(90-alpha2)/90.+(Ra+Rb)*pi*alpha2/180.;
*dist=Ds-*dist;
}

return 1;
}


int calc_azimuth(double lati1, double long1, double lati2, double long2, double *alpha1)
{
  double dlt_lati,dlt_long;
  double Ra,Rb,f;
  double U1,U2;
  double pi;

  pi=4.0*atan(1.0);
  Ra = 6378.137; Rb = 6356.7523142;
  f = 1/298.257223563;
  long1=long1-(int)floor(long1)/360*360;
  if(long1<0) long1+=360;
  long2=long2-(int)floor(long2)/360*360;
  if(long2<0) long2+=360;
  if(lati1==-lati2 && fabs(long1-long2)==180){
  *alpha1=999;
  return 1;
  }
  dlt_lati=fabs(lati2-lati1);
  dlt_long=long2-long1;

if (dlt_long > 180.000)  dlt_long = 360.000000 - dlt_long;
if (dlt_long < -180.000) dlt_long = 360.000 - fabs(dlt_long);
dlt_long = fabs(dlt_long);

U1 = atan((1-f)*tan(lati1/180*pi));
U2 = atan((1-f)*tan(lati2/180*pi));
dlt_long = dlt_long*pi/180;

*alpha1=atan2(cos(U2)*sin(dlt_long), cos(U1)*sin(U2) - sin(U1)*cos(U2)*cos(dlt_long))*180/pi;
//*alpha2=atan2(cos(U1)*sin(dlt_long), -sin(U1)*cos(U2) + cos(U1)*sin(U2)*cos(dlt_long))*180/pi;
if( fabs(long2-long1)>180 ) { *alpha1 = 360-*alpha1; }
if( long2 < long1 ) { *alpha1 = 360-*alpha1; }

return 1;
}

int Check_azi_cov(double *lon, double *lat, int nsta, double mdis, double *flag)
{
   int i, j, jj, k, ndata, ntmp;
   double azi[300], azimin, azitmp, dist;
   double mazi=110.;

   for(i=0;i<nsta;i++){
      if(flag[i]==-1) continue;
      ndata=0;
      for(j=0;j<nsta;j++){
         if(j==i || flag[j]==-1) continue;
         calc_dist(lat[i], lon[i], lat[j], lon[j], &dist);
         if(dist>mdis) continue;
         calc_azimuth(lat[i], lon[i], lat[j], lon[j], &azi[ndata]);
         ndata++;
      }
      //cout<<"Check azi cov: "<<ndata<<" nearby stations"<<endl;
      for(j=0;j<ndata;j++) {
         azimin=azi[j]; jj=j;
         for(k=j+1;k<ndata;k++)
            if(azimin>azi[k]) {
              azimin=azi[k];
              jj=k;
            }
         if(jj==j) continue;
         azitmp=azi[j];
         azi[j]=azi[jj];
         azi[jj]=azitmp;
      }
      if(azi[0]+360.-azi[ndata-1]>mazi) {
         flag[i]=-2;
         continue;
      }
      for(j=1;j<ndata;j++)
         if(azi[j]-azi[j-1]>mazi) {
            flag[i]=-2;
            break;
         }
   }
   for(i=0;i<nsta;i++) if(flag[i]==-2) flag[i]=-1;
   return 1;
}

