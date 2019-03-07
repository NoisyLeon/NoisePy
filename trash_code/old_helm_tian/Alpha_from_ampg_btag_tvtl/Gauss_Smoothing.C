#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

#define NSTA 2000

int calc_dist(double lati1, double long1, double lati2, double long2, double *dist);

int gau_fit_line ( double *datax, double *datay, int idata, double slpmn, double slpmx, double alpha, double *itcptout, double *weitout )
{
  FILE *f;
  int i,j,ii;
  double weit, weight;
  double slp,slpstp,itcpt,itcptmx,itcptmn,itcptstp,slpfit,itcptfit,slpp[100],itcptt[100],rsd[100],dataxx,datayy,datamin,rsdmn,rsdfit;

  for(i=0;i<idata;i++) {
     datamin=datax[i]; ii=i;
     for(j=i+1;j<idata;j++)
        if(datamin>datax[j]){
           datamin=datax[j];
           ii=j;
        }
     if(ii==i) continue;
     dataxx=datax[i];
     datayy=datay[i];
     datax[i]=datax[ii];
     datay[i]=datay[ii];
     datax[ii]=dataxx;
     datay[ii]=datayy;
  }

  itcptmx=-999999999.;itcptmn=999999999.;
  for(i=0;i<idata;i++){
     itcpt=datay[i]-datax[i]*slpmx;
     if(itcptmn>itcpt)itcptmn=itcpt;
     itcpt=datay[i]-datax[i]*slpmn;
     if(itcptmx<itcpt)itcptmx=itcpt;
    }


  for(;;){
//    printf("slpmin: %f  slpmax: %f\n",slpmn,slpmx);
    slpstp=(slpmx-slpmn)/9;
    itcptstp=(itcptmx-itcptmn)/9;

    weight=0;
    for(ii=0;ii<idata;ii++) weight += exp(-alpha*pow(datax[ii],2)); 
    rsdmn=999999999.;
    for(i=0;i<10;i++){
       slp=slpmn+i*slpstp;
       for(j=0;j<10;j++){
          itcpt=itcptmn+j*itcptstp;
          rsd[i*10+j]=0; //weight=0;
          slpp[i*10+j]=slp;
          itcptt[i*10+j]=itcpt;
          for(ii=0;ii<idata;ii++){
             weit=exp(-alpha*pow(datax[ii],2)); //weight+=weit;
             rsd[i*10+j] += pow((datay[ii]-itcpt-datax[ii]*slp),2)*weit;
             //rsd[i*10+j]+=(slp*datax[ii]+itcpt-datay[ii])*(slp*datax[ii]+itcpt-datay[ii]);
            }
          //rsd[i*10+j]=sqrt(rsd[i*10+j]/(idata-1));
          rsd[i*10+j] = sqrt(rsd[i*10+j]);///(weight-1);
         }
      }
    for(i=0;i<5;i++)
       for(j=i;j<100;j++)
          if(rsd[i]>rsd[j]){
             rsdfit=rsd[i];
             slpfit=slpp[i];
             itcptfit=itcptt[i];
             rsd[i]=rsd[j];
             slpp[i]=slpp[j];
             itcptt[i]=itcptt[j];
             rsd[j]=rsdfit;
             slpp[j]=slpfit;
             itcptt[j]=itcptfit;
            }

    slpmn=999999999;slpmx=-999999999;
    itcptmn=999999999;itcptmx=-999999999;
    for(i=0;i<5;i++){
//printf("residue: %f\n",rsd[i]);
       if(slpmn>slpp[i])slpmn=slpp[i];
       if(slpmx<slpp[i])slpmx=slpp[i];
       if(itcptmn>itcptt[i])itcptmn=itcptt[i];
       if(itcptmx<itcptt[i])itcptmx=itcptt[i];
      }
    if((slpmn-slpmx)*(slpmn-slpmx)<slpmn*slpmn/100000)break;
    //cout<<slpmx<<" "<<slpmn<<endl;
   }
//  printf("slope: %f  intercept: %f  residue: %f\n",slpp[0],itcptt[0],rsd[0]);
//  printf("%f %f %f\n",slpp[0],itcptt[0],rsd[0]);
//  *slpout=slpp[0]; *rsdout=rsd[0]; 
  *itcptout=itcptt[0]; *weitout=weight;
  return 1;
}

int Gauss_Smoothing(char *fname, double hdis)
{
   FILE *f1;
   int i,j,nsta,ndata;
   char buff[300];
   double lon[NSTA], lat[NSTA], dat[NSTA];
   double dis, itcpt, dist[NSTA/4], data[NSTA/4];
   double alpha=0.5/hdis/hdis;
   if((f1=fopen(fname,"r")) == NULL) {
      cout<<"Cannot open file "<<fname<<endl;
      return 0;
   }
   for(nsta=0;;nsta++){
      if(fgets(buff, 300, f1) == NULL) break;
      sscanf(buff,"%lf %lf %lf", &lon[nsta], &lat[nsta], &dat[nsta]);
   }
   fclose(f1);

   double dsmd[nsta], weit[nsta], mean, slpmn, slpmx;

   for(i=0;i<nsta;i++){
      ndata=0; mean=0;
      for(j=0;j<nsta;j++){
         calc_dist(lat[i], lon[i], lat[j], lon[j], &dis);
         if(dis>2*hdis) continue;
         dist[ndata]=dis; data[ndata]=dat[j];
         mean += dat[j];
         ndata++;
      }
      mean = fabs(mean/ndata);
      slpmx = 0.05*mean; slpmn = -slpmx;
      //f1=fopen("temp","w");
      //fprintf(f1,"%lf %lf\n",slpmn, slpmx);
      //for(j=0;j<ndata;j++) fprintf(f1,"%lf %lf\n",dist[j], data[j]);
      //fclose(f1);
      if(ndata<3) {dsmd[i]=dat[i]; weit[i]=1;}
      else gau_fit_line(&dist[0], &data[0], ndata, slpmn, slpmx, alpha, &dsmd[i], &weit[i]);
      //dsmd[i]=itcpt;
   }

   sprintf(buff,"%s_smd\0",fname);
   f1=fopen(buff,"w");
   for(i=0;i<nsta;i++)
      fprintf(f1,"%lf %lf %lf %lf\n", lon[i], lat[i], dsmd[i], weit[i]);
   fclose(f1);

   return 1;
}

