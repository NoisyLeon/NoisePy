#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

int Gauss_Smoothing(double *lon, double *lat, double *dat, double *densc, int nsta, double hdis);

int calc_dist(double lati1, double long1, double lati2, double long2, double *dist);

int calc_azimuth(double lati1, double long1, double lati2, double long2, double *alpha1);

int Check_azi_cov(double *lon, double *lat, int nsta, double mdis, double *flag)
{
   int i, j, jj, k, ndata;
   double azi[300], azimin, azitmp, dist;

   for(i=0;i<nsta;i++){
      flag[i]=0; ndata=0;
      for(j=0;j<nsta;j++){
         if(j==i) continue;
         calc_dist(lat[i], lon[i], lat[j], lon[j], &dist);
         if(dist>mdis) continue;
         calc_azimuth(lat[i], lon[i], lat[j], lon[j], &azi[ndata]);
         ndata++;
      }
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
      if(azi[0]+360.-azi[ndata-1]>90) {
         flag[i]=-1;
         continue;
      }
      for(j=1;j<ndata;j++)
         if(azi[j]-azi[j-1]>90) {
            flag[i]=-1;
            break;
         }
   }
   return 1;
}

int calc_smth_grdt(char *file, double *slong, double *slati, double *sdat, double hdis, double *grdt, double *azi, int *nsta)
{
   FILE *ff;
   char buff[300], buff2[300];
   float longmin=360, longmax=0, latimin=90, latimax=-90, longtmp, latitmp;
   double radius=6371.1391285, pi=4.0*atan(1.0), dgr=0.05, dattmp;
   double grdtx, grdty, dis, alpha, weight;
   int i, j, ii, jj, ista, nrand, ntmpp;

   if((ff=fopen(file,"r"))==NULL){
     printf("Can't open %s to read!\n",file);
     return 0;
   }
   for(*nsta=0;;){
      if(fgets(buff, 300, ff) == NULL ) break;
      if((sscanf(buff,"%lf %lf %lf", &slong[*nsta], &slati[*nsta], &sdat[*nsta]))!=3) {
         cout<<"Wrong format! Stopped: "<<buff<<endl;
         break;
      }
      if(longmin>slong[*nsta]) longmin=slong[*nsta];
      if(longmax<slong[*nsta]) longmax=slong[*nsta];
      if(latimin>slati[*nsta]) latimin=slati[*nsta];
      if(latimax<slati[*nsta]) latimax=slati[*nsta];
      *nsta=*nsta+1;
   }
   fclose(ff);
   
   //double hdis=0.;
   double densc[*nsta];
   Gauss_Smoothing(&slong[0], &slati[0], &sdat[0], &densc[0], *nsta, hdis);
   Check_azi_cov(&slong[0], &slati[0], *nsta, 200, &azi[0]);

   sprintf(buff, "%s_smd", file);
   ff=fopen(buff,"w");
   for(i=0;i<*nsta;i++) { 
      //if(densc[i]<3) azi[i] = -1;
      //else azi[i] = 0;
      fprintf(ff,"%lf %lf %lf\n",slong[i], slati[i], sdat[i]);
   }
   fclose(ff);

   longmin=floor(longmin/dgr)*dgr-dgr;
   latimin=floor(latimin/dgr)*dgr-dgr;
   longmax=ceil(longmax/dgr)*dgr+dgr;
   latimax=ceil(latimax/dgr)*dgr+dgr;
   int npts_long=int((longmax-longmin)/dgr+1);
   int npts_lati=int((latimax-latimin)/dgr+1);
   double dislong[npts_lati], dislati, distmp;
   double gradtr, grdtmp, gradx[npts_long][npts_lati], grady[npts_long][npts_lati], dat[npts_long][npts_lati];
   for(i=0;i<npts_long;i++) for(j=0;j<npts_lati;j++){
      gradx[i][j] = 1e10;
      dat[i][j] = -1;
   }
   for(i=1;i<npts_lati-1;i++){
      dislati=atan(0.993277*tan((latimin+i*dgr)/180*pi))*180/pi;
      dislong[i]=radius*sin((90-dislati)/180*pi)*dgr/180*pi;
   }
   dislati=radius*dgr/180*pi;

   nrand=rand();
   sprintf(buff,"region_%d",nrand);
   ff=fopen(buff,"w");
   fprintf(ff, "-R%d/%d/%d/%d\0", (int)floor(longmin), (int)ceil(longmax), (int)floor(latimin), (int)ceil(latimax));
   fclose(ff);
   //sprintf(buff,"/home/tianye/code/Script/GMT/C_plot_travel_positive %s region_%d 0.5", file, nrand);
   //system(buff);
   sprintf(buff,"/home/tianye/code/Script/GMT/C_plot_travel_positive %s_smd region_%d %f", file, nrand, dgr);
   system(buff);
   //sprintf(buff,"mv %s.HD.HD %s.HD",file,file);
   //system(buff);
   sprintf(buff,"rm -f region_%d",nrand);
   system(buff);

   sprintf(buff,"%s_smd.HD",file);
   if((ff=fopen(buff,"r"))==NULL){
      printf("Can't open %s to read!\n",buff);
      return 0;
   }
   for(;;){
      if( fgets(buff, 300, ff) == NULL ) break;
      sscanf(buff,"%f %f %lf", &longtmp, &latitmp, &dattmp);
      i=int((longtmp-longmin)/dgr+0.1);
      j=int((latitmp-latimin)/dgr+0.1);
      if(i<0||i>=npts_long||j<0||j>=npts_lati) continue;
      dat[i][j]=dattmp;
   }
   fclose(ff);

   sprintf(buff,"rm -f %s.HD %s_smd %s_smd.HD", file, file, file);
   system(buff);

   for(ista=0;ista<*nsta;ista++){
      if(azi[ista]==-1) continue;
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradx[i][j] != 1e10) continue;
            gradx[i][j]=(dat[i+1][j]-dat[i-1][j])/2.0/dislong[j];
            grady[i][j]=(dat[i][j+1]-dat[i][j-1])/2.0/dislati;
            gradtr=sqrt(gradx[i][j]*gradx[i][j]+grady[i][j]*grady[i][j]);
            //azig[i][j]=pi/2-atan2(grdty,grdtx);
            //if(azig[i][j]<0-1e-8)azig[i][j]+=2*pi;
            distmp=sqrt(dislong[j]*dislong[j]+dislati*dislati);
            grdtx=(dat[i+1][j-1]-dat[i-1][j+1])/2.0/distmp;
            grdty=(dat[i+1][j+1]-dat[i-1][j-1])/2.0/distmp;
            alpha=2*atan(dislati/dislong[j]);
            alpha=atan((grdtx/grdty-cos(alpha))/sin(alpha));
            grdtmp=fabs(grdty/cos(alpha));
            //cout<<gradtr[i][j]<<" "<<grdtmp<<endl;
            if(fabs((gradtr-grdtmp)/gradtr)>0.15) gradx[i][j]=1e20;
            //else gradtr[i][j]=(gradtr[i][j]+grdtmp)/2.;
         }
   }

   for(ista=0;ista<*nsta;ista++){
      if(azi[ista]==-1) continue;
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      ntmpp=0; weight=0; grdtx=0; grdty=0;
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradx[i][j]>=1e10) continue;
            ntmpp++;
            dis=sqrt(pow((longmin+i*dgr-slong[ista])*dislong[j],2)+pow((latimin+j*dgr-slati[ista])*dislati,2))+1e-20;
            grdtx+=gradx[i][j]/dis;
            grdty+=grady[i][j]/dis;
            weight+=1/dis;
         }
      if(ntmpp<2) { azi[ista]=-1; continue; }
      grdt[ista]=sqrt(grdtx*grdtx+grdty*grdty)/weight;
//  if((grdt[ista]/sdat[ista])>0.005 || (grdt[ista]/sdat[ista])<-0.006 ) { azi[ista]=-1; continue; }
      azi[ista]=pi/2-atan2(grdty,grdtx);
      if(azi[ista]<0-1e-8)azi[ista]+=2*pi;
   }
   return 1;
}

