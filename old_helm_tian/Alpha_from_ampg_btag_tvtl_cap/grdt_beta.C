#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

int calc_grdt_beta(char *file, double *slong, double *slati, double *sdat, double *grdt, double *azi, int *nsta, double tension)
{
   FILE *ff;
   char buff[300], buff2[300];
   float longmin=360, longmax=0, latimin=90, latimax=-90, longtmp, latitmp;
   double radius=6371.1391285, pi=4.0*atan(1.0), dgr=0.05, dattmp;
   double grdtx, grdty, dis, alpha, weight;
   int i, j, ii, jj, ista, nrand, ntmpp;

   double blc_size=1;

/*
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
*/
   for(i=0;i<*nsta;i++) {
      if(longmin>slong[i]) longmin=slong[i];
      if(longmax<slong[i]) longmax=slong[i];
      if(latimin>slati[i]) latimin=slati[i];
      if(latimax<slati[i]) latimax=slati[i];
   }

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
   sprintf(buff,"/home/tianye/code/Programs/Attenuation/Alpha_from_ampg_btag_tvtl_cap/C_plot_travel_positive %s region_%d %f %lf %lf", file, nrand, dgr, tension, blc_size);
   system(buff);
   sprintf(buff,"rm -f region_%d",nrand);
   system(buff);


   sprintf(buff,"%s.HD",file);
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

   for(ista=0;ista<*nsta;ista++){
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(dat[i+1][j]==-1. || dat[i-1][j]==-1. || dat[i][j+1]==-1. || dat[i][j-1]==-1.) continue;
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
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      ntmpp=0; weight=0; grdtx=0; grdty=0;
      sdat[ista]=0;
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradx[i][j]>=1e10) continue;
            ntmpp++;
            dis=sqrt(pow((longmin+i*dgr-slong[ista])*dislong[j],2)+pow((latimin+j*dgr-slati[ista])*dislati,2))+1e-10;
            grdtx+=gradx[i][j]/dis;
            grdty+=grady[i][j]/dis;
            sdat[ista]+=dat[i][j]/dis;
            weight+=1/dis;
         }
      if(ntmpp<2) { azi[ista]=-1; continue; }
      sdat[ista]=sdat[ista]/weight;
      grdt[ista]=sqrt(grdtx*grdtx+grdty*grdty)/weight;
      azi[ista]=pi/2-atan2(grdty,grdtx);
      if(azi[ista]<0-1e-8)azi[ista]+=2*pi;
   }
   return 1;
}

