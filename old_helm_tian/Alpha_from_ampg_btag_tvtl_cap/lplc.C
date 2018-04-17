#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
using namespace std;

int Check_azi_cov(double *lon, double *lat, int nsta, double mdis, double *flag);

int calc_lplc(char *file, double *slong, double *slati, double *grdt, double *azi, double *lplc, int *nsta, double tension, double blc_size)
{
   FILE *ff;
   char buff[300], buff2[300];
   float longmin=360, longmax=0, latimin=90, latimax=-90, longtmp, latitmp;
   double radius=6371.1391285, pi=4.0*atan(1.0), dgr=0.05, dattmp;
   double grdtx, grdty, dis, alpha;
   double sdat[2000], weight;
   int i, j, ii, jj, ista, nrand, ntmpp, nstl;

   if(blc_size<1) blc_size=1;

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

   for(i=0;i<*nsta;i++) azi[i] = 0;
   Check_azi_cov(&slong[0], &slati[0], *nsta, 200, &azi[0]);
   //nstl=0;
   //for(i=0;i<*nsta;i++) if(azi[i]!=-1) nstl++;
   //cout<<"lplc: found "<<nstl<<" out of "<<*nsta<<" stations with good coverage."<<endl;

   longmin=floor(longmin/dgr)*dgr-dgr;
   latimin=floor(latimin/dgr)*dgr-dgr;
   longmax=ceil(longmax/dgr)*dgr+dgr;
   latimax=ceil(latimax/dgr)*dgr+dgr;
   int npts_long=int((longmax-longmin)/dgr+1);
   int npts_lati=int((latimax-latimin)/dgr+1);
   double dislong[npts_lati], dislati, distmp;
   double gradtr[npts_long][npts_lati], grdtmp, gradx[npts_long][npts_lati], grady[npts_long][npts_lati], grdx[*nsta], grdy[*nsta], dat[npts_long][npts_lati];

   for(i=0;i<npts_long;i++) for(j=0;j<npts_lati;j++){
      gradtr[i][j] = 1e10;
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
      if(azi[ista]==-1) continue;
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradtr[i][j] != 1e10) continue;
            gradx[i][j]=(dat[i+1][j]-dat[i-1][j])/2.0/dislong[j];
            grady[i][j]=(dat[i][j+1]-dat[i][j-1])/2.0/dislati;
            gradtr[i][j]=sqrt(pow(gradx[i][j],2)+pow(grady[i][j],2));
            //azig[i][j]=pi/2-atan2(grady[i][j],gradx[i][j]);
            //if(azig[i][j]<0-1e-8)azig[i][j]+=2*pi;
            distmp=sqrt(dislong[j]*dislong[j]+dislati*dislati);
            grdtx=(dat[i+1][j-1]-dat[i-1][j+1])/2.0/distmp;
            grdty=(dat[i+1][j+1]-dat[i-1][j-1])/2.0/distmp;
            alpha=2*atan(dislati/dislong[j]);
            alpha=atan((grdtx/grdty-cos(alpha))/sin(alpha));
            grdtmp=fabs(grdty/cos(alpha));
            if(fabs((gradtr[i][j]-grdtmp)/gradtr[i][j])>0.15) gradtr[i][j]=1e20;
//            else gradtr[i][j]=(gradtr[i][j]+grdtmp)/2.;
         }
   }

   nstl=0;
   for(ista=0;ista<*nsta;ista++){
      if(azi[ista]==-1) continue;
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      ntmpp=0; weight=0; grdx[ista]=0, grdy[ista]=0, grdt[ista]=0;
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradtr[i][j]>=1e10) continue;
            ntmpp++;
            dis=sqrt(pow((longmin+i*dgr-slong[ista])*dislong[j],2)+pow((latimin+j*dgr-slati[ista])*dislati,2))+1e-20;
            grdx[ista]+=gradx[i][j]/dis;
            grdy[ista]+=grady[i][j]/dis;
            //grdt[ista]+=gradtr[i][j]/dis;
            //azi[ista]+=azig[i][j]/dis;
            weight+=1./dis;
         }
      if(ntmpp<2){ azi[ista]=-1; continue; }
      grdx[ista]=grdx[ista]/weight;
      grdy[ista]=grdy[ista]/weight;
      grdt[ista]=sqrt(pow(grdx[ista],2)+pow(grdy[ista],2));
      azi[ista]=pi/2-atan2(grdy[ista],grdx[ista]);
      if(azi[ista]<0-1e-8)azi[ista]+=2*pi;
      nstl++;
   }

   if(nstl<10) {
      cout<<nstl<<" stations left. Not enough for computing lplc map."<<endl;
      for(ista=0;ista<*nsta;ista++) azi[ista]=-1; 
      sprintf(buff,"rm -f region_%d", nrand);
      system(buff);
      return 0;
   }

   sprintf(buff,"%s_grdx",file);
   if((ff=fopen(buff,"w"))==NULL){
      printf("Can't open %s to write!\n",buff);
      return 0;
   }
   for(ista=0;ista<*nsta;ista++) {
      if(azi[ista]==-1) continue; 
      fprintf(ff,"%8.4f  %8.4f  %8g\n",slong[ista], slati[ista], grdx[ista]);
   }
   fclose(ff);

   sprintf(buff,"%s_grdy",file);
   if((ff=fopen(buff,"w"))==NULL){
      printf("Can't open %s to write!\n",buff);
      return 0;
   }
   for(ista=0;ista<*nsta;ista++) {
      if(azi[ista]==-1) continue; 
      fprintf(ff,"%8.4f  %8.4f  %8g\n",slong[ista], slati[ista], grdy[ista]);
   }
   fclose(ff);

   sprintf(buff,"/home/tianye/code/Programs/Attenuation/Alpha_from_ampg_btag_tvtl_cap/C_plot_travel %s_grdx region_%d %f %lf %lf", file, nrand, dgr, tension, blc_size);
   system(buff);
   sprintf(buff,"/home/tianye/code/Programs/Attenuation/Alpha_from_ampg_btag_tvtl_cap/C_plot_travel %s_grdy region_%d %f %lf %lf", file, nrand, dgr, tension, blc_size);
   system(buff);

   for(i=0;i<npts_long;i++) for(j=0;j<npts_lati;j++){
      gradtr[i][j] = 1e10;
      gradx[i][j] = -1;
   }

   sprintf(buff,"%s_grdx.HD",file);
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
      gradx[i][j]=dattmp;
   }
   fclose(ff);

   sprintf(buff,"%s_grdy.HD",file);
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
      grady[i][j]=dattmp;
   }
   fclose(ff);

   Check_azi_cov(&slong[0], &slati[0], *nsta, 200, &azi[0]);

   for(ista=0;ista<*nsta;ista++){
      if(azi[ista]==-1) continue;
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradtr[i][j] != 1e10) continue;
            grdtx=(gradx[i+1][j]-gradx[i-1][j])/2.0/dislong[j];
            grdty=(grady[i][j+1]-grady[i][j-1])/2.0/dislati;
            grdtmp=grdtx+grdty;
            alpha=atan(dislong[j]/dislati);
            gradtr[i][j]=-gradx[i-1][j+1]*sin(alpha)+grady[i-1][j+1]*cos(alpha)+grady[i][j+1]+gradx[i+1][j+1]*sin(alpha)+grady[i+1][j+1]*cos(alpha)-gradx[i-1][j]+gradx[i+1][j]-gradx[i-1][j-1]*sin(alpha)-grady[i-1][j-1]*cos(alpha)-grady[i][j-1]+gradx[i+1][j-1]*sin(alpha)-grady[i+1][j-1]*cos(alpha);
            gradtr[i][j]=gradtr[i][j]/8*(dislati+dislong[j])/dislati/dislong[j];
            if(fabs((grdtmp-gradtr[i][j])/gradtr[i][j])>0.5 && fabs((grdtmp-gradtr[i][j]))>0.0001) gradtr[i][j]=1e20;
            //else gradtr[i][j]=grdtmp;
         }
   }

   for(ista=0;ista<*nsta;ista++){
      if(azi[ista]==-1) continue;
      ii=int((slong[ista]-longmin)/dgr);
      jj=int((slati[ista]-latimin)/dgr);
      ntmpp=0; weight=0; lplc[ista]=0;
      for(i=ii;i<ii+2;i++)
         for(j=jj;j<jj+2;j++){
            if(i==0||j==0||i>npts_long-2||j>npts_lati-2) continue;
            if(gradtr[i][j]>=1e10) continue;
            ntmpp++;
            dis=sqrt(pow((longmin+i*dgr-slong[ista])*dislong[j],2)+pow((latimin+j*dgr-slati[ista])*dislati,2))+1e-20;
            lplc[ista]+=gradtr[i][j]/dis;
            weight+=1/dis;
         }
      if(ntmpp<2) { azi[ista]=-1; continue; }
      lplc[ista]=lplc[ista]/weight;
   }

   sprintf(buff,"rm -f %s_grdx %s_grdx.HD %s_grdy %s_grdy.HD region_%d", file, file, file, file, nrand);
   system(buff);

   return 1;
}

