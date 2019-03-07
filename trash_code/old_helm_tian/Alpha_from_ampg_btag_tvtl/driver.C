#define MAIN
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <time.h>
using namespace std;

#define NSTA 2000


int calc_dist(double lati1, double long1, double lati2, double long2, double *dist);

int calc_azimuth(double lati1, double long1, double lati2, double long2, double *alpha1);

int calc_azimuth(double lati1, double long1, double lati2, double long2, double *alpha1);

int calc_smth_grdt(char *file, double *slong, double *slati, double *sdat, double *grdt, double *azi, int *nsta);

int calc_grdt_beta(char *file, double *slong, double *slati, double *sdat, double *grdt, double *azi, int *nsta);

int calc_lplc(char *file, double *slong, double *slati, double *grdt, double *azi, double *lplc, int *nsta);


int main (int argc, char *argv[])
{
   if(argc!=4){
      cout<<"usage:csta_map_gradient [event_file.lst (tvtf ampf evlon evlat)] [station.lst] [input_beta_map]"<<endl;
      return 0;
   }
   FILE *ff, *fout;
   char buff[300], fname[100], ampf[100], tvtf[100], *tmpc;
   double grdt[NSTA], azit[NSTA], lplc[NSTA], amp[NSTA], grda[NSTA], azia[NSTA], beta[NSTA], grdb[NSTA], azib[NSTA], azi;
   double tlong[NSTA], tlati[NSTA], along[NSTA], alati[NSTA], dist[NSTA];
   double clong, clati, lon[NSTA], lat[NSTA];
   double dec[NSTA], alpha[NSTA], site[NSTA], prp[NSTA];
   double alpraw[NSTA][500], alpcrd[NSTA][500], alpsit[NSTA][500], alpprp[NSTA][500], alperr[NSTA];
   char sta[NSTA][10];
   double lonmin=360, lonmax=0, latmax=-90, latmin=90;
   int i, j, iev, ilon, ilat, ist, nst, nsta, nstt, nbad, nalp[NSTA];
   double per=25., pi=3.1415926536, alpmn=-0.002, alpmx=0.004;
//alpmn= 2*pi/3.5/per/300.-0.001, alpmx=2*pi/3.5/per/18.+0.001;
   if((ff=fopen(argv[2],"r")) == NULL) {
      cout<<"Can't open station list "<<argv[2]<<endl;
      return 0;
   }
   for(nst=0;;nst++){
      if(fgets(buff, 300, ff) == NULL ) break;
      if((sscanf(buff,"%s %lf %lf", &sta[nst][0], &lon[nst], &lat[nst]))!=3) {
         cout<<"Wrong format! Stopped: "<<buff<<endl;
         break;
      }
      if(lon[nst]<0) lon[nst]+=360.;
      if(lonmin>lon[nst]) lonmin=lon[nst];
      if(lonmax<lon[nst]) lonmax=lon[nst];
      if(latmin>lat[nst]) latmin=lat[nst];
      if(latmax<lat[nst]) latmax=lat[nst];
   }
   fclose(ff);
   cout<<"Initializing station matrices and beta model..."<<endl;
   int lonmn=(int)floor(lonmin), latmn=(int)floor(latmin);
   lonmax=ceil(lonmax); latmax=ceil(latmax);
   int nlon=int(lonmax-lonmn+1);
   int nlat=int(latmax-latmn+1);
   int *stan[nlon][nlat];
   for(ilon=0;ilon<nlon;ilon++) for(ilat=0;ilat<nlat;ilat++){
      stan[ilon][ilat] = (int *)malloc(sizeof(int));
      stan[ilon][ilat][0] = 0;
   }
   for(i=0;i<nst;i++){
      ilon=(int)floor(lon[i]-lonmn+0.5);
      ilat=(int)floor(lat[i]-latmn+0.5);
      stan[ilon][ilat][0] += 1;
      j=stan[ilon][ilat][0];
      stan[ilon][ilat] = (int *)realloc(stan[ilon][ilat], (j+1) * sizeof(int));
      stan[ilon][ilat][j]=i;
   }

   for(i=0;i<nst;i++) {
      //alpraw[i]=0; alpcrd[i]=0; 
      //alpsit[i]=0, alpprp[i]=0;
      nalp[i]=0; 
   }
   calc_grdt_beta(argv[3], lon, lat, beta, grdb, azib, &nst);
 //for(i=0;i<nst;i++) cout<<lon[i]<<" "<<lat[i]<<"  beta: "<<beta[i]<<"  grdb: "<<grdb[i]<<"  azib: "<<azib[i]<<endl;

   if((ff=fopen(argv[1],"r")) == NULL) {
      cout<<"Can't open ev_files list "<<argv[1]<<endl;
      return 0;
   }
   cout<<"Computing alpha from each events..."<<endl;
   for(iev=0;;iev++){
      if(fgets(buff, 300, ff) == NULL ) break;
      if((sscanf(buff,"%s %s %lf %lf", &tvtf[0], &ampf[0], &clong, &clati))!=4) {
         cout<<"Wrong format! Stopped: "<<buff<<endl;
         break;
      }     
      if((fout=fopen(ampf,"r"))==NULL) continue;
      fclose(fout);
      if((fout=fopen(tvtf,"r"))==NULL) continue;
      fclose(fout);
      calc_azimuth(41, 250, clati, clong, &azi);
      if(azi<90 || azi>180) continue;
      calc_smth_grdt(ampf, along, alati, amp, grda, azia, &nsta);
      calc_lplc(tvtf, tlong, tlati, grdt, azit, lplc, &nstt);
      if(nsta != nstt){
        cout<<"Sta# mismatch between "<<ampf<<" and "<<tvtf<<endl;
        continue;
      }

      for(i=0;i<nsta;i++){
         if(azia[i]==-1||azit[i]==-1) continue;
         if(1./grdt[i]<2. || 1./grdt[i]>5. || fabs(lplc[i])>0.005) continue;
         ilon=(int)floor(along[i]-lonmn+0.5);
         ilat=(int)floor(alati[i]-latmn+0.5);
         for(j=1;j<=stan[ilon][ilat][0];j++){
            ist=stan[ilon][ilat][j];
            if(fabs(along[i]-lon[ist])+fabs(alati[i]-lat[ist])<0.01) break;
         }
         if(j>stan[ilon][ilat][0]) {
            cout<<"Station "<<along[i]<<" "<<alati[i]<<" not found in station.lst. Skipped."<<endl;
            continue;
         }
         if(azib[ist]==-1) continue;
         calc_dist(clati, clong, tlati[i], tlong[i], &dist[i]);
         dec[i]=-(grda[i]*cos(azia[i]-azit[i]))/amp[i]-0.5/dist[i];
         site[i]=1.5*(grdb[ist]*cos(azib[ist]-azit[i]))/beta[ist];
         prp[i]=-1.*0.5/grdt[i]*lplc[i]+0.5/dist[i];
         alpha[i]=dec[i]+site[i]+prp[i];
         //cout<<dec[i]<<" to "<<alpha[i]<<endl;
         if(alpha[i]<alpmn || alpha[i]>alpmx) continue;
         //calc_azimuth(clati, clong, tlati[i], tlong[i], &azi);
         //if(azi>90) continue;
         alpraw[ist][nalp[ist]]=dec[i];
         alpcrd[ist][nalp[ist]]=alpha[i];
         alpsit[ist][nalp[ist]]=site[i];
         alpprp[ist][nalp[ist]]=prp[i];
         nalp[ist] += 1;
      }
      //cout<<"alpcrd[0]: "<<alpcrd[0][nalp[0]-1]<<"  nalp[0]: "<<nalp[0]<<"  alpcrd[585]: "<<alpcrd[585][nalp[585]-1]<<"  nalp[585]: "<<nalp[585]<<endl;

      tmpc = strdup(ampf);
      tmpc = strtok(tmpc, "_");
/*  
      sprintf(fname,"%s_alpha_decay",tmpc);
      fout=fopen(fname,"w");
      for(i=0;i<nsta;i++){
         if(azia[i]==-1||azit[i]==-1||azib[i]==-1) continue;
         fprintf(fout,"%8.4f  %8.4f  %8g\n", along[i], alati[i], dec[i]);
      }
      fclose(fout);
 
      sprintf(fname,"%s_alpha_site",tmpc);
      fout=fopen(fname,"w");
      for(i=0;i<nsta;i++){
         if(azia[i]==-1||azit[i]==-1||azib[i]==-1) continue;
         fprintf(fout,"%8.4f  %8.4f  %8g\n", along[i], alati[i], site[i]);
      }
      fclose(fout);

      sprintf(fname,"%s_alpha_prp",tmpc);
      fout=fopen(fname,"w");
      for(i=0;i<nsta;i++){
         if(azia[i]==-1||azit[i]==-1||azib[i]==-1) continue;
        fprintf(fout,"%8.4f  %8.4f  %8g\n", along[i], alati[i], prp[i]);
      }
      fclose(fout);

      sprintf(fname,"%s_tgrdt",tmpc);
      fout=fopen(fname,"w");
      for(i=0;i<nsta;i++){
         if(azia[i]==-1||azit[i]==-1||azib[i]==-1) continue;
         fprintf(fout,"%8.4f  %8.4f  %8g\n", along[i], alati[i], grdt[i]);
      }
      fclose(fout);

      sprintf(fname,"%s_alpha",tmpc);
      fout=fopen(fname,"w");
      for(i=0;i<nsta;i++){
         if(azia[i]==-1||azit[i]==-1||azib[i]==-1) continue;
         fprintf(fout,"%8.4f  %8.4f  %8g\n", along[i], alati[i], alpha[i]);
      }
      fclose(fout);
*/
      cout<<"Event "<<iev<<": "<<tmpc<<" finished"<<endl;
   }
   fclose(ff);

   double avg, std, tmp;
   int count;
   for(i=0;i<nst;i++) {
      avg=0; std=0;
      for(j=0;j<nalp[i];j++) avg += alpcrd[i][j];
      avg /= nalp[i];
      for(j=0;j<nalp[i];j++) std += pow((avg-alpcrd[i][j]),2);
      std = sqrt(std/(nalp[i]-1));
      count=1;
      tmp=alpcrd[i][0];
      if(fabs(alpcrd[i][0]-avg)>1.5*std){
         tmp = -1;
         alpcrd[i][0] =0;
         alpraw[i][0] =0;
         alpsit[i][0] =0;
         alpprp[i][0] =0;
         count --;
      }
      for(j=1;j<nalp[i];j++) {
         if(fabs(alpcrd[i][j]-avg)>1.5*std) continue;
         alpcrd[i][0] += alpcrd[i][j];
         alpraw[i][0] += alpraw[i][j];
         alpsit[i][0] += alpsit[i][j];
         alpprp[i][0] += alpprp[i][j];
         count ++;
      }
      alperr[i]=0;
      for(j=1;j<nalp[i];j++) {
         if(fabs(alpcrd[i][j]-avg)>1.5*std) continue;
         alperr[i] += pow((alpcrd[i][j]-alpcrd[i][0]/count),2);
      }
      if(tmp!=-1) alperr[i] += pow((tmp-alpcrd[i][0]/count),2);
      std = sqrt(alperr[i]/(count-1.));
    cout<<lon[i]<<"_"<<lat[i]<<": "<<std<<endl;
      if(std>0.002) {nalp[i]=-1; continue;}
      alperr[i] = std/sqrt((float)count-1.);
      nalp[i] = count;
   }

   fout=fopen("alpha_raw_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],alpraw[i][0]/nalp[i], nalp[i]);
   fclose(fout);

   fout=fopen("alpha_crd_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],alpcrd[i][0]/nalp[i], nalp[i], alperr[i]);
   fclose(fout);

   fout=fopen("alpha_prp_crd_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],(alpraw[i][0]+alpprp[i][0])/nalp[i], nalp[i]);
   fclose(fout);

   fout=fopen("alpha_sit_crd_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],(alpraw[i][0]+alpsit[i][0])/nalp[i], nalp[i]);
   fclose(fout);

   fout=fopen("alpha_site_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],alpsit[i][0]/nalp[i], nalp[i]);
   fclose(fout);

   fout=fopen("alpha_prp_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],alpprp[i][0]/nalp[i], nalp[i]);
   fclose(fout);

   return 1;
}
