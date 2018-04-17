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

int discard_by_std (double *dat, double *weit, int ndat, double std_coef, int *count, double *avgo, double *stdo);

int sort ( double *datax, double *datay1, double *datay2, double *datay3, double *datay4, double *datay5, int ndat );

int sort2 ( double *datax, double *datay1, int ndat );

int fit_line( double *datax, double *datay, int idata, int indv, double *slpout, double *itcptout, double *sdout );

int least_fit_sine (double *theta, double *dat, double *weit, int ndat, double *A0, double *A1, double *phi, double *std);

int Gauss_Smoothing(double *lon, double *lat, double *dat, double *densc, int nsta, double hdis);

int calc_smth_grdt(char *file, double *slong, double *slati, double *sdat, double hdis, double *grdt, double *azi, int *nsta);

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
   double grdt[NSTA], azit[NSTA], lplc[NSTA], amp[NSTA], grda[NSTA], azia[NSTA], beta[NSTA], grdb[NSTA], azib[NSTA];
   double tlong[NSTA], tlati[NSTA], along[NSTA], alati[NSTA], dist[NSTA][500];
   double clong, clati, lon[NSTA], lat[NSTA];
   double dec[NSTA], alpha[NSTA], site[NSTA], prp[NSTA];
   double alpraw[NSTA][500], alpcrd[NSTA][500], alpsit[NSTA][500], alpprp[NSTA][500], vel[NSTA][500], alperr[NSTA], velerr[NSTA], azi[NSTA][500], aziv[NSTA][500];
   double araw[NSTA], acrd[NSTA], asit[NSTA], aprp[NSTA];
   char sta[NSTA][10];
   double lonmin=360, lonmax=0, latmax=-90, latmin=90;
   int i, j, k, iev, ilon, ilat, ist, nst, nsta, nstt, nbad, nvel[NSTA], nvele[NSTA], nalp[NSTA], nalpe[NSTA];
   double weitsum[NSTA];
   double per=60., pi=3.1415926536, R=6367.5, alpmn=-0.004, alpmx=0.0045, hdis=0., pcoef=1.0, scoef=1.3;
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
      nvel[i]=0;
   }
   calc_grdt_beta(argv[3], lon, lat, beta, grdb, azib, &nst);
 //for(i=0;i<nst;i++) cout<<lon[i]<<" "<<lat[i]<<"  beta: "<<beta[i]<<"  grdb: "<<grdb[i]<<"  azib: "<<azib[i]<<endl;

   if((ff=fopen(argv[1],"r")) == NULL) {
      cout<<"Can't open ev_files list "<<argv[1]<<endl;
      return 0;
   }
   cout<<"Computing alpha from each events..."<<endl;
   double *densc = NULL;
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
      calc_smth_grdt(ampf, along, alati, amp, hdis, grda, azia, &nsta);
      calc_lplc(tvtf, tlong, tlati, grdt, azit, lplc, &nstt);
      densc= (double *)realloc(densc, nstt * sizeof(double));
      Gauss_Smoothing(&tlong[0], &tlati[0], &lplc[0], densc, nstt, hdis);
      Gauss_Smoothing(&tlong[0], &tlati[0], &grdt[0], densc, nstt, hdis);
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
         vel[ist][nvel[ist]] = 1.0/grdt[i];
         calc_azimuth(tlati[i], tlong[i], clati, clong, &aziv[ist][nvel[ist]]);
         nvel[ist] += 1;
         if(azib[ist]==-1) continue;
         calc_dist(clati, clong, alati[i], along[i], &dist[ist][nalp[ist]]);
         dec[i]=-(grda[i]*cos(azia[i]-azit[i]))/amp[i]-0.5/R/tan(dist[ist][nalp[ist]]/R);
         site[i]=scoef*(grdb[ist]*cos(azib[ist]-azit[i]))/beta[ist];
         prp[i]=pcoef*(-0.5/grdt[i]*lplc[i]+0.5/R/tan(dist[ist][nalp[ist]]/R));
         alpha[i]=dec[i]+site[i]+prp[i];
         //cout<<dec[i]<<" to "<<alpha[i]<<endl;
         if(alpha[i]<alpmn || alpha[i]>alpmx) continue;
         azi[ist][nalp[ist]] = aziv[ist][nvel[ist]-1];
         //calc_azimuth(tlati[i], tlong[i], clati, clong, &azi[ist][nalp[ist]]);
         alpraw[ist][nalp[ist]]=dec[i];
         //alpcrd[ist][nalp[ist]]=alpha[i];
         alpsit[ist][nalp[ist]]=site[i];
         alpprp[ist][nalp[ist]]=prp[i];
         nalp[ist] += 1;
      }
      //cout<<"alpcrd[0]: "<<alpcrd[0][nalp[0]-1]<<"  nalp[0]: "<<nalp[0]<<"  alpcrd[585]: "<<alpcrd[585][nalp[585]-1]<<"  nalp[585]: "<<nalp[585]<<endl;

      tmpc = strdup(ampf);
      tmpc = strtok(tmpc, "_");
      cout<<"Event "<<iev<<": "<<tmpc<<" finished"<<endl;
//*
      if(nsta<100) continue;
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
//*/
   }
   fclose(ff);

   double avg, std, stdr, tmp, coef, avgr, slp, slpr, itcpt, fct, weit, A0, A1, phi, gaua=0.5/0.25/0.25, std_coef=2.0; //gaua=0.5/1.0/1.0;
   //double slpmn=, slpmx=;
   int count, icoef, fgap;
   sprintf(buff,"mkdir -p test_alp_sm_%.0f_%.1f_%.1f", hdis, pcoef, scoef);
   system(buff);

   double weitv[500];
   for(i=0;i<nst;i++) {
      if(nvel[i]<5) continue;
      sort2(&aziv[i][0], &vel[i][0], nvel[i]);
      sprintf(buff,"test_alp_sm_%.0f_%.1f_%.1f/%.3f_%.3f_vel\0",hdis,pcoef,scoef,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<nvel[i];j++) fprintf(fout,"%f %lf\n",aziv[i][j],vel[i][j]);
      fclose(fout);
      for(j=0;j<nvel[i];j++) weitv[j]=1;
      discard_by_std (&vel[i][0], &weitv[0], nvel[i], std_coef, &count, &A0, &std);
      fgap=1;
      if(vel[i][0]!=-1 && vel[i][nvel[i]-1]!=-1) if(vel[i][0]+360.-vel[i][nvel[i]-1]>250.) fgap=-1;
      if(fgap==1) for(j=1;j<nvel[i];j++) {
         if(vel[i][j]==-1) continue;
         if(vel[i][j]-vel[i][j-1]>250.) { fgap=-1; break; }
      }
      if(fgap==1) least_fit_sine (&aziv[i][0], &vel[i][0], &weitv[0], nvel[i], &A0, &A1, &phi, &std);
      if(std>0.1) {nvele[i]=-1; continue;}
      vel[i][0] = A0;
      velerr[i] = std/sqrt((float)count-1.);
      nvele[i] = count;
   }

   int ntmp, nscap=3000;
   double azitmp[nscap], disttmp, distmp[nscap], rawtmp[nscap], prptmp[nscap], sittmp[nscap], crdtmp[nscap], weittmp, weight[nscap];
   double hcap=10., gaucap=0.5/hcap/hcap;
   for(i=0;i<nst;i++) { 
      ntmp=0; weitsum[i]=0;
      for(j=0;j<nst;j++) {
         calc_dist(lat[i], lon[i], lat[j], lon[j], &disttmp);
         //disttmp = sqrt(pow((lon[j]-lon[i]),2)+pow((lat[j]-lat[i]),2));
         if(disttmp>hcap*2) continue; 
         weittmp = exp(-gaucap*pow(disttmp,2));
         for(k=0;k<nalp[j];k++) {
            if(dist[j][k]<4000.) continue;
            azitmp[ntmp] = azi[j][k];
            distmp[ntmp] = dist[j][k];
            rawtmp[ntmp] = alpraw[j][k];
            prptmp[ntmp] = alpprp[j][k];
            sittmp[ntmp] = alpsit[j][k];
            weight[ntmp] = weittmp;
            weitsum[i] += weittmp;
    //cout<<ntmp<<": a "<<azitmp[ntmp]<<" d "<<distmp[ntmp]<<" r "<<rawtmp[ntmp]<<" p "<<prptmp[ntmp]<<" s "<<sittmp[ntmp]<<" w "<<weight[ntmp]<<" ws "<<weitsum[i]<<endl;
            ntmp++;
         }
      }
      if(weitsum[i]<10) continue;
      fct=1e9;
      sort(&azitmp[0], &distmp[0], &rawtmp[0], &prptmp[0], &sittmp[0], &weight[0], ntmp);
/*
      for(icoef=6;icoef<=7;icoef++){
         coef = icoef*0.2; //avg=0; std=0;
         weit = exp(-gaua*pow((coef-scoef),2));
         for(j=0;j<ntmp;j++) {
            crdtmp[j] = rawtmp[j] + prptmp[j] + coef*sittmp[j];
            //avg += alpcrd[i][j];
         }
         discard_by_std (&crdtmp[0], &weight[0], ntmp, std_coef, &count);
         fit_line ( &azitmp[0], &crdtmp[0], ntmp, 0, &slp, &itcpt, &std );
         if(fabs(slp*std)/weit<fct) {
            slpr=slp; stdr=std; avgr = itcpt+slp*180.; scoef[i]=coef;
            fct=fabs(slp*std)/weit;
         }
      }
*/
      for(j=0;j<ntmp;j++) crdtmp[j] = rawtmp[j] + prptmp[j] + sittmp[j];

      sprintf(buff,"test_alp_sm_%.0f_%.1f_%.1f/%.3f_%.3f_sit\0",hdis,pcoef,scoef,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf\n",azitmp[j],distmp[j],sittmp[j]);
      fclose(fout);
      sprintf(buff,"test_alp_sm_%.0f_%.1f_%.1f/%.3f_%.3f_prp\0",hdis,pcoef,scoef,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf\n",azitmp[j],distmp[j],prptmp[j]);
      fclose(fout);
      sprintf(buff,"test_alp_sm_%.0f_%.1f_%.1f/%.3f_%.3f_raw\0",hdis,pcoef,scoef,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf\n",azitmp[j],distmp[j],rawtmp[j]);
      fclose(fout);
      sprintf(buff,"test_alp_sm_%.0f_%.1f_%.1f/%.3f_%.3f_prp_crd\0",hdis,pcoef,scoef,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf\n",azitmp[j],distmp[j],rawtmp[j]+prptmp[j]);
      fclose(fout);
      sprintf(buff,"test_alp_sm_%.0f_%.1f_%.1f/%.3f_%.3f_crd\0",hdis,pcoef,scoef,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf\n",azitmp[j],distmp[j],crdtmp[j]);
      fclose(fout);

      //if(fabs(slpr)>2e-6) {nalpe[i]=-1; continue;}
      
//cout<<"slpr: "<<slpr<<"  stdr: "<<stdr<<"  avgr: "<<avgr<<"  scoef: "<<scoef[i]<<endl;

      discard_by_std (&crdtmp[0], &weight[0], ntmp, std_coef, &count, &A0, &std);
      araw[i]=0; aprp[i]=0; asit[i]=0;
      for(j=0;j<ntmp;j++) {
         if(crdtmp[j]==-1) continue;
         araw[i] += rawtmp[j];
         asit[i] += sittmp[j];
         aprp[i] += prptmp[j];
      }
      araw[i] /= count;
      asit[i] /= count;
      aprp[i] /= count;
      //fit_line ( &azitmp[0], &crdtmp[0], ntmp, 0, &slp, &itcpt, &std );
      //for(j=0;j<ntmp;j++) if(crdtmp[j]!=-1) { cout<<azitmp[j]<<" "<<crdtmp[j]<<endl; }
      //cout<<"[line]  slp: "<<slp<<"  itcpt: "<<itcpt<<"  std: "<<std<<endl;
      fgap=1;
      if(crdtmp[0]!=-1 && crdtmp[ntmp-1]!=-1) if(azitmp[0]+360.-azitmp[ntmp-1]>250.) fgap=-1;
      if(fgap==1) for(j=1;j<ntmp;j++) {
         if(crdtmp[j]==-1) continue;
         if(azitmp[j]-azitmp[j-1]>250.) { fgap=-1; break; }
      }
      A1=0;
      if(fgap==1) least_fit_sine (&azitmp[0], &crdtmp[0], &weight[0], ntmp, &A0, &A1, &phi, &std);
      //cout<<"[sine]  A0: "<<A0<<"  std: "<<std<<endl<<endl;
//cout<<A0<<" + "<<A1<<" * sin( theta + "<<phi<<" )  std: "<<std<<endl;
      if(std>0.001 || fabs(A1)>0.0005) {nalpe[i]=-1; continue;}
      //alpcrd[i][0] = itcpt + slp*180.;
      acrd[i] = A0;
      alperr[i] = std/sqrt((float)count-1.);
      nalpe[i] = count;
   }

   fout=fopen("vel_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],vel[i][0], nvele[i]);
   fclose(fout);

   fout=fopen("alpha_raw_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],araw[i], nalpe[i], weitsum[i]);
   fclose(fout);

   fout=fopen("alpha_crd_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf %lf\n",lon[i],lat[i],acrd[i], nalpe[i], weitsum[i], alperr[i]);
   fclose(fout);

   fout=fopen("alpha_prp_crd_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],(araw[i]+aprp[i]), nalpe[i], weitsum[i]);
   fclose(fout);

   fout=fopen("alpha_sit_crd_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],(araw[i]+asit[i]), nalpe[i], weitsum[i]);
   fclose(fout);

   fout=fopen("alpha_site_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],asit[i], nalpe[i], weitsum[i]);
   fclose(fout);

   //fout=fopen("site_coef_map","w");
   //for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],scoef[i],nalpe[i]);
   //fclose(fout);

   fout=fopen("alpha_prp_map","w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],aprp[i], nalpe[i], weitsum[i]);
   fclose(fout);

   return 1;
}
