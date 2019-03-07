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
#define NEV 2000


int calc_dist(double lati1, double long1, double lati2, double long2, double *dist);

int calc_azimuth(double lati1, double long1, double lati2, double long2, double *alpha1);

int discard_by_std (double *dat, double *weit, int ndat, double std_coef, int *count, double *weittmp, double *avgo, double *stdo);

int sort ( double *datax, double *datay1, double *datay2, double *datay3, double *datay4, double *datay5, double *datay6, double *datay7, double *datay8, int ndat );

int sort2 ( double *datax, double *datay1, int ndat );

int fit_line( double *datax, double *datay, int idata, int indv, double *slpout, double *itcptout, double *sdout );

int least_fit_sine (double *theta, double *dat, double *weit, int ndat, double *A0, double *A1, double *phi, double *std);

int Gauss_Smoothing(double *lon, double *lat, double *dat, double *densc, int nsta, double hdis);

int calc_smth_grdt(char *file, double *slong, double *slati, double *sdat, double hdis, double *grdt, double *azi, int *nsta, double tension, double blc_size);

int calc_grdt_beta(char *file, double *slong, double *slati, double *sdat, double *grdt, double *azi, int *nsta, double tension);

int calc_lplc(char *file, double *slong, double *slati, double *grdt, double *azi, double *lplc, int *nsta, double tension, double blc_size);


int main (int argc, char *argv[])
{
   if(argc!=5){
      cout<<"usage:csta_map_gradient [event_file.lst (tvtf ampf evlon evlat)] [station.lst] [input_beta_map] [in_parameter_file]"<<endl;
      return 0;
   }
   FILE *ff, *fin, *fout;
   char buff[300], iname[300], dirname[100], ampf[100], tvtf[100], *tmpc;
   double grdt[NSTA], thetat[NSTA], lplc[NSTA], amp[NSTA], grda[NSTA], thetaa[NSTA], beta[NSTA], grdb[NSTA], thetab[NSTA];
   double tlong[NSTA], tlati[NSTA], along[NSTA], alati[NSTA], dist[NSTA][NEV], disttmp, ftmp;
   double clong, clati, cdep, cmag, lontmp, lattmp, lon[NSTA], lat[NSTA];
   double sazi[NSTA], sdis[NSTA], velt[NSTA], dec[NSTA], alpha[NSTA], site[NSTA], prp[NSTA], scoef2[NSTA];
   double alpraw[NSTA][NEV], alpcrd[NSTA][NEV], alpsit[NSTA][NEV], alpprp[NSTA][NEV], vel[NSTA][NEV], alperr[NSTA], velerr[NSTA], azi[NSTA][NEV], azit[NSTA][NEV], aziv[NSTA][NEV], dep[NSTA][NEV], mag[NSTA][NEV];
   double araw[NSTA], acrd[NSTA], asit[NSTA], aprp[NSTA];
   char sta[NSTA][10];
   double lonmin=360, lonmax=0, latmax=-90, latmin=90;
   int i, j, k, iev, ilon, ilat, ist, nst, nsta, nstt, bflag, nvel[NSTA], nvele[NSTA], nalp[NSTA], nalpe[NSTA];
   double weitsum[NSTA];
   double pi=3.1415926536, R=6367.5, std_coef=2.0;
   double hsmt, hcap, pcoef, scoef, hsco, scoef_mn, scoef_mx, std_m, A1_m, azi_gap, min_dis, tension, blc_size;
/*-----------------------------input parameters--------------------------------------
hsmt----------Gaussian smoothing halfdist
hcap----------Gaussian halfdist for station group cap
pcoef---------The factor by which the computed propagating maps are multiplied
scoef---------The factor by which the computed site amplification maps are multiplied
hsco----------Gaussian halfwidth for dynamic scoef
scoef_mn------Allowed minimum dynamic scoef factor
scoef_mx------Allowed maximum dynamic scoef factor
std_m---------Allowed maximum std. of corrected alpha values on a single station
A1_m----------Allowed maximum sinusoidal variation of the corrected alpha
azi_gap-------Allowed maximum azimuthal gap for a sinusoidal fitting
min_dis-------Allowed minimum station to event distance
tension-------Tension factor of spline fit for computing gradient and laplacian
blc_size------Length of blocks in km for blockmean before interpolation
*///---------------------------------------------------------------------------------
   if((ff=fopen(argv[4],"r")) == NULL) {
      cout<<"Can't open parameter file "<<argv[4]<<endl;
      return 0;
   }
   if((fscanf(ff, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &hsmt, &hcap, &pcoef, &scoef, &hsco, &scoef_mn, &scoef_mx, &std_m, &A1_m, &azi_gap, &min_dis, &tension, &blc_size))!=13) {
      cout<<"Incorrect # of parameters in file "<<argv[4]<<endl;
      cout<<"Shoud have: hsmt hcap pcoef scoef hsco scoef_mn scoef_mx std_m A1_m azi_gap min_dis tension blc_size"<<endl;
      return 0;
   }
   fclose(ff);
   double gausco=0.5/hsco/hsco, gaucap=0.5/hcap/hcap;
   char stype[5];
   if(scoef_mn == scoef_mx) sprintf(stype, "fix\0");
   else sprintf(stype, "dyn\0");

   //double alpmn=-0.005, alpmx=0.006;
   if((ff=fopen(argv[2],"r")) == NULL) {
      cout<<"Can't open station list "<<argv[2]<<endl;
      return 0;
   }
   for(nst=0;;nst++){
      if(fgets(buff, 300, ff) == NULL ) break;
      if((sscanf(buff,"%s %lf %lf", &sta[nst][0], &lon[nst], &lat[nst]))!=3) {
         cout<<"Stopped! Wrong format: "<<buff<<endl;
         exit(0);
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

   sprintf(dirname, "ev.dis_azi_vel_dec_sit_prp_alp.sm%.1f.ts%.2f.bs%.0f\0", hsmt, tension, blc_size);
   sprintf(buff,"mkdir -p %s",dirname);
   system(buff);
   sprintf(iname,"%s/beta_lon_lat_beta_grdb_thetab.txt", dirname);
   bflag=-1;
   if((fin=fopen(iname,"r"))!=NULL) {
      bflag=1;
      for(i=0;i<nst;i++) {
         if((fgets(buff, 300, fin))==NULL) break;
         if(sscanf(buff, "%lf %lf %lf %lf %lf", &lontmp, &lattmp, &beta[i], &grdb[i], &thetab[i])!=5) break;
         if(lontmp<0.) lontmp += 360.;
         if(fabs(lon[i]-lontmp)+fabs(lat[i]-lattmp)>0.01) break;
      }
      if(i<nst) bflag=-1;
      fclose(fin);
   }
   if(bflag!=1) {
      calc_grdt_beta(argv[3], lon, lat, beta, grdb, thetab, &nst, tension);
      fin=fopen(iname,"w");
      for(i=0;i<nst;i++) fprintf(fin, "%lf %lf %lf %lf %lf\n", lon[i], lat[i], beta[i], grdb[i], thetab[i]);
      fclose(fin);
   }
 //for(i=0;i<nst;i++) cout<<lon[i]<<" "<<lat[i]<<"  beta: "<<beta[i]<<"  grdb: "<<grdb[i]<<"  thetab: "<<thetab[i]<<endl;

   if((ff=fopen(argv[1],"r")) == NULL) {
      cout<<"Can't open ev_files list "<<argv[1]<<endl;
      return 0;
   }
   cout<<"Computing alpha from each events..."<<endl;
   double *densc = NULL;
   for(iev=0;;iev++){
      if(fgets(buff, 300, ff) == NULL ) break;
      if((sscanf(buff,"%s %s %lf %lf %lf %lf", &tvtf[0], &ampf[0], &clong, &clati, &cdep, &cmag))!=6) {
         cout<<"Stopped! Wrong format: "<<buff<<endl;
         exit(0);
      }
      tmpc = strdup(ampf);
      tmpc = strtok(tmpc, "_");
      sprintf(iname,"%s/%s.dis_azi_vel_dec_sit_prp_thetat.txt", dirname, tmpc);
      if((fin=fopen(iname,"r"))==NULL) {
         if((fout=fopen(ampf,"r"))==NULL) continue;
         fclose(fout);
         if((fout=fopen(tvtf,"r"))==NULL) continue;
         fclose(fout);
         calc_smth_grdt(ampf, along, alati, amp, hsmt, grda, thetaa, &nsta, tension, blc_size);
         calc_lplc(tvtf, tlong, tlati, grdt, thetat, lplc, &nstt, tension, blc_size);
         densc= (double *)realloc(densc, nstt * sizeof(double));
         Gauss_Smoothing(&tlong[0], &tlati[0], &lplc[0], densc, nstt, hsmt);
         Gauss_Smoothing(&tlong[0], &tlati[0], &grdt[0], densc, nstt, hsmt);
         if(nsta != nstt){
           cout<<"Sta# mismatch between "<<ampf<<" and "<<tvtf<<endl;
           continue;
         }
         if((fout=fopen(iname,"w"))==NULL) {
            cout<<"Cannot open file "<<iname<<" to write."<<endl;
            exit(0);
         }
         for(i=0;i<nsta;i++){
            if(thetaa[i]==-1||thetat[i]==-1) continue;
            if(1./grdt[i]<2. || 1./grdt[i]>5. || fabs(lplc[i])>0.005) continue;
            ilon=(int)floor(along[i]-lonmn+0.5);
            ilat=(int)floor(alati[i]-latmn+0.5);
            for(j=1;j<=stan[ilon][ilat][0];j++){
               ist=stan[ilon][ilat][j];
               if(fabs(along[i]-lon[ist])+fabs(alati[i]-lat[ist])<0.01) break;
            }
            if(j>stan[ilon][ilat][0]) continue;
            velt[i]=1.0/grdt[i];
            calc_azimuth(alati[i], along[i], clati, clong, &sazi[i]);
            calc_dist(clati, clong, alati[i], along[i], &sdis[i]);
            dec[i]=-(grda[i]*cos(thetaa[i]-thetat[i]))/amp[i]-0.5/R/tan(sdis[i]/R);
            site[i]=(grdb[ist]*cos(thetab[ist]-thetat[i]))/beta[ist];
            prp[i]=(-0.5/grdt[i]*lplc[i]+0.5/R/tan(sdis[i]/R));
            fprintf(fout, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", along[i], alati[i], sdis[i], sazi[i], velt[i], dec[i], site[i], prp[i], thetat[i]);
         }
         fclose(fout);
      }
      else {
         for(i=0;;i++) {
            if(fgets(buff, 300, fin)==NULL) break;
            sscanf(buff, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &along[i], &alati[i], &sdis[i], &sazi[i], &velt[i], &dec[i], &site[i], &prp[i], &thetat[i]);
            thetaa[i]=1; //thetat[i]=1;
         }
         nsta=i;
         fclose(fin);
      }

      nstt=0;
      for(i=0;i<nsta;i++) {
         if(thetaa[i]==-1||thetat[i]==-1) continue;
         //alpha[i]=dec[i]+scoef*site[i]+pcoef*prp[i];
         //if(alpha[i]<alpmn || alpha[i]>alpmx) continue;
         nstt++;
      }
      if(nstt<50) {
         cout<<"Event "<<iev<<": "<<tmpc<<": Only "<<nstt<<" stations left. Skipped."<<endl;
         continue;
      }
      for(i=0;i<nsta;i++) {
         if(thetaa[i]==-1||thetat[i]==-1) continue;
         //if(alpha[i]<alpmn || alpha[i]>alpmx) continue;
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
         vel[ist][nvel[ist]] = velt[i];
         ftmp = thetat[i]*180./pi-180.;
         if(ftmp<0) ftmp += 360.;
         aziv[ist][nvel[ist]] = ftmp;
         nvel[ist] += 1;
         if(nvel[ist]>NEV) {
            cout<<"Stopped: # of events on station "<<sta[ist]<<" passes "<<NEV<<"!"<<endl;
            exit(0);
         }
         if(thetab[ist]==-1) continue;
         dist[ist][nalp[ist]] = sdis[i];
         azi[ist][nalp[ist]] = sazi[i];
         azit[ist][nalp[ist]] = ftmp;
         alpraw[ist][nalp[ist]]=dec[i];
         alpsit[ist][nalp[ist]]=scoef*site[i];
         alpprp[ist][nalp[ist]]=pcoef*prp[i];
         dep[ist][nalp[ist]] = cdep;
         mag[ist][nalp[ist]] = cmag;
         nalp[ist] += 1;
      }

      cout<<"Event "<<iev<<": "<<tmpc<<" finished"<<endl;
   }
   fclose(ff);

   double avg, std, stdr, d_tmp, coef, avgr, slp, slpr, itcpt, fct, weit, A0, A1, phi;
   //double slpmn=, slpmx=;
   int count, fgap;
   sprintf(dirname, "test_alp_sm%.1f_ts%.2f_bs%.0f_pc%.1f_sc%.1f", hsmt, tension, blc_size, pcoef, scoef);
   sprintf(buff,"mkdir -p %s", dirname);
   system(buff);

   cout<<"Computing phase velocity map..."<<endl;
   double weitv[NEV], weittmp;
   for(i=0;i<nst;i++) {
      if(nvel[i]<5) continue;
      sort2(&aziv[i][0], &vel[i][0], nvel[i]);
      sprintf(buff,"%s/%.3f_%.3f_vel\0",dirname,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<nvel[i];j++) fprintf(fout,"%f %lf\n",aziv[i][j],vel[i][j]);
      fclose(fout);
      for(j=0;j<nvel[i];j++) weitv[j]=1;
      discard_by_std (&vel[i][0], &weitv[0], nvel[i], std_coef, &count, &weittmp, &A0, &std);
      fgap=1;
      for(j=0;j<nvel[i];j++) {
         if(vel[i][j]==-1) continue;
         k=j+1;
         for(;;) {
            if(k>=nvel[i]) k -= nvel[i];
            if(vel[i][j]!=-1) break;
            k++;
         }
         d_tmp = vel[i][k]-vel[i][j];
         if(d_tmp<0) d_tmp += 360.;
         if(d_tmp>azi_gap) { fgap=-1; break; }
      }
      if(fgap==1) least_fit_sine (&aziv[i][0], &vel[i][0], &weitv[0], nvel[i], &A0, &A1, &phi, &std);
      if(std>0.2) {nvele[i]=-1; continue;}
      vel[i][0] = A0;
      velerr[i] = std/sqrt((float)count-1.);
      nvele[i] = count;
   }

   int ntmp, nscap=30000;
   double azitmp[nscap], azittmp[nscap], distmp[nscap], deptmp[nscap], magtmp[nscap], rawtmp[nscap], prptmp[nscap], sittmp[nscap], crdtmp[nscap], weight[nscap];
   for(i=0;i<nst;i++) { 
   cout<<"Computing attenuation ceof at station "<<i+1<<"/"<<nst<<"..."<<endl;
      ntmp=0; weitsum[i]=0;
      for(j=0;j<nst;j++) {
         calc_dist(lat[i], lon[i], lat[j], lon[j], &disttmp);
         //disttmp = sqrt(pow((lon[j]-lon[i]),2)+pow((lat[j]-lat[i]),2));
         if(disttmp>hcap*2) continue; 
         weittmp = exp(-gaucap*pow(disttmp,2));
         for(k=0;k<nalp[j];k++) {
            if(dist[j][k]<min_dis) continue;
            //if(azi[j][k]>320. || azi[j][k]<170.) continue;
            azitmp[ntmp] = azi[j][k];
            azittmp[ntmp] = azit[j][k];
            distmp[ntmp] = dist[j][k];
            deptmp[ntmp] = dep[j][k];
            magtmp[ntmp] = mag[j][k];
            rawtmp[ntmp] = alpraw[j][k];
            prptmp[ntmp] = alpprp[j][k];
            sittmp[ntmp] = alpsit[j][k];
            weight[ntmp] = weittmp;
            weitsum[i] += weittmp;
    //cout<<ntmp<<": a "<<azitmp[ntmp]<<" d "<<distmp[ntmp]<<" r "<<rawtmp[ntmp]<<" p "<<prptmp[ntmp]<<" s "<<sittmp[ntmp]<<" w "<<weight[ntmp]<<" ws "<<weitsum[i]<<endl;
            ntmp++;
         }
      }
      scoef2[i] = -1;
      if(weitsum[i]<10) {cout<<"Less than 10 events on this station, skipped"<<endl; continue;}

      sort(&azittmp[0], &deptmp[0], &magtmp[0], &distmp[0], &rawtmp[0], &prptmp[0], &sittmp[0], &weight[0], &azitmp[0], ntmp);

      fgap=1;
      if(azittmp[0]+360.-azittmp[ntmp-1]>azi_gap) fgap=-1;
      if(fgap==1) for(j=1;j<ntmp;j++) {
         if(azittmp[j]-azittmp[j-1]>azi_gap) { fgap=-1; break; }
      }
      fct=1e9;
      if(fgap==-1) {
         scoef2[i]=1.;
      }
      else for(coef=scoef_mn;coef<=scoef_mx;coef+=0.2){
         //coef = icoef*0.2; //avg=0; std=0;
         weit = exp(-gausco*pow((coef-1.),2));
         for(j=0;j<ntmp;j++) {
            crdtmp[j] = rawtmp[j] + prptmp[j] + coef*sittmp[j];
            //avg += alpcrd[i][j];
         }
         discard_by_std (&crdtmp[0], &weight[0], ntmp, std_coef, &count, &weittmp, &A0, &std);
         fit_line ( &azittmp[0], &crdtmp[0], ntmp, 0, &slp, &itcpt, &std );
         //cout<<"coef: "<<coef<<"  fct: "<<fabs(slp*std)/weit<<"( slp: "<<slp<<" std: "<<std<<" weit: "<<weit<<" )"<<endl;
         if(fabs(slp*std)/weit<fct) {
            //slpr=slp; stdr=std; avgr = itcpt+slp*180.; 
            scoef2[i]=coef;
            fct=fabs(slp*std)/weit;
         }
      }
  //cout<<"ntmp: "<<ntmp<<endl;
      for(j=0;j<ntmp;j++) {
         sittmp[j] = sittmp[j]*scoef2[i];
         crdtmp[j] = rawtmp[j] + prptmp[j] + sittmp[j];
  //cout<<"crdtmp: "<<crdtmp[j]<<endl;
      }

      sprintf(buff,"%s/%.3f_%.3f_sit\0",dirname,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf %lf %lf %lf\n",azittmp[j],distmp[j],sittmp[j],azitmp[j],deptmp[j],magtmp[j]);
      fclose(fout);
      sprintf(buff,"%s/%.3f_%.3f_prp\0",dirname,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf %lf %lf %lf\n",azittmp[j],distmp[j],prptmp[j],azitmp[j],deptmp[j],magtmp[j]);
      fclose(fout);
      sprintf(buff,"%s/%.3f_%.3f_raw\0",dirname,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf %lf %lf %lf\n",azittmp[j],distmp[j],rawtmp[j],azitmp[j],deptmp[j],magtmp[j]);
      fclose(fout);
      sprintf(buff,"%s/%.3f_%.3f_prp_crd\0",dirname,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf %lf %lf %lf\n",azittmp[j],distmp[j],rawtmp[j]+prptmp[j],azitmp[j],deptmp[j],magtmp[j]);
      fclose(fout);
      sprintf(buff,"%s/%.3f_%.3f_crd\0",dirname,lon[i],lat[i]);
      fout=fopen(buff,"w");
      for(j=0;j<ntmp;j++) fprintf(fout,"%f %f %lf %lf %lf %lf\n",azittmp[j],distmp[j],crdtmp[j],azitmp[j],deptmp[j],magtmp[j]);
      fclose(fout);

      //if(fabs(slpr)>2e-6) {nalpe[i]=-1; continue;}
      
//cout<<"slpr: "<<slpr<<"  stdr: "<<stdr<<"  avgr: "<<avgr<<"  scoef: "<<scoef[i]<<endl;

      discard_by_std (&crdtmp[0], &weight[0], ntmp, std_coef, &count, &weittmp, &A0, &std);
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
      //fit_line ( &azittmp[0], &crdtmp[0], ntmp, 0, &slp, &itcpt, &std );
      //for(j=0;j<ntmp;j++) if(crdtmp[j]!=-1) { cout<<azittmp[j]<<" "<<crdtmp[j]<<endl; }
      //cout<<"[line]  slp: "<<slp<<"  itcpt: "<<itcpt<<"  std: "<<std<<endl;
      fgap=1;
      for(j=0;j<ntmp;j++) {
         if(crdtmp[j]==-1) continue;
         k=j+1;
         for(;;) {
            if(k>=ntmp) k -= ntmp;
            if(crdtmp[k]!=-1) break;
            k++;
         }
         d_tmp = azittmp[k]-azittmp[j];
         if(d_tmp<0) d_tmp += 360.;
         if(d_tmp>azi_gap) { fgap=-1; break; }
      }     
      A1=0;
      if(fgap==1) least_fit_sine (&azittmp[0], &crdtmp[0], &weight[0], ntmp, &A0, &A1, &phi, &std);
      //cout<<"[sine]  A0: "<<A0<<"  std: "<<std<<endl<<endl;
//cout<<A0<<" + "<<A1<<" * sin( theta + "<<phi<<" )  std: "<<std<<endl;
      if(std>std_m || (scoef!=0 && fabs(A1)>A1_m)) {nalpe[i]=-1; continue;}
      //alpcrd[i][0] = itcpt + slp*180.;
      acrd[i] = A0;
      alperr[i] = std/sqrt((float)count-1.);
      nalpe[i] = count;
      weitsum[i] = weittmp;
   }

   avg=0.; j=0;
   for(i=0;i<nst;i++) {
      if(nalpe[i]<10) continue;
      avg += acrd[i];
      j++;
   }
   avg /= j; std=0.;
   for(i=0;i<nst;i++) {
      if(nalpe[i]<10) continue;
      std += pow(acrd[i]-avg, 2);
   }
   std = sqrt(std/(j-1));
   for(i=0;i<nst;i++) {
      if(nalpe[i]<0) continue;
      if(fabs(acrd[i]-avg)>3.5*std) nalpe[i]=-1;
   }

   sprintf(dirname, "alpha_sm%.1f_ts%.2f_bs%.0f_pc%.1f_sc%.1f_%s", hsmt, tension, blc_size, pcoef, scoef, stype);

   sprintf(buff, "mkdir -p %s\n", dirname);
   system(buff);
   sprintf(iname,"%s/vel_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d\n",lon[i],lat[i],vel[i][0], nvele[i]);
   fclose(fout);

   sprintf(iname,"%s/alpha_raw_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],araw[i], nalpe[i], weitsum[i]);
   fclose(fout);

   sprintf(iname,"%s/alpha_crd_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf %lf\n",lon[i],lat[i],acrd[i], nalpe[i], weitsum[i], alperr[i]);
   fclose(fout);

   sprintf(iname,"%s/alpha_prp_crd_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],(araw[i]+aprp[i]), nalpe[i], weitsum[i]);
   fclose(fout);

   sprintf(iname,"%s/alpha_sit_crd_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],(araw[i]+asit[i]), nalpe[i], weitsum[i]);
   fclose(fout);

   sprintf(iname,"%s/alpha_site_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],asit[i], nalpe[i], weitsum[i]);
   fclose(fout);

   sprintf(iname,"%s/site_coef_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],scoef2[i],nalpe[i], weitsum[i]);
   fclose(fout);

   sprintf(iname,"%s/alpha_prp_map",dirname);
   fout=fopen(iname,"w");
   for(i=0;i<nst;i++) fprintf(fout,"%lf %lf %lf %d %lf\n",lon[i],lat[i],aprp[i], nalpe[i], weitsum[i]);
   fclose(fout);

   return 1;
}
