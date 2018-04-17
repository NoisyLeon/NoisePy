#define MAIN
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
using namespace std;

double get_dist(double lat1,double lon1,double lat2,double lon2)
{
  double theta,pi,temp;
  double radius=6371;
  pi=4.0*atan(1.0);

  lat1=atan(0.993277*tan(lat1/180*pi))*180/pi;
  lat2=atan(0.993277*tan(lat2/180*pi))*180/pi;

  temp=sin((90-lat1)/180*pi)*cos(lon1/180*pi)*sin((90-lat2)/180*pi)*cos(lon2/180*pi)+sin((90-lat1)/180*pi)*sin(lon1/180*pi)*sin((90-lat2)/180*pi)*sin(lon2/180*pi)+cos((90-lat1)/180*pi)*cos((90-lat2)/180*pi);
  if(temp>1)
    {
      cout<<"warning cos(theta)>1 and correct to 1!!"<<temp<<endl;
      //cout<<lat1<<" "<<lon1<<" "<<lat2<<" "<<lon2<<endl;
      temp=1;
    }
  if(temp<-1)
    {
      cout<<"warning cos(theta)<-1 and correct to -1!!"<<temp<<endl;
      temp=-1;
    }
  theta=fabs(acos(temp));
  return theta*radius;
}


int main(int na, char *arg[])
{
  if(na!=7)
    {
      cout<<"usage:amp_gra_laplace_HD event.lst period x0 y0 npts_x npts_y"<<endl;
      return 0;
    }
  FILE *ff,*fin,*fin2,*fin3,*fout,*file1,*fin_amp,*fgradx,*fgrady;
  int i,j,ii,jj,iii,jjj;
  int npts_x,npts_y;
  char buff1[300],sta1[10];
  double lat,lon,lat2,lon2,t_lat,t_lon,radius,pi,sta1_lon,sta1_lat;
  int t_i,t_j;
  int marker_nn,marker_EN[2][2],marker_E,marker_N;
  double period,dist;
  int nn;
  period=atof(arg[2]);
  
  radius=6371.1391285;
  pi=4.0*atan(1.0);
  double w2;
  w2=(2*pi/period)*(2*pi/period);
  double dx,dy,x0,y0,x1,y1,temp,temp1,temp2,lat_temp, temp3, temp4,temp_amp;
//  npts_x=176;
  npts_x=atoi(arg[5]);
//  npts_y=126;
  npts_y=atoi(arg[6]);
  fprintf(stderr,"Memory check!!\n");
  double tr_t[npts_x][npts_y],amp[npts_x][npts_y],amp_gradx[npts_x][npts_y],amp_laplace[npts_x][npts_y];
  double amp_grady[npts_x][npts_y];
  double dx_km[npts_y],dy_km;
  double slow_all[npts_x][npts_y][4];
  fprintf(stderr,"Memory enough!!\n");
  
  dx=0.2;//degree
  dy=0.2;//degree
//  x0=235;
  x0=atof(arg[3]);
//  y0=25;
  y0=atof(arg[4]);
  x1=x0+(npts_x-1)*dx;
  y1=y0+(npts_y-1)*dy;
  for(j=0;j<npts_y;j++)
    {
      lat_temp=y0+j*dy;
      lat_temp=atan(0.993277*tan(lat_temp/180*pi))*180/pi;
      dx_km[j]=radius*sin((90-lat_temp)/180*pi)*dx/180*pi;
    }
    
    ////for(j=1;j<npts_y-1;j++)
    ////{
    ////  lat_temp=y0+j*dy;
    ////  lat_temp=atan(0.993277*tan(lat_temp/180*pi))*180/pi;
    ////  dx_km[j]=radius*sin((90-lat_temp)/180*pi)*dx/180*pi;
    ////}
  dy_km=radius*dy/180*pi;
  
  char event_name[300];
  
  file1=fopen(arg[1],"r");
  for(;;)
    {
      if(fscanf(file1,"%s",&event_name)==EOF)
	break;
      sprintf(buff1,"%s_am_laplace.txt.HD",event_name);
      if(access(buff1, F_OK) == 0)
	{
	  fprintf(stderr,"%s exist!! skip\n",buff1);
	  continue;
	}
      sprintf(buff1,"%s_v2_am.HD",event_name);
      if((fin=fopen(buff1,"r"))==NULL)
       {
	 cout<<buff1<<" not exist!!"<<endl;
	 return 1;
       }
      sprintf(buff1,"%s_am_gradx.txt_v2",event_name);
      if((fgradx=fopen(buff1,"w"))==NULL)
       {
	 cout<<buff1<<" cannot be written!!"<<endl;
	 return 1;
       }
      sprintf(buff1,"%s_am_grady.txt_v2",event_name);
      if((fgrady=fopen(buff1,"w"))==NULL)
       {
	 cout<<buff1<<" cannot be written!!"<<endl;
	 return 1;
       }
      
      //      sprintf(buff1,"%s.ph.txt_v2.HD_0.2",event_name);
      //if((fin2=fopen(buff1,"r"))==NULL)
      //{
      //  cout<<buff1<<" not exist!!"<<endl;
      //  return 1;
      //}
      //      sprintf(buff1,"%s_am.txt_v2.HD",event_name);
      //if((fin_amp=fopen(buff1,"r"))==NULL)
      //{
      //  cout<<buff1<<" not exist!!"<<endl;
      //  return 1;
      //}

      sprintf(buff1,"%s_am_laplace.txt.HD",event_name);
      fout=fopen(buff1,"w");
      for(i=0;i<npts_x;i++)
	{
	  for(j=0;j<npts_y;j++)
	    {
	      tr_t[i][j]=0;
	      amp[i][j]=0;
	      amp_gradx[i][j]=999999;
	      amp_grady[i][j]=999999;
	      amp_laplace[i][j]=999999;
	    }
	}
      sprintf(buff1,"%s_am",event_name);
      if((fin3=fopen(buff1,"r"))==NULL)
	{
	  cout<<buff1<<" not exist!!"<<endl;
	  return 1;
	}      
      //      fclose(fin3);
      for(;;)
	{
	  if(fscanf(fin,"%lf %lf %lf",&lon,&lat,&temp)==EOF) break;
	  //if(fscanf(fin2,"%lf %lf %lf",&lon2,&lat2,&temp2)==EOF) break;
	  //if(lon!=lon2||lat!=lat2)
	  //{
	  //  fprintf(stderr,"HD and HD_0.2 files not compatiable!!\n");
	  //  return 0;
	  // }
	  if(lon>x1+0.01||lon<x0-0.01|lat>y1+0.01||lat<y0-0.01)
	    {
	      //	      printf("grid point pass the boundary!!\n"); 
	      continue;
	    }
	  i=int((lon-x0)/dx+0.1);
	  j=int((lat-y0)/dy+0.1);
	  amp[i][j]=temp;	  
	}
      fclose(fin);
      for(;;)
	{
	  if(fscanf(fin3,"%lf %lf %lf",&lon,&lat,&temp)==EOF) break;
	  if(lon>x1+0.01||lon<x0-0.01|lat>y1+0.01||lat<y0-0.01)
	    {
	      //	      printf("grid point pass the boundary!!\n"); 
	      continue;
	    }
	  i=int((lon-x0)/dx+0.5);
	  j=int((lat-y0)/dy+0.5);
	  if(amp[i+1][j]==999999||amp[i-1][j]==999999||amp[i][j+1]==999999||amp[i][j-1]==999999)
	    {	 
	      fprintf(stderr,"amp = 999999 %s\n",event_name);
	      return 0;
	    }
	  temp1=(amp[i+1][j]-amp[i-1][j])/2.0/dx_km[j];
	  temp2=(amp[i][j+1]-amp[i][j-1])/2.0/dy_km;
    //if (dx_km[j] == 0.) {cout<<"zero "<<dx_km[j]<<" "<<j<<endl;}
	  fprintf(fgradx,"%lf %lf %lf\n",i*dx+x0,j*dy+y0,temp1);
	  fprintf(fgrady,"%lf %lf %lf\n",i*dx+x0,j*dy+y0,temp2);
	}
      fclose(fin3);
      fclose(fgradx);
      fclose(fgrady);
      sprintf(buff1,"/projects/life9360/code/NoisePy/old_helm/C_plot_travel %s_am_gradx.txt_v2 %g %g %g %g",event_name,x0,x1,y0,y1);
      system(buff1);
      sprintf(buff1,"/projects/life9360/code/NoisePy/old_helm/C_plot_travel %s_am_grady.txt_v2 %g %g %g %g",event_name,x0,x1,y0,y1);
      system(buff1);
      sprintf(buff1,"%s_am_gradx.txt_v2.HD",event_name);
      if((fgradx=fopen(buff1,"r"))==NULL)
	{
	  cout<<buff1<<" not exist!!"<<endl;
	  return 1;
	}
      sprintf(buff1,"%s_am_grady.txt_v2.HD",event_name);
      if((fgrady=fopen(buff1,"r"))==NULL)
	{
	  cout<<buff1<<" not exist!!"<<endl;
	  return 1;
	}
      for(;;)
	{
	  if(fscanf(fgradx,"%lf %lf %lf",&lon,&lat,&temp)==EOF) break;
	  if(fscanf(fgrady,"%lf %lf %lf",&lon2,&lat2,&temp2)==EOF) break;
	  if(lon!=lon2||lat!=lat2)
	    {
	      fprintf(stderr,"gradx_HD and grady_HD files not compatiable!!\n");
	      return 0;
	    }
	  if(lon>x1+0.01||lon<x0-0.01|lat>y1+0.01||lat<y0-0.01)
	    {
	      //	      printf("grid point pass the boundary!!\n"); 
	      continue;
	    }
	  i=int((lon-x0)/dx+0.1);
	  j=int((lat-y0)/dy+0.1);
	  amp_gradx[i][j]=temp;
	  amp_grady[i][j]=temp2; 
	} 
      fclose(fgradx);
      fclose(fgrady);
      for(j=0;j<npts_y;j++)
	{
	  for(i=0;i<npts_x;i++)
	    {
              if (i==0 || i==npts_x || j==0 || j == npts_y)  {
                fprintf(fout,"%lf %lf -999\n",x0+i*dx,y0+j*dy);
                continue;
                }
	      if(amp[i+1][j]==999999||amp[i-1][j]==999999||amp[i][j+1]==999999||amp[i][j-1]==999999)
		{	 
		  fprintf(stderr,"amp = 999999 %s\n",event_name);
		  return 0;
		}
	      temp1=(amp_gradx[i+1][j]-amp_gradx[i-1][j])/2.0/dx_km[j];
	      temp2=(amp_grady[i][j+1]-amp_grady[i][j-1])/2.0/dy_km;
	      //temp=(temp1+temp2)/amp[i][j]/4/pi/pi*period*period;
        temp=(temp1+temp2);
        //cout<<"temp1 "<<temp<<" j "<<j<<endl;
	      //  temp=temp2/amp[i][j]/4/pi/pi*period*period;
	      fprintf(fout,"%lf %lf %lf\n",x0+i*dx,y0+j*dy,temp);
	    } 
	}
      fclose(fout);
    }
  fclose(file1);
  return 0;
}
