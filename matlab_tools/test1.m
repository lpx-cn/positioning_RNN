%Ðý×ª¾ØÕó
clear all;clc
a=[0,0,1]'

alpha=pi/3
beta=0
gamma=0

R_x=[1,0,0;
     0,cos(alpha),-sin(alpha);
     0,sin(alpha),cos(alpha)];
 R_y=[cos(beta),0,sin(beta);
     0,1,0;
     -sin(beta),0,cos(beta)];
 R_z=[cos(gamma),-sin(gamma),0;
     sin(gamma),cos(gamma),0;
     0,0,1];
 b=R_x*R_y*R_z*a
 
alpha2=-alpha
beta2=-beta
gamma2=-gamma

r_x=[1,0,0;
     0,cos(alpha2),-sin(alpha2);
     0,sin(alpha2),cos(alpha2)];
r_y=[cos(beta2),0,sin(beta2);
     0,1,0;
     -sin(beta2),0,cos(beta2)];
r_z=[cos(gamma2),-sin(gamma2),0;
     sin(gamma2),cos(gamma2),0;
     0,0,1];
 
 a=r_z*r_y*r_x*b