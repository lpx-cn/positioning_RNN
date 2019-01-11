%�ɽ��ࡢ�����ܶȡ�led���ꡢ���λ�õõ���Ӧ���λ��
function image_point=get_image_point_coordinate(f,density,LED_WC,camera_point,Z_dir,X_dir)

%%%%%%%%%%%%%%%%%%%%%%% ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;clc;
% f=25;
% density=36/1024;
% LED_WC=[3.5,3.5,5]'*1000;
% camera_point=[0,0,f]';
% Z_dir=[0,0.6,0.8]';
% X_dir=[1,0,0]';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_LED=size(LED_WC);
N_LED=N_LED(1);
Y_dir=cross(Z_dir,X_dir);
Rotate=[X_dir,Y_dir,Z_dir];      %WCS=��ת����*CCS ��=��ת*��
Translate=[Rotate,camera_point;  %WCS=�任����*CCS ��=�任*��
            0,0,0,1];
%��һCCS��        
LED_CC=inv(Translate)*[LED_WC;1];
LED_CC=LED_CC([1,2,3]);
t=-f/LED_CC(3);
image_point_real=t*LED_CC;

% %����WCS��
% P_WC=(Translate)*[0,0,-f,1]'
% P_WC=P_WC([1,2,3])
% 
% t2=dot(Z_dir,(P_WC-LED_WC))/dot(Z_dir,(camera_point-LED_WC))
% image_point2=t2*(camera_point-LED_WC)+LED_WC
% image_point2=inv(Translate)*[image_point2;1]
% image_point2=image_point2([1,2,3])

image_point=image_point_real;
%%%%%%%%%%%%%%%%%%%%%%%% �������� %%%%%%%%%%%%%%%%%%%%%%%%
% for j=1:2  %��������
%     if mod(image_point_real(j),density)<density/2
%         image_point(j)=image_point_real(j)-mod(image_point_real(j),density);
%     else
%         image_point(j)=image_point_real(j)-mod(image_point_real(j),density)+density;
%     end
% end
% image_point(3)=image_point_real(3);
%   
%   
