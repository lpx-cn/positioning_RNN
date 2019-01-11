%由焦距、像素密度、led坐标、相机位置得到对应像点位置
function image_point=get_image_point_coordinate(f,density,LED_WC,camera_point,Z_dir,X_dir)

%%%%%%%%%%%%%%%%%%%%%%% 测试 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
Rotate=[X_dir,Y_dir,Z_dir];      %WCS=旋转矩阵*CCS 旧=旋转*新
Translate=[Rotate,camera_point;  %WCS=变换矩阵*CCS 旧=变换*新
            0,0,0,1];
%法一CCS：        
LED_CC=inv(Translate)*[LED_WC;1];
LED_CC=LED_CC([1,2,3]);
t=-f/LED_CC(3);
image_point_real=t*LED_CC;

% %法二WCS：
% P_WC=(Translate)*[0,0,-f,1]'
% P_WC=P_WC([1,2,3])
% 
% t2=dot(Z_dir,(P_WC-LED_WC))/dot(Z_dir,(camera_point-LED_WC))
% image_point2=t2*(camera_point-LED_WC)+LED_WC
% image_point2=inv(Translate)*[image_point2;1]
% image_point2=image_point2([1,2,3])

image_point=image_point_real;
%%%%%%%%%%%%%%%%%%%%%%%% 量化过程 %%%%%%%%%%%%%%%%%%%%%%%%
% for j=1:2  %量化过程
%     if mod(image_point_real(j),density)<density/2
%         image_point(j)=image_point_real(j)-mod(image_point_real(j),density);
%     else
%         image_point(j)=image_point_real(j)-mod(image_point_real(j),density)+density;
%     end
% end
% image_point(3)=image_point_real(3);
%   
%   
