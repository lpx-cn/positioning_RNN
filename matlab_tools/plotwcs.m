function plotwcs(f,LED_WC,image_point,camera_point,Z_dir,X_dir)


Lw1=LED_WC(1,:);       %世界坐标系灯的位置
Lw2=LED_WC(2,:);
Lw3=LED_WC(3,:);

Y_dir=cross(Z_dir,X_dir);

Rotate=[X_dir,Y_dir,Z_dir];      %WCS=旋转矩阵*CCS 旧=旋转*新
Translate=[Rotate,camera_point;  %WCS=变换矩阵*CCS 旧=变换*新
            0,0,0,1];
        
Pc=[0 0 0];  %相机坐标系中镜头的位置

Ic1=image_point(1,:); %相机坐标系中像点的位置
Ic2=image_point(2,:);
Ic3=image_point(3,:);

Pw=camera_point;

figure
grid on
hold on

iw=Pw(1);jw=Pw(2);kw=Pw(3);

Iw1=Translate*[Ic1,1]'; %求像点的wcs坐标
Iw2=Translate*[Ic2,1]';
Iw3=Translate*[Ic3,1]';

Iw1=Iw1([1,2,3]); 
Iw2=Iw2([1,2,3]);
Iw3=Iw3([1,2,3]);

plot3(iw,jw,kw,'ro','markersize',10);%相机位置
text(iw,jw,kw,['相机位置','(',num2str(iw),',',num2str(jw),',',num2str(kw),')']);
text(Iw1(1),Iw1(2),Iw1(3),['像1','(',num2str(Iw1(1)),',',num2str(Iw1(2)),',',num2str(Iw1(3)),')']);
text(Iw2(1),Iw2(2),Iw2(3),['像2','(',num2str(Iw2(1)),',',num2str(Iw2(2)),',',num2str(Iw2(3)),')']);
text(Iw3(1),Iw3(2),Iw3(3),['像3','(',num2str(Iw3(1)),',',num2str(Iw3(2)),',',num2str(Iw3(3)),')']);
text(Lw1(1),Lw1(2),Lw1(3),['原1','(',num2str(Lw1(1)),',',num2str(Lw1(2)),',',num2str(Lw1(3)),')']);
text(Lw2(1),Lw2(2),Lw2(3),['原2','(',num2str(Lw2(1)),',',num2str(Lw2(2)),',',num2str(Lw2(3)),')']);
text(Lw3(1),Lw3(2),Lw3(3),['原3','(',num2str(Lw3(1)),',',num2str(Lw3(2)),',',num2str(Lw3(3)),')']);


X=[Iw1(1),Iw2(1),Iw3(1),Iw1(1)];%像点实际3d位置
Y=[Iw1(2),Iw2(2),Iw3(2),Iw1(2)];
Z=[Iw1(3),Iw2(3),Iw3(3),Iw1(3)];
plot3(X,Y,Z,X,Y,Z,'b.','markersize',25);

X1=[Lw1(1),Lw2(1),Lw3(1),Lw1(1)];%物体实际3d位置
Y1=[Lw1(2),Lw2(2),Lw3(2),Lw1(2)];
Z1=[Lw1(3),Lw2(3),Lw3(3),Lw1(3)];  
plot3(X1,Y1,Z1,X1,Y1,Z1,'bx','markersize',15);

plot3([Iw1(1),Lw1(1)],[Iw1(2),Lw1(2)],[Iw1(3),Lw1(3)],'k:','LineWidth',2); %三条直线连线
plot3([Iw2(1),Lw2(1)],[Iw2(2),Lw2(2)],[Iw2(3),Lw2(3)],'k:','LineWidth',2);
plot3([Iw3(1),Lw3(1)],[Iw3(2),Lw3(2)],[Iw3(3),Lw3(3)],'k:','LineWidth',2);

plot3([iw,Lw1(1)],[jw,Lw1(2)],[kw,Lw1(3)],'y--'); %相机和原物体连线
plot3([iw,Lw2(1)],[jw,Lw2(2)],[kw,Lw2(3)],'y--');
plot3([iw,Lw3(1)],[jw,Lw3(2)],[kw,Lw3(3)],'y--');

xlabel('x'),ylabel('y'),zlabel('z');
axis xy


% quiver3(Z(:,1),Z(:,2),Z(:,3),CCS(:,1),CCS(:,2),CCS(:,3),1,'Linewidth',2)%矢量图
CCS=[X_dir';
    Y_dir';
    Z_dir']; %CCS坐标轴在WCS中的坐标
Pp=camera_point;  %底片中心在WCS中的坐标
CCS=f*CCS; %放大合适倍数

quiver3(Pp(1),Pp(2),Pp(3),CCS(1,1),CCS(1,2),CCS(1,3),1,'r','Linewidth',3)
quiver3(Pp(1),Pp(2),Pp(3),CCS(2,1),CCS(2,2),CCS(2,3),1,'y','Linewidth',3)
quiver3(Pp(1),Pp(2),Pp(3),CCS(3,1),CCS(3,2),CCS(3,3),1,'b','Linewidth',3)


