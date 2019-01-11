function Pw=fun_position(Lw1,Lw2,Lw3,Ic1,Ic2,Ic3,f)
%调用了fun_forK 牛顿迭代来计算K

Pc=[0 0 0];  %相机坐标系中镜头的位置

a1=sum(Ic1.*Ic1);
a2=sum(Ic2.*Ic2);
a3=sum(Ic3.*Ic3);
b=sum(Ic1.*Ic2);
c=sum(Ic2.*Ic3);
d=sum(Ic1.*Ic3);
e1=sum((Lw1-Lw2).^2);
e2=sum((Lw2-Lw3).^2);
e3=sum((Lw1-Lw3).^2);

P0 = [a1 a2 a3 b c d e1 e2 e3]; %参数传递
K0 = [1,1,1]';%初始化参数

eps = 0.000001;
i=0;
M=1000;

while (i<M)
    [F dF]=fun_forK(K0,P0);
    if norm(F) < eps
        %i
        break;
    end
    K = K0 - pinv(double(dF))*double(F);
    K0 = K;
    i=i+1;
end

if i==(M) && (norm(F) > eps)
    disp('求解错误');
    [Lw1,Lw2,Lw3,Ic1,Ic2,Ic3,f]
    err=eval(norm(F))
end
% F
% err=eval(norm(F))

Lc1=-K(1)*Ic1;  %用已知的实际位置矢量来定位镜头位置
Lc2=-K(2)*Ic2;
Lc3=-K(3)*Ic3;
Tc=[(Lc2-Lc1);
   (Lc3-Lc2);
   cross((Lc2-Lc1),(Lc3-Lc2)) ];
Coe=Lc1*inv(Tc);

Tw=[(Lw2-Lw1);
   (Lw3-Lw2);
   cross((Lw2-Lw1),(Lw3-Lw2))];
Pw=Lw1-Coe*Tw;

end
