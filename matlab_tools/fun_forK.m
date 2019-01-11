function [F dF] = fun(K,P)
    % define nonlinear equations as following
    % variables 
    % functions F1 F2 F3 
    syms k1 k2 k3

    a1=P(1);a2=P(2);a3=P(3);b=P(4);c=P(5);d=P(6);e1=P(7);e2=P(8);e3=P(9);
    
    F1 = a1*k1^2+a2*k2^2-2*b*k1*k2-e1;
    F2 = a2*k2^2+a3*k3^2-2*c*k2*k3-e2;
    F3 = a3*k3^2+a1*k1^2-2*d*k1*k3-e3;

    
    F = [F1;F2;F3];
    dF = jacobian(F,[k1 k2 k3]);
    
    [k1 k2 k3]=deal(K(1),K(2),K(3));
    F=subs(F);
    dF=subs(dF);
   
end