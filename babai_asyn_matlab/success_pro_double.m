function p=success_pro_double(R,noise)
%SP of the double layer Sphere decoding
%R in 2*2 upper triagular
%noise - sigma
r1=R(1,1);
r2=R(1,2);
r3=R(2,2);
p=0;
r1=abs(r1);
r2=abs(r2);
r3=abs(r3);
ymax1 = @(x) ((r2)^2/r3-2*r2/r3*x+r3)/2;
 ymax2 =@(x) ((r1-r2)^2/r3-2*(r2/r3-r1/r3)*x+r3)/2;
fun=@(x,y) 1/sqrt((2*pi)^2)*exp(-0.5*(x.^2+y.^2)/noise^2)/noise^2 ;
p=p+integral2(fun,(2*r2-r1)/2,abs(r1)/2,0,ymax1,'AbsTol',0,'RelTol',1e-10)*2;
p=p+integral2(fun,-abs(r1)/2,(2*r2-r1)/2,0,ymax2,'AbsTol',0,'RelTol',1e-10)*2;
