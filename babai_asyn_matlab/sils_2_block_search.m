function x=sils_2_block_search(Rb,yb)
%BOB 2-block
[m,n]=size(Rb);
q=n/2;
xx1=sils_search(Rb(q+1:n,q+1:n),yb(q+1:n),1);
yb2=yb(1:q)-Rb(1:q,q+1:n)*xx1;
xx2=sils_search(Rb(1:q,1:q),yb2,1);   
x=[xx2;xx1];