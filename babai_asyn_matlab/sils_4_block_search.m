function x=sils_4_block_search(Rb,yb)
%BOB 4-block

[m,n]=size(Rb);
q=n/2;
yb(q+1:n)'
xx1=sils_2_block_search(Rb(q+1:n,q+1:n),yb(q+1:n));
yb2=yb(1:q)-Rb(1:q,q+1:n)*xx1;
yb2
xx2=sils_2_block_search(Rb(1:q,1:q),yb2);   
x=[xx2;xx1];