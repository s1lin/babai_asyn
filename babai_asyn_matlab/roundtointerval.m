function [x,c]=roundtointerval(x,lb,ub)
%round x into lb and ub to compute the VER and BER
n=size(x);
c=0;
for i=1:n
    if x(i)>ub
        x(i)=ub;
        c=c+1;
    else
        if x(i)<lb
            x(i)=lb;
            c=c+1;
        end
    end
end
           