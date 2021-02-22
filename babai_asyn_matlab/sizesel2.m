function d= sizesel2(left,right,tr,prebabai,predignol,noise)
%block size selection for BOB, tr is the treshold
%improved verision
%calll it with left = 1, right =n, 
% tr- tresh hold
% [prebabai, predignol] = precomputation before call this function

if right==left
   d=1;
else
    mid = floor((left+right)/2);
    PB1= prebabai(mid+1)/prebabai(left);
    PB2= prebabai(right+1)/prebabai(mid+1);
    if PB1*PB2>=tr
        d = ones(right-left+1,1);
    else 
        if mid~=left
            PB1 = max(PB1,pest3(left,mid,predignol,noise));
        end
        if mid+1~=right
            PB2 = max(PB2,pest3(mid+1,right,predignol,noise));
        end
        if PB1*PB2<=tr
            d = right-left+1;
        else
            w1 = sqrt(tr*PB1/PB2);
            w2 = sqrt(tr*PB2/PB1);
            d = [sizesel2(left,mid,w1,prebabai,predignol,noise);sizesel2(mid+1,right,w2,prebabai,predignol,noise) ];
        end
        
        
        
    end
end