function parallel_test(A, n)
    Acpu=A(:,:,1:n/2);                  %chunk #1 : send to CPU
    Agpu=gpuArray(A(:,:,n/2+1:end));    %chunk #2 : send to GPU with device index 1
    p=parpool(2);
    F(1)=parFeval(p, @deploy,2,Acpu);
    F(2)=parFeval(p, @deploy,2,Agpu,1);
    [Q,R] = fetchOutputs(F,'UniformOutput',false); % Blocks until complete
    Q=cat(3,Q{1},gather(Q{2}))
    R=cat(3,R{1},gather(R{2}))
end

function [q,r]=deploy(a,Id)
     if nargin>1, gpuDevice(Id);end
     for i=size(a,3):-1:1  
         [q(:,:,i),r(:,:,i)]=qr(A(:,:,i));
     end
 
end