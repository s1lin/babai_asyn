
function y = callmyabsval()  
    %#codegen
    % Check the target. Do not use coder.ceval if callmyabsval is
    % executing in MATLAB
    A = randn(2048);

    if coder.target('MATLAB')
      % Executing in MATLAB, call function myabsval
      y = qrtest(A);
      display('here')
    else
      % add the required include statements to generated function code
      coder.updateBuildInfo('addIncludePaths','$(START_DIR)\codegen\lib\qrtest');
      coder.cinclude('qrtest.h');

      % Call the generated C library function myabsval
      y = coder.ceval('qrtest',A);

      % Call the terminate function after
      % calling the C function for the last time
      coder.ceval('myabsval_terminate');
    end
end
