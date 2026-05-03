function [ng]=OtherMakeNewMultiWavelet(ng,g1,h1)
     [N,M]=size(ng);
     gg=zeros(2*N-1,M);
     gg(1:2:end,:)=ng;
     ng1=convn(gg,g1(:));
     ng2=convn(gg,h1(:));
     ng=[ng1,ng2];
end