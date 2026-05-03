function [Ad]=GetDecImageFast(A,DecFilter)
% 构建Ad矩阵  A是图像  
[m,n]=size(DecFilter);
[m1,n1]=size(A);
K1=1:m1+n-1;
K2=1:n1+n-1;
L1=length(K1(1:m:end));
L2=length(K2(1:m:end));
Ad=zeros(L1*L2*m*m,1);
Ad=reshape(Ad,L1,L2,m*m);
%卷积
for i=1:m
    Fi=DecFilter(i,:);
    Ci=convn(A,Fi');   % % %
    Ci=Ci(1:m:end,:);
    for j=1:m
        Fj=DecFilter(j,:);
        a=convn(Ci,Fj);
        Ad(:,:,(i-1)*m+j)=a(:,1:m:end);
    end
end
end
