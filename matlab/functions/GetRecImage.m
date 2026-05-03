function [S]=GetRecImage(Ad,RecFilter,m1,n1)
[m,n]=size(RecFilter);
K1=m1+n-1;
K2=n1+n-1;
a=zeros(K1,K2);
S=0;
for i=1:m
    for j=1:m
    Fi=RecFilter(i,end:-1:1);
    Fj=RecFilter(j,end:-1:1);
    a(1:m:end,1:m:end)=Ad(:,:,(i-1)*m+j);
    S=S+convn(convn(Fi',a),Fj);
    end
end
S=S(n:end-n+1,n:end-n+1);
end
