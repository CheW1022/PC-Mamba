function [ H ] = getHankel( img,ncalib )
tmp=im2row(img,ncalib); [tsx,tsy,tsz] = size(tmp);
H = (reshape(tmp,tsx,tsy*tsz));
end

