function [ img ] = invHankel( H,sx,sy,nc,ksize )
    tsx=size(H,1);
    tsy=size(H,2)/nc;
    H2 = reshape(H,tsx,tsy,nc);
img = row2im_tpose(H2,[sx,sy,nc],ksize);
end

