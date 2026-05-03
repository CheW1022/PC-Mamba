

clear;
% load data
kdata = load('kdata.mat');  % K-space data
kdata = kdata.kspace;
csm = load('csm.mat');  % % Prescan data
csm = csm.csm;

% Estimated coil sensitivity maps
[ro,pe,Nc,Nint] = size(kdata);
ncalib = 24;
ksize = [6,6];
calib = crop(csm,[ncalib,ncalib,Nc]);
eigThresh_1 = 0.01;
eigThresh_2 = 0.7;
[k,S] = dat2Kernel(calib,ksize);
idx = max(find(S >= S(1)*eigThresh_1));
[M,W] = kernelEig(k(:,:,:,1:idx),[ro,pe]);
maps = M(:,:,:,end).*repmat(W(:,:,end)>eigThresh_2,[1,1,Nc]); 

% Structured low-rank matrix completion
mask = squeeze(sum(abs(kdata),3))>0;
[recon] = mussels(kdata,maps,mask,[4,4],2,2,.01,1,0);
recon = sum(ifft2c(recon),3);
figure();imshow(recon,[]);


