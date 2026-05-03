clear;
% load data
load T2.mat
[g,h]=wfilters('db3'); % filter
[g1,h1]=wfilters('db3'); % filter

% Decomposition and reconstruction
DecFilter= GetNewMultiWavelet(g,h,g1,h1)'; % new filter
[H,W]=size(T2);
[dec]=GetDecImageFast(T2,DecFilter); % Decomposition
[recon]=GetRecImage(dec,DecFilter,H,W); % Reconstruction

% show 
figure();imagesc(T2*100,[0 200]);axis image;colormap jet; 
for i = 1:16
    img{i} = dec(:,:,i);
end
figure('Position',[100,100,1000,800]);
for i = 1:16
    subplot(4,4,i);     
    imagesc(img{i});
    axis image off;   
    colormap gray;    
end
figure();imagesc(recon*100,[0 200]);axis image;colormap jet; 

% error
mae_value = (mean(abs(T2(:) - recon(:))))

