% You can add your path here and comment out the next line
% addpath(genpath('C:.....'))
%% Load Image
load Indian
[nx1,ny1,nz1]=size(Indian);
RH=reshape(Indian,nx1*ny1,nz1)';
m = min(Indian(:));
M = max(Indian(:));
Y_true=(RH-m)/(M-m);
Indian=reshape(Y_true',nx1,ny1,nz1);
%%
L=5;%level of decompositions
NFC=10;%number of filter coefficients
qmf = daubcqf(NFC,'min');%wavelet filter
t_max=4;   % maximum search interval
n=10; % search interval number
t1=linspace(0,t_max,n);
options.noisest=0;
options.scaling=1;
options.type='FORPDNT';
options.WaveletToolbox='WaveLab_Fast';
%% 
M=20;
stepsize=8;
noise_type = 'additive';  
Y_HyRes = HyRes(Indian);
[Y_fasthyde, time_fasthyde] = FastHyDe(Indian,  noise_type, 1, 10);
Yrec_SSTV=funSSTV(Indian,40,.1,.2,.2);
[R_NAIRLMA,~,~] =NAIRLMA_denosing(Indian,Indian,M,stepsize,1);
[Wavelet3D,~,~,~]=Model_1(Indian,t1);
[~,~,~,R_FORPDN]=FOSRPDN_SURE(Indian,[0,0,0,0,0,0],[.001,.001,.001,.01,.1,1],[.01,.01,.01,.1,1,10],qmf,options);
%% Showing Denoised data
BN=1; % Number of band
figure(1)
subplot(2,4,1),imagesc(Indian(:,:,1));colormap(gray);axis image;axis off;title(['Band', num2str(BN)]);
subplot(2,4,2),imagesc(Wavelet3D(:,:,1));colormap(gray);axis image;axis off;title('3D Wavelet');
subplot(2,4,3),imagesc(R_FORPDN(:,:,1));colormap(gray);axis image;axis off;title('FORPDN');
subplot(2,4,4),imagesc(Yrec_SSTV(:,:,1));colormap(gray);axis image;axis off;title('SSTV');
subplot(2,4,5),imagesc(R_NAIRLMA(:,:,1));colormap(gray);axis image;axis off;title('NAIRLMA');
subplot(2,4,6),imagesc(Y_HyRes(:,:,1));colormap(gray);axis image;axis off;title('HyRes');
%subplot(2,4,7),imagesc(out_avg_np(:,:,1));colormap(gray);axis image;axis off;title('HSI-DIP');
subplot(2,4,8),imagesc(Y_fasthyde(:,:,1));colormap(gray);axis image;axis off;title('FastHyDe');


