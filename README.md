# Hyperspectral-Image-Denoising-Toolbox-V2
This toolbox contains the following HSI denoising methods: 3D Wavelets,FORPDN, and HyRes (all can be found in
[Hyperspectral Image Denoising Matlab Toolbox](https://www.researchgate.net/publication/328027880_Hyperspectral_Image_Denoising_Matlab_Toolbox)),
[SSTV](https://www.mathworks.com/matlabcentral/fileexchange/49145-mixed-noise-reduction),
[NAIRLMA](https://sites.google.com/site/rshewei/home),
[FastHyDe](https://github.com/LinaZhuang/FastHyDe_FastHyIn),
[DIP](https://github.com/DmitryUlyanov/deep-image-prior), and [SDeCNN](https://github.com/mhaut/HSI-SDeCNN). For SDeCNN, after downloading the codes only replace the Indian.mat which has 206 bands with the one that we used with 220 bands (The link is given below).
For DIP, you need to download the original toolbox and copy the python file (DN4Rev_Real.py) into the master folder (change the path, line 48 of DN4Rev_Real.py, to the mat file of Indian Pine) and run. Addtionally, you need to install the dependencies (see [readme](https://github.com/DmitryUlyanov/deep-image-prior)) 



To reproduce the resutls below you can find the Indian Pines dataset in [Hyperspectral Image Denoising Matlab Toolbox](https://www.researchgate.net/publication/328027880_Hyperspectral_Image_Denoising_Matlab_Toolbox) and use the matlab and python demos provided to run the codes

![image](https://user-images.githubusercontent.com/61419984/125260332-88d38200-e300-11eb-90a6-90f2285135aa.png)

