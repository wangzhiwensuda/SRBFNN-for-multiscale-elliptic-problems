# SRBFNN for multiscale elliptic problems

 

This is the code for the paper: solving multiscale elliptic problems by sparse radial
basis function neural networks
<h3>1. Requirements</h3>

(Our experiment environment for reference)

- Python 3.7

- Pytorch 1.7.1




 <h3>2. Folders</h3>
The folder of SRBFNN includes three subfolders, which respresent the 
experiments in different dimensions. The folder of SRBFNN-compare 
contains the experiments using comparing methods, DGM,DRM,PINN,MscaleDNN.

<h3>3. Instructions</h3>

<h4>3.1 generating the FDM solution</h4>

```bash
cd SRBFNN/1d/
python 1d_FDM.py
```
<h4>3.2 estimating the slope using least squares method</h4>
```bash
cd SRBFNN/1d/
python tool.py
```
<h4>3.3 running the examples</h4>
```bash
cd SRBFNN/1d/
python 1d_example.py --eps=0.5 --N=100 
```




