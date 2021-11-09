# Analyzing Visual Attention Mechanisms for Handwritten Digit Classification

<p align="center">
  <img src="https://github.com/biswassanket/STN_FGC/blob/master/files/output.png">
  <br>
  <br>
  <b><i>Visual Attention Analysis with <a href="https://arxiv.org/pdf/1506.02025.pdf">Spatial Transformer Networks</a> for Handwritten Digit Classification on MNIST</i></b>
</p>


<p align="center">
  
### Getting Started
  
#### Step 1: Clone this repository and change directory to repository root
```bash
git clone https://github.com/biswassanket/STN_FGC.git
cd STN_FGC
```
#### Step 2: Create a conda environment to run the above project and install required dependencies.
* To create **conda** environment: `conda env create -f environment.yml`
  
#### Step 3: Activate the conda environment 
```bash
conda activate stn_fgc
```
#### Step 4: Training STN Models on MNIST 
* To run base STN with standard Conv layers:
  
```bash
$ python main.py --stn
```
* To run STN with Coordconv layers:

```bash
$ python main.py --stncoordconv --localization
```
  
#### Step 5: Training ViT Model on MNIST 
  
```bash
$ python main.py --vit
``` 
  
#### Step 6: For the detailed analysis on the experimented visual attention models, here is the complete <a href="https://github.com/biswassanket/STN_FGC/blob/master/demo.ipynb"> report </a>

### Results 
  
| Model Variant | Accuracy | Best Epoch |
| --- | --- | --- |
| Simple Conv     | 0.9879 | 48 |
| Simple STN+Conv | 0.9889 | 44 |
| Simple STN+CoordConv| 0.9850| 43 |
| Simple STN+CoordConv+localization| 0.9910 | 47 |
| Simple STN=CoordConv+localization+r-channel| 0.9868 | 40 |
| Vision Transformers| 0.9844 | 49|

### Authors
* [Sanket Biswas](https://github.com/biswassanket)
  
### Conclusion
  
Enjoyed playing with the models. Stay tuned, more implementations of visual attention models on fine-grained image classification task is coming soon. 
Thank you and sorry for the bugs,as usual.
  
  
  
  
  
