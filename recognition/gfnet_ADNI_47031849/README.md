# COMP3710 Pattern Recognition and Analysis  
## Alzheimer’s Disease Classifier using ADNI Brain Data  
### Model Chosen: GFNet  

The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.
It should also list any dependencies required, including versions and address reproduciblility of results, if applicable
provide example inputs, outputs and plots of your algorithm
Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
Include a conclusion, future works,  - many people lost marks due to this

(Include a small summary of the task and the result, include GPU type)

#### Introduction  
(Short intro to what Alzhiemers is: cause, symptoms, treatement (reference))  
(Intro to dataset: folder size, image size, test vs train set, how to id alzheimers?, example images)  
(Intro to GFNet: what is it, how do we use to solve task?)

(Intro to GFNet: How does it work, layers, include GIF)

#### Dependencies  

- Python: 3.12.12
- Pytorch: 2.8.0  
- TorchVision: 0.23.0
- Numpy: 2.0.2  
- Scikit-learn: 1.6.1 
- Timm: 1.0.20 
- Tqdm: 4.67.1 
- Matplotlib: 3.10.0 
- PIL: 11.3.0
- cv2: 4.12.0

#### Reproducing Results  
Within the train.py file, we set a seed to all of the random elements within the code. 
This ensures that we can reproduce the same result when re-running the code or when running 
on a different machine. 
  
```
def set_seed(seed: int):
    """
    Sets the seed of the random elements of the code in order to maintain consistency between trainings
    """
    torch.manual_seed(seed)                     # sets the seed for RNG on device
    torch.cuda.manual_seed(seed)                # sets the seed for current GPU
    torch.cuda.manual_seed_all(seed)            # sets the seed for all GPUs
    np.random.seed(seed)                        # sets the seed for Numpy library
    random.seed(seed)                           # sets the seed for Random library
    torch.backends.cudnn.deterministic = True   # sets cuDNN to only use deterministic convolution
    torch.backends.cudnn.benchmark = False      # sets cuDNN to not perform benchmarking
```

#### Pre-processing Data  
(Cropping & resizing) (Transforms here?)

#### Datasets
(ADNITrain vs ADNITest)
(BCEWithLogitsLoss, AdamW, CosineAnnealingLR)

#### Transforms
(Define each transform, transform helps with overfitting)
(calculating mean and STD values)


#### Results  


#### Usage


#### Conclusion 

dataset.py  
Resizing image, Otsu's threshold method, resize dimension (210), Lancoz interpolation  

train.py  
epochs, early stopping, BCEWithLogitsLoss, AdamW, CosineAnnealingLR, 
scheduler
