# COMP3710 Pattern Recognition and Analysis  
## Alzheimer’s Disease Classifier using ADNI Brain Data  
### Model Chosen: GFNet  

The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.
It should also list any dependencies required, including versions and address reproduciblility of results, if applicable
provide example inputs, outputs and plots of your algorithm
Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
Include a conclusion, future works,  - many people lost marks due to this

A GFNet model was designed and trained to classify Alzheimer’s in MRI brain scan images, with training and testing data provided by the [ADNI dataset](https://adni.loni.usc.edu/). 
The model was trained on the Google Colab A100 GPU, and managed to to have an 80.00% accuracy on the test set.  

### Introduction  
(Short intro to what Alzhiemers is: cause, symptoms, treatement (reference))  
Alzheimer’s Disease is a type of dementia that affects memory, thinking and behaviour of the affected patient [1][1]. 
It typically shrinks the brain, kills neurons and a buildup of plaque can accumulate in areas such as the hippocampus [2][2].  
![Normal Brain vs Advanced Alzheimer's Visual](resources/alzheimers_vs_typical_drawing.avif "Normal Brain vs Advanced Alzheimer's Visual") [2][2]  


(Intro to dataset: folder size, image size, test vs train set, how to id alzheimers?, example images)  

The ADNI dataset consists of 21525 images in the training set (10401 AD, 11124 NC) and 9000 images in the testing dataset (4460 AD, 4546 NC). 
6 images wihin the NC testing dataset were deleted as they were duplicates and cause issues with the testing data. These images could've been an error 
from when data was being transferred locally.  
![Neurotypical Brain Image from NC training data](resources/808819_88_NC_train.jpeg "Neurotypical Brain Image from NC training data")  
![Alzheimer Affected Brain Image from AD training data](resources/218391_78_AD_train.jpeg "Alzheimer Affected Brain Image from AD training data")  
As you can see from the images, it can be difficult to determine which image has 

(Intro to GFNet: what is it, how do we use to solve task?)

(Intro to GFNet: How does it work, layers, include GIF)

### Dependencies  

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

### Reproducing Results  
Within the train.py file, we set a seed to all of the random elements within the code. 
This ensures that we can reproduce the same result/model when re-running the code or when running 
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

### Pre-processing Data  
(Cropping & resizing)
(Define each transform, transform helps with overfitting)
(calculating mean and STD values)

### Datasets
(ADNITrain vs ADNITest)
(BCEWithLogitsLoss, AdamW, CosineAnnealingLR)

### Results  
80.00% accuracy on test set
(early stopping)

We first set MAX_EPOCHS to 100 to find where the model started overfitting
![Finding Ideal Epoch - Accuracy](resources/acc_vs_epoch_100_seed1.png "Finding Ideal Epoch - Accuracy")
![Finding Ideal Epoch - Loss](resources/loss_vs_epoch_100_seed1.png "Finding Ideal Epoch - Loss")  
As seen from the figures, the performance of the model peaks at around 60 epochs.  



![Confusion Matrix](resources/confusion_matrix.png "Confusion Matrix")  

### Usage
For training the model using train.py:  
```
usage: train.py [-h] [-dp TRAINPATH] [-sp SAVEPATH] [-s SEED]

options:
  -h, --help            show this help message and exit
  -dp TRAINPATH, --trainpath TRAINPATH
                        Filepath to ADNI training dataset
  -sp SAVEPATH, --savepath SAVEPATH
                        Filepath to saved elements
  -s SEED, --seed SEED  Seed for reproducibility
```  

For predicting on the test dataset using predict.py:  
```
usage: predict.py [-h] [-dp TESTPATH] [-sp SAVEPATH]

options:
  -h, --help            show this help message and exit
  -dp TESTPATH, --testpath TESTPATH
                        Filepath to ADNI testing dataset
  -sp SAVEPATH, --savepath SAVEPATH
                        Filepath to saved elements
```  

### Conclusion 



### References  
[1]: <https://www.alz.org/alzheimers-dementia/what-is-alzheimers>  
[2]: <https://www.medicalnewstoday.com/articles/alzheimers-brain-vs-normal-brain#alzheimers-brain>  

(look to changing mean and std)
(experiment with other transforms)
(try training with patients rather than images? -> may need more data)

dataset.py  
Resizing image, Otsu's threshold method, resize dimension (210), Lancoz interpolation  

train.py  
epochs, early stopping, BCEWithLogitsLoss, AdamW, CosineAnnealingLR, 
scheduler
