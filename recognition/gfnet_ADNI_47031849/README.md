# COMP3710 Pattern Recognition and Analysis  
## Alzheimer’s Disease Classifier using ADNI Brain Data  
### Model Chosen: GFNet  
A GFNet model was designed and trained to classify Alzheimer’s in MRI brain scan images, with training and testing data provided by the [ADNI dataset](https://adni.loni.usc.edu/). 
The model was trained on the Google Colab A100 GPU, and managed to to have an 80.00% accuracy on the test set.  

### Introduction  
Alzheimer’s Disease is a type of dementia that affects memory, thinking and behaviour of the affected patient[^1]. 
It typically shrinks the brain, kills neurons and a buildup of plaque can accumulate in areas such as the hippocampus[^2].  
![Normal Brain vs Advanced Alzheimer's Visual](resources/alzheimers_vs_typical_drawing.avif "Normal Brain vs Advanced Alzheimer's Visual")[^2] 
**Normal Brain vs Advanced Alzheimer's Visual**  

The ADNI dataset consists of 21525 images in the training set (10401 AD, 11124 NC) and 9000 images in the testing dataset (4460 AD, 4546 NC). 
6 images wihin the NC testing dataset were deleted as they were duplicates and cause issues with the testing data. These images could've been an error 
from when data was being transferred locally.  

| ![Neurotypical Brain Image from NC training data](resources/808819_88_NC_train.jpeg "Neurotypical Brain Image from NC training data") | ![Alzheimer Affected Brain Image from AD training data](resources/218391_78_AD_train.jpeg "Alzheimer Affected Brain Image from AD training data") |
| ----- | ----- |
| **Neurotypical Brain Image from NC training data** | **Alzheimer Affected Brain Image from AD training data** | 

As you can see from the images, it can be difficult and tedious to determine which brain has been affected by the disease. In situation like these, 
it can be helpful to use an image classifier to speed up the identification process.  

We have implemented a GFNet to tackle this image classifcation problem. 
GFNet stands for Global Filter Networks, and it is similar to a vision transform with some key differences. 
It makes use of a 2D Fourier Transformation to find frequency-domain features and runs an element-wise multiplication between said features and learnable global filters. 
The multiplied result is then converting the result back to the time domain using a 2D Inverse Fourier Transform[^3]. 
This mechanism is used to replace the self-attention layer found in vision tranformers. 

![GFNet Visualised](resources/GFNet_visual.gif "GFNet Visualised")[^4] 
**GFNet Visualised**  

This resolves the complexity that the self-attention brings to the model for larger images, and thus allows scaling for for higher level resolutions. 

#### GFNet Layers  
Patch Embedding
Global Filter
MLP
global avg pooling


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
#### Cropping and Resizing Data  
Throughout the dataset, we could see slight variations in the positioning of the 
brain (see above) between images, as well as the brain sizing. Additionally, 
the brain scan images typically included a lot of unneccessary background pixels. 
In order to remove the unneccesary pixels and to keep a consistent brain size, 
we crop and resize the dataset so that it maximises the brain region. This was done 
by extracting the rectangular region containing the non-background pixels and adding 
padding to the borders to fit a 210x210 image. These dimensions were chosen as there 
was no cropping dimension that exceeded 210 within the dataset. 

#### Transforms  
Transforms are used to provide variability in the training data, 

RandomResizedCrop
RandomRotation
RandomAffine
RandomApply
RandomErasing

#### Normalisation
Both testing and training data need to be normalised in order to 

In order to find appropriate mean and standard deviation values for the dataset, 
we iterated through all of the images in the training data and calculated the average 
mean and standard deviation of the training dataset. These were found to be 
``MEAN = 0.11486841564676334`` and ``STD = 0.21826585544938487`` respectively, and 
were used to normlise the data. 

| ![Unprocessed Image](resources/808819_88_NC_train.jpeg "Unprocessed Image") | ![Pre-processed Image](resources/808819_88_NC_processed_train.jpeg "Pre-processed Image") |
| ----- | ----- |
| **Unprocessed Image** | **Pre-processed Image** | 

### Datasets  
#### ADNIDatasetTest Image Stacking
There are some small difference between the designed training dataset and the testing 
dataset. Namely, the testing dataset retrieves the data by patient rather than by image. 
This allows us to test on a stack of images common to a single patient rather than 
try to predict based off a single image alone. 

#### Loss function, Optimizer and Scheduler  

The *loss function* is used to calculate the difference between the predicted values and the 
true values of the data set. We have opted to use Binary Cross Entropy Loss for the loss function, 
since the task is a binary classification.  

The *optimizer* is used to change the parameters of the model to minimise the loss. We chose the 
AdamW for the optimizer due to its flexible nature and it's ability to remain stable and reduce overfitting, 
especially in large models like transformers.  

The *learning rate scheduler* is used to adjust the learning rate of the optimizer when training. 
It helps prevent overfitting of the data and improves overall convergence. We decided to use a 
CosineAnnealingLR for the scheduler due to it's reputation of being an effective scheduler for a 
wide range of models.  

### Results  
We first set MAX_EPOCHS to 100 to find where the model started overfitting


| ![Finding Ideal Epoch - Accuracy](resources/acc_vs_epoch_100_seed1.png "Finding Ideal Epoch - Accuracy") | ![Finding Ideal Epoch - Loss](resources/loss_vs_epoch_100_seed1.png "Finding Ideal Epoch - Loss") |
| ----- | ----- | 
| **Finding Ideal Epoch - Accuracy** | **Finding Ideal Epoch - Loss** |

As seen from the figures, the performance of the model peaks at around 60 epochs and then proceed to 
start overfitting. Following this, we adjusted the number of MAX_EPOCHS to 75 to limit redundant training loops. 

| ![Final Model - Accuracy](resources/acc_vs_epoch_seed10.png "Final Model - Accuracy") | ![Final Model - Loss](resources/loss_vs_epoch_seed10.png "Final Model - Loss") |
| ----- | ----- |
| **Final Model - Accuracy** | **Final Model - Loss**  


![Confusion Matrix](resources/confusion_matrix.png "Confusion Matrix")  
**Confusion Matrix** 

| ![ROC Curve](resources/roc_curve.png "ROC Curve") | ![Precision Recall Curve](resources/precision_recall_curve.png "Precision Recall Curve") |
| ----- | ----- |
| **ROC Curve** | **Precision Recall Curve** |

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
usage: predict.py [-h] [-dp TESTPATH] [-sp SAVEPATH] [-mp MODELPATH]

options:
  -h, --help            show this help message and exit
  -dp TESTPATH, --testpath TESTPATH
                        Filepath to ADNI testing dataset
  -sp SAVEPATH, --savepath SAVEPATH
                        Filepath to saved elements
  -mp MODELPATH, --modelpath MODELPATH
                        Filepath to saved model
```  

### Conclusion 
good success, change threshold in testing to reduce false negative rate, since it is medical we do not want to miss possible diagnosis
although 80% acc is high, in a medical context it may not be as ideal since it directly relates to the health of people. 
(look to changing mean and std)
(experiment with other transforms)
(try training with patients rather than images? -> may need more data)

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

### References  
[^1]: https://www.alz.org/alzheimers-dementia/what-is-alzheimers  
[^2]: https://www.medicalnewstoday.com/articles/alzheimers-brain-vs-normal-brain#alzheimers-brain  
[^3]: https://arxiv.org/abs/2107.00645  
[^4]: https://github.com/raoyongming/GFNet


dataset.py  
Resizing image, Otsu's threshold method, resize dimension (210), Lancoz interpolation  

The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.
It should also list any dependencies required, including versions and address reproduciblility of results, if applicable
provide example inputs, outputs and plots of your algorithm
Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
Include a conclusion, future works,  - many people lost marks due to this