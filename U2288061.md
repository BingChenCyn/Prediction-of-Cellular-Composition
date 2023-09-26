```python
!pip install tqdm
!pip install tabulate
!pip install scipy
!pip install torch
!pip install torchvision
```

    Requirement already satisfied: tqdm in c:\users\cynthia\anaconda3\envs\pytorch\lib\site-packages (4.65.0)
    Requirement already satisfied: colorama in c:\users\cynthia\anaconda3\envs\pytorch\lib\site-packages (from tqdm) (0.4.6)
    


```python
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hed, hed2rgb
from sklearn.decomposition import PCA
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import*
import warnings
warnings.filterwarnings("ignore")
from tabulate import tabulate
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
```


```python
X = np.load("images.npy")#read images
Y = pd.read_csv('counts.csv')#read cell counts 
F = np.loadtxt('split.txt')#read fold information
```

# Question No. 1: (Data Analysis)

### i. How many examples are there in each fold?


```python
value,count=np.unique(F,return_counts=True)
# print('There are', fold_sizes, 'in each fold')
for i in range(len(value)):
    print('There are', count[i], 'examples in fold',int(value[i]),'.')
```

    There are 1622 examples in fold 1 .
    There are 1751 examples in fold 2 .
    There are 1608 examples in fold 3 .
    

### ii. Show some image examples using plt.imshow. Describe your observations on what you see in the images and how it correlates with the cell counts of different types of cells and the overall number of cells.

From the randomly generated images, I have observed distinct differences in cell morphology, size, and staining between different types of cells. For instance, in example 2162, connective cells exhibit a longer morphology, whereas lymphocytes and eosinophils are rounder. Furthermore, different types of cell staining can be used to highlight specific cell components, with lymphocytes and eosinophils staining darker than connective cells in the same sample.

By analyzing the cell counts of different types of cells and the overall number of cells, I have noticed that the density and size of cells can vary significantly between different cell types. By analyzing the number of cells in a given area, it is possible to estimate the cell density and identify any differences in cell distribution. Additionally, by measuring the size of cells, one can determine the average size of the cells and look for variations in size between different cell types. Overall, these observations allow for a more accurate identification and characterization of different types of cells, which is crucial for medical research, pathology, and cell biology.

Images with a higher total cell count tend to exhibit more diversity in cell types. The count of eosinophil cells is not correlated with the total cell count, as it remains constant regardless of changes in the total count. However, an increase in the count of each cell type will result in an increase in the total cell count.

There are 20 examples. Each image displays the count of each cell type above it, while the total cell count for each image is included in the image title.


```python
# randomly pick 20 examples
idx = np.random.randint(X.shape[0], size=20)
idx
for i in idx:
    plt.imshow(X[i])
    print('Type'+str(Y.iloc[i][Y.iloc[i]>0])) 

    plt.title(' The total numbers of cells '+str(sum(Y.iloc[i]))+' in example '+str(i))
    plt.show()
```

    Typeepithelial    49
    Name: 2999, dtype: int64
    


    
![png](output_9_1.png)
    


    Typeepithelial    26
    lymphocyte     2
    plasma         8
    connective     4
    Name: 1896, dtype: int64
    


    
![png](output_9_3.png)
    


    Typeepithelial    53
    lymphocyte    16
    plasma         7
    connective    27
    Name: 511, dtype: int64
    


    
![png](output_9_5.png)
    


    Typelymphocyte    161
    plasma          3
    connective     68
    Name: 35, dtype: int64
    


    
![png](output_9_7.png)
    


    Typeneutrophil     4
    epithelial    43
    lymphocyte    41
    plasma        10
    connective    12
    Name: 2810, dtype: int64
    


    
![png](output_9_9.png)
    


    Typeepithelial    168
    lymphocyte     13
    plasma          1
    connective      6
    Name: 2674, dtype: int64
    


    
![png](output_9_11.png)
    


    Typeneutrophil     1
    epithelial    69
    lymphocyte    28
    plasma         6
    eosinophil     1
    connective     5
    Name: 4675, dtype: int64
    


    
![png](output_9_13.png)
    


    Typeepithelial    33
    lymphocyte    51
    plasma        23
    connective    26
    Name: 454, dtype: int64
    


    
![png](output_9_15.png)
    


    Typelymphocyte     2
    eosinophil     2
    connective    51
    Name: 2162, dtype: int64
    


    
![png](output_9_17.png)
    


    Typeepithelial    27
    lymphocyte    30
    plasma         9
    eosinophil     3
    connective     9
    Name: 4099, dtype: int64
    


    
![png](output_9_19.png)
    


    Typeepithelial    69
    lymphocyte    19
    plasma         8
    connective    24
    Name: 4338, dtype: int64
    


    
![png](output_9_21.png)
    


    Typeepithelial    75
    lymphocyte    74
    plasma        17
    eosinophil     1
    connective    27
    Name: 1935, dtype: int64
    


    
![png](output_9_23.png)
    


    Typelymphocyte    12
    eosinophil     1
    connective    58
    Name: 937, dtype: int64
    


    
![png](output_9_25.png)
    


    Typeepithelial    77
    lymphocyte    30
    plasma        34
    eosinophil     1
    connective    25
    Name: 2102, dtype: int64
    


    
![png](output_9_27.png)
    


    Typeepithelial    70
    lymphocyte    51
    plasma        14
    connective    28
    Name: 4258, dtype: int64
    


    
![png](output_9_29.png)
    


    Typeneutrophil     2
    epithelial     9
    lymphocyte    45
    plasma        14
    connective    17
    Name: 2534, dtype: int64
    


    
![png](output_9_31.png)
    


    Typeneutrophil    12
    epithelial    56
    lymphocyte    26
    plasma         1
    connective    15
    Name: 1517, dtype: int64
    


    
![png](output_9_33.png)
    


    Typeepithelial    49
    lymphocyte    18
    plasma        16
    eosinophil     1
    connective    19
    Name: 437, dtype: int64
    


    
![png](output_9_35.png)
    


    Typeepithelial    59
    lymphocyte    23
    plasma        16
    eosinophil     1
    connective    21
    Name: 4585, dtype: int64
    


    
![png](output_9_37.png)
    


    Typeepithelial    48
    lymphocyte     1
    plasma         1
    connective     2
    Name: 107, dtype: int64
    


    
![png](output_9_39.png)
    


### iii. For each fold, plot the histogram of counts of each cell type separately as well as the total number of cells (7 plots in total). How many images have counts within each of the following bins? 

Note: I also include the counts which the total cell number exceed 100.
The code can be divided into several sections.

First, the script defines the names of the cell types and the three folds used for counting and visualization.

Next, the number of cells falling within each cell count range is counted using predefined bins, and the results are stored in a DataFrame object.

Subsequently, histograms are plotted using the plt.hist() function for each cell type in each fold as well as the total number of cells, and printed out.

Finally, the code outputs a plot displaying the number of cells in each cell count range and for each cell type in each of the three folds.

Through these statistics and visualizations, we can get a general idea of the distribution of different cell types and the total number of cells in the training set. Specifically, we can observe that some cell types are less abundant while others are more abundant, and we can also see that the distribution of cell numbers varies slightly across different folds. This information can be useful for subsequent model training and evaluation.


```python
type_name = ["neutrophil","epithelial","lymphocyte","plasma","eosinophil","connective"]
fold = [1,2,3]
# Print the counts in each bin
bins = [0,1,6, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, float('inf')]
fold1_bin=pd.value_counts(pd.cut(Y[F==1].apply(lambda x:x.sum(),axis =1),bins,right=False),sort=False).values

fold2_bin=pd.value_counts(pd.cut(Y[F==2].apply(lambda x:x.sum(),axis =1),bins,right=False),sort=False).values

fold3_bin=pd.value_counts(pd.cut(Y[F==3].apply(lambda x:x.sum(),axis =1),bins,right=False),sort=False).values

bin=pd.value_counts(pd.cut(Y[F==1].apply(lambda x:x.sum(),axis =1),bins,right=False),sort=False).index.tolist()

fold_bin=[fold1_bin,fold2_bin,fold3_bin]
df=DataFrame(fold_bin,index=['fold 1','fold 2','fold 3'],columns=bin).T
print(df)


Y_fold1=Y[F==1]
To_Y_fold1=Y_fold1.apply(lambda x:x.sum(),axis =1)


for i in fold:
    for j in range(len(type_name)):
        a =  Y[F==i][type_name[j]]
        n, bins_1, patches=plt.hist(a,bins=[0,1,6, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100, np.max(np.concatenate([a,[100]]))])
        plt.title("Fold"+str(i)+","+str(type_name[j]))
        plt.show()
    n, bins_2, patches=plt.hist(To_Y_fold1,bins=[0,1,6, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100, np.max(np.concatenate([To_Y_fold1,[100]]))])
    plt.title("Overall in Fold"+str(i))
    plt.show()

```

                   fold 1  fold 2  fold 3
    [0.0, 1.0)         39      54      57
    [1.0, 6.0)         17      13      31
    [6.0, 11.0)        15      30      36
    [11.0, 21.0)       23      51      85
    [21.0, 31.0)       47      47      90
    [31.0, 41.0)       43      66      97
    [41.0, 51.0)       65     107      71
    [51.0, 61.0)       83     137     105
    [61.0, 71.0)       94     134     118
    [71.0, 81.0)      132     135     124
    [81.0, 91.0)      164     162     133
    [91.0, 101.0)     132     160     128
    [101.0, inf)      768     655     533
    


    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    



    
![png](output_12_5.png)
    



    
![png](output_12_6.png)
    



    
![png](output_12_7.png)
    



    
![png](output_12_8.png)
    



    
![png](output_12_9.png)
    



    
![png](output_12_10.png)
    



    
![png](output_12_11.png)
    



    
![png](output_12_12.png)
    



    
![png](output_12_13.png)
    



    
![png](output_12_14.png)
    



    
![png](output_12_15.png)
    



    
![png](output_12_16.png)
    



    
![png](output_12_17.png)
    



    
![png](output_12_18.png)
    



    
![png](output_12_19.png)
    



    
![png](output_12_20.png)
    



    
![png](output_12_21.png)
    


### iv. Pre-processing: Convert and show a few images from RGB space to HED space and show the H-channel which should indicate cellular nuclei. For this purpose, you can use the color separation notebook available here: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html

What this code does is to select 10 samples at random from the image dataset, convert them to the HED (Hematoxylin-Eosin-DAB) colour space and display the original image and Hematoxylin channel for one of the samples. By looping, for each selected sample, the code calculates the HED colour space and converts the result to an RGB image, finally displaying the original image and the Hematoxylin image in a chart containing two subplots. This can be used to observe the structure and position of the cell nuclei in the image.



```python
idx = np.random.randint(X.shape[0], size=10)
idx

ihc_hed = []
for i in range(len(X)):
    ihc_hed.append(rgb2hed(X[i]))
# Example IHC image
for i in idx:
    null = np.zeros_like(ihc_hed[i][:, :, 0])
    ihc_h=hed2rgb(np.stack((ihc_hed[i][:, :, 0], null, null), axis=-1))
#     # Display
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()
    
    ax[0].imshow(X[i])
    ax[0].set_title("Original image")


    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")
```


    
![png](output_15_0.png)
    



    
![png](output_15_1.png)
    



    
![png](output_15_2.png)
    



    
![png](output_15_3.png)
    



    
![png](output_15_4.png)
    



    
![png](output_15_5.png)
    



    
![png](output_15_6.png)
    



    
![png](output_15_7.png)
    



    
![png](output_15_8.png)
    



    
![png](output_15_9.png)
    


### v. Do a scatter plot of the average of the H-channel for each image vs. its cell count of a certain type and the total number of cells for images in Fold-1 (7 plots in total). Do you think this feature would be useful in your regression model? Explain your reasoning.

I think that in a regression model, the mean of the H-channel of the total number of cells would be useful. However, there is no significant linear correlation between the mean of the H-channel for a particular type of cell number and the number of cells. In the first, third, fourth and seven scatter plots we can see a positive and linear correlation between the mean H-channel value and the total number of cells, but we may need to transform the data, for example by taking the logarithm, to make it easier to interpret. Other images show that possibly performing a transformation would make the relationship between the H-channel mean and the number of cells a little more obvious.



```python
np.random.randint(X[F==1].shape[0],size=10)
```




    array([1204,  562, 1270, 1010, 1350,  918,   96,  711,  213,  186])




```python

neutrophil=Y_fold1.loc[:,'neutrophil'].tolist()
epithelial=Y_fold1.loc[:,'epithelial'].tolist()
lymphocyte=Y_fold1.loc[:,'lymphocyte'].tolist()
plasma=Y_fold1.loc[:,'plasma'].tolist()
eosinophil=Y_fold1.loc[:,'eosinophil'].tolist()
connective=Y_fold1.loc[:,'connective'].tolist()

rgb_fold1= X[F==1]
fold1_hed=rgb2hed(rgb_fold1)
fold1_h=fold1_hed[:,:,:,0]
h_avg=np.mean(fold1_h,axis=(1,2))
    
fig, axs = plt.subplots(3, 3, figsize=(15,15))


axs[0,0].scatter(To_Y_fold1,h_avg,s=4,alpha=0.5)
axs[0,0].set_xlabel('Image')
axs[0,0].set_ylabel('Average H-channel value')
axs[0,0].set_title('Average value vs. total')

axs[0,1].scatter(neutrophil,h_avg,s=4,alpha=0.5)
axs[0,1].set_xlabel('Image')
axs[0,1].set_ylabel('Average H-channel value')
axs[0,1].set_title('Average H-channel value vs neutrophil')

axs[0,2].scatter(epithelial,h_avg,s=4,alpha=0.5)
axs[0,2].set_xlabel('Image')
axs[0,2].set_ylabel('Average H-channel value')
axs[0,2].set_title('Average H-channel value vs epithelial')

axs[1,0].scatter(lymphocyte,h_avg,s=4,alpha=0.5)
axs[1,0].set_xlabel('Image')
axs[1,0].set_ylabel('Average H-channel value')
axs[1,0].set_title('Average H-channel value vs lymphocyte')

axs[1,1].scatter(plasma,h_avg,s=4,alpha=0.5)
axs[1,1].set_xlabel('Image')
axs[1,1].set_ylabel('Average H-channel value')
axs[1,1].set_title('Average H-channel value vs plasma')

axs[1,2].scatter(eosinophil,h_avg,s=4,alpha=0.5)
axs[1,2].set_xlabel('Image')
axs[1,2].set_ylabel('Average H-channel value')
axs[1,2].set_title('Average H-channel value vs eosinophil')

axs[2,0].scatter(connective,h_avg,s=4,alpha=0.5)
axs[2,0].set_xlabel('Image')
axs[2,0].set_ylabel('Average H-channel value')
axs[2,0].set_title('Average H-channel value vs connective')


```




    Text(0.5, 1.0, 'Average H-channel value vs connective')




    
![png](output_19_1.png)
    



```python

```

### vi. What performance metrics can you use for this problem? Which one will be the best performance metric for this problem? Please give your reasoning.

The choice of the best performance metric for regression problems depends on the specific objective of the regression model. If the aim is to make accurate predictions of the total number of cells in an image, then metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) may be appropriate since they penalize larger errors more heavily, which is crucial when minimizing the overall prediction error.

In regression problems, other metrics are commonly used to assess the accuracy of predicted continuous values, with Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²) being the most frequently used ones.

Alternatively, if the objective is to predict the relative cell count of a particular type, then metrics such as Mean Absolute Error (MAE) or Mean Absolute Percentage Error (MAPE) may be more suitable since they give equal importance to all errors, which is crucial when accurately predicting the proportion of a particular cell type.

Ultimately, the choice of the best performance metric will depend on the specific objectives of the regression model and the context in which it will be applied.


# Question No. 2: (Feature Extraction and Classical Regression)

### i. Extract features from a given image. Specifically, calculate the:
### a. average of the “H”, red, green and blue channels


```python
def avg_H_R_G_B(dataset_rgb):  
    r_avg=[]
    g_avg=[]
    b_avg=[]
    fold_hed=rgb2hed(dataset_rgb)
    fold_h=fold_hed[:,:,:,0]
    h_avg=np.mean(fold_h,axis=(1,2))
    for i in range(len(dataset_rgb)):
        # Extract the H, R, G, and B channels
        r_channel = dataset_rgb[i][:, :, 0]
        g_channel = dataset_rgb[i][:, :, 1]
        b_channel = dataset_rgb[i][:, :, 2]

        # Calculate the average values of each channel
        r_avg.append(np.mean(r_channel))
        g_avg.append(np.mean(g_channel))
        b_avg.append(np.mean(b_channel))
    return h_avg, r_avg, g_avg, b_avg
```


```python
h_avg, r_avg, g_avg, b_avg=avg_H_R_G_B(dataset_rgb=X[(F==1)|(F==2)])
```


```python

Y_fold12=Y[(F==1)|(F==2)]
To_Y_fold12=Y_fold12.apply(lambda x:x.sum(),axis =1)
# scatter plot
fig, axs = plt.subplots(2, 2, figsize=(12,12))

axs[0,0].scatter(To_Y_fold12,h_avg,s=4,alpha=0.5)
axs[0,0].set_xlabel('Image')
axs[0,0].set_ylabel('Average H-channel value')
axs[0,0].set_title('Average H vs. total')

axs[0,1].scatter(To_Y_fold12,r_avg,s=4,alpha=0.5)
axs[0,1].set_xlabel('Image')
axs[0,1].set_ylabel('Average R-channel value')
axs[0,1].set_title('Average R vs. total')

axs[1,0].scatter(To_Y_fold12,g_avg,s=4,alpha=0.5)
axs[1,0].set_xlabel('Image')
axs[1,0].set_ylabel('Average G-channel value')
axs[1,0].set_title('Average G vs. total')

axs[1,1].scatter(To_Y_fold12,b_avg,s=4,alpha=0.5)
axs[1,1].set_xlabel('Image')
axs[1,1].set_ylabel('Average B-channel value')
axs[1,1].set_title('Average B value vs total')

print('The correlation coefficient of average H-channel and cell count is', str(round(np.corrcoef(h_avg, To_Y_fold12)[0][1],5)))
print('The correlation coefficient of average R-channel and cell count is', str(round(np.corrcoef(r_avg, To_Y_fold12)[0][1],5)))
print('The correlation coefficient of average G-channel and cell count is', str(round(np.corrcoef(g_avg, To_Y_fold12)[0][1],5)))
print('The correlation coefficient of average B-channel and cell count is', str(round(np.corrcoef(b_avg, To_Y_fold12)[0][1],5)))
```

    The correlation coefficient of average H-channel and cell count is 0.43365
    The correlation coefficient of average R-channel and cell count is -0.48831
    The correlation coefficient of average G-channel and cell count is -0.58009
    The correlation coefficient of average B-channel and cell count is -0.46792
    


    
![png](output_27_1.png)
    


### b. variance of the “H”, red, green and blue channels


```python
def var_H_R_G_B(dataset_rgb): 
    r_var=[]
    g_var=[]
    b_var=[]
    fold_hed=rgb2hed(dataset_rgb)
    fold_h=fold_hed[:,:,:,0]
    h_var=np.var(fold_h,axis=(1,2))
    for i in range(len(dataset_rgb)):
        # Extract the H, R, G, and B channels
#         h_channel = dataset_H[i][:, :, 0]
        r_channel = dataset_rgb[i][:, :, 0]
        g_channel = dataset_rgb[i][:, :, 1]
        b_channel = dataset_rgb[i][:, :, 2]
        # Calculate the average values of each channel
#         h_var.append(np.var(h_channel))
        r_var.append(np.var(r_channel))
        g_var.append(np.var(g_channel))
        b_var.append(np.var(b_channel))
    return h_var, r_var, g_var, b_var
```


```python
h_var, r_var, g_var, b_var=var_H_R_G_B(dataset_rgb=X[(F==1)|(F==2)])

# scatter plot
fig, axs = plt.subplots(2, 2, figsize=(12,12))


axs[0,0].scatter(To_Y_fold12,h_var,s=4,alpha=0.5)
axs[0,0].set_xlabel('Image')
axs[0,0].set_ylabel('Variance H-channel value')
axs[0,0].set_title('Variance H vs. total')

axs[0,1].scatter(To_Y_fold12,r_var,s=4,alpha=0.5)
axs[0,1].set_xlabel('Image')
axs[0,1].set_ylabel('Variance R-channel value')
axs[0,1].set_title('Variance R vs. total')

axs[1,0].scatter(To_Y_fold12,g_var,s=4,alpha=0.5)
axs[1,0].set_xlabel('Image')
axs[1,0].set_ylabel('Variance G-channel value')
axs[1,0].set_title('Variance G vs. total')

axs[1,1].scatter(To_Y_fold12,b_var,s=4,alpha=0.5)
axs[1,1].set_xlabel('Image')
axs[1,1].set_ylabel('Variance B-channel value')
axs[1,1].set_title('Variance B value vs total')

print('The correlation coefficient of variance H-channel and cell count is', str(round(np.corrcoef(h_var, To_Y_fold12)[0][1],5)))
print('The correlation coefficient of variance R-channel and cell count is', str(round(np.corrcoef(r_var, To_Y_fold12)[0][1],5)))
print('The correlation coefficient of variance G-channel and cell count is', str(round(np.corrcoef(g_var, To_Y_fold12)[0][1],5)))
print('The correlation coefficient of variance B-channel and cell count is', str(round(np.corrcoef(b_var, To_Y_fold12)[0][1],5)))
```

    The correlation coefficient of variance H-channel and cell count is 0.42902
    The correlation coefficient of variance R-channel and cell count is 0.48163
    The correlation coefficient of variance G-channel and cell count is 0.2405
    The correlation coefficient of variance B-channel and cell count is 0.28622
    


    
![png](output_30_1.png)
    


### c. Any other features that you think can be useful for this work. Describe your reasoning for using these features. HINT/Suggestion: You may want to use PCA Coefficients of image data (you may want to use randomized PCA or incremental PCA, see:

### PCA

PCA is primarily used to reduce the dimensionality of a high-dimensional dataset while retaining as much of the original data information as possible. It achieves this by generating a set of principal components, which are linear combinations of the original data. The weights or coefficients of these principal components describe their contribution to the original data.

If the size of the original dataset is increased from 1000 to 2000 data points, the coefficients of the principal components in the PCA result may change. This is because the increased dataset size may reveal more data variability that can impact the coefficients of the principal components.

However, in practical scenarios, the changes in principal component coefficients may not be significant, particularly when large datasets are used for PCA calculation, and these changes may tend to stabilize over time. Moreover, even if the coefficients undergo some changes, they can still explain the principal components' contribution to the original data.

Thus, increasing the dataset size may lead to some changes in principal component coefficients, but these changes may not significantly affect the interpretation of the principal components.


```python
# find the best number component which can explain 95% variance
def pca_component(X_fold):
    avg_component=[]
    
    fold_hed=rgb2hed(X_fold)
    # pca coefficients
    h_channel = fold_hed[:,:, :, 0].reshape(-1,256*256)
    # Centralize
    h_channel = h_channel-np.mean(h_channel, axis=0)
    pca_h = PCA()
    pca_h.fit(h_channel)
    avg_component.append(np.where(np.cumsum(pca_h.explained_variance_ratio_)>=0.95)[0][0]+1)
    del h_channel,pca_h
    
    r_channel = X_fold[:,:, :, 0].reshape(-1,256*256)
    r_channel = r_channel-np.mean(r_channel, axis=0)
    pca_r = PCA()
    pca_r.fit(r_channel)
    avg_component.append(np.where(np.cumsum(pca_r.explained_variance_ratio_)>=0.95)[0][0]+1)
    del r_channel,pca_r
    
    g_channel = X_fold[:,:, :, 1].reshape(-1,256*256)
    g_channel = g_channel-np.mean(g_channel, axis=0)
    pca_g = PCA()
    pca_g.fit(g_channel)
    avg_component.append(np.where(np.cumsum(pca_g.explained_variance_ratio_)>=0.95)[0][0]+1)
    del g_channel,pca_g

    b_channel = X_fold[:,:, :, 2].reshape(-1,256*256)
    b_channel = b_channel-np.mean(b_channel, axis=0)
    pca_b = PCA()
    pca_b.fit(b_channel)
    avg_component.append(np.where(np.cumsum(pca_b.explained_variance_ratio_)>=0.95)[0][0]+1)
    del b_channel,pca_b
    
    return int(np.mean(avg_component))
```


```python
component=pca_component(X[F==1])
component
```




    907



Due to the large size of the data, it was necessary to run it separately, and in order to avoid running out of memory and to reduce the runtime, I stored the results after running them.


```python
component=907
def pca_H_feature(X_fold,component):
    fold_hed=rgb2hed(X_fold)
       
    # Process H channel
    h_channel = fold_hed[:,:, :, 0].reshape(-1,256*256)
    # 中心化处理
    h_channel = h_channel-np.mean(h_channel, axis=0)
    pca_h = PCA(n_components=component)
    X_H_pca=pca_h.fit_transform(h_channel)
    return X_H_pca

```


```python
H_pca_train_feature=pca_H_feature(X[(F==1)| (F==2)],907)
#saving the result to CSV document.
np.savetxt('H_pca_train_feature.csv', H_pca_train_feature, delimiter=',')
del H_pca_train_feature
```


```python
def pca_rgb_feature(X_fold,rgb,component):
       
    r_channel = X_fold[:,:, :, rgb].reshape(-1,256*256)
    r_channel = r_channel-np.mean(r_channel, axis=0)
    pca_r = PCA(n_components=component)
    X_R_pca=pca_r.fit_transform(r_channel)
    
    return X_R_pca

```


```python
r_pca_train_feature=pca_rgb_feature(X[(F==1)| (F==2)],0,component)
np.savetxt('r_pca_train_feature.csv', r_pca_train_feature, delimiter=',')
del r_pca_train_feature

g_pca_train_feature=pca_rgb_feature(X[(F==1)| (F==2)],1,component)
np.savetxt('g_pca_train_feature.csv', g_pca_train_feature, delimiter=',')
del g_pca_train_feature

b_pca_train_feature=pca_rgb_feature(X[(F==1)| (F==2)],2,component)
np.savetxt('b_pca_train_feature.csv', b_pca_train_feature, delimiter=',')
del b_pca_train_feature
```


```python
X_H_pca = np.loadtxt('H_pca_train_feature.csv', delimiter=',')
X_R_pca = np.loadtxt('r_pca_train_feature.csv', delimiter=',')
X_G_pca = np.loadtxt('g_pca_train_feature.csv', delimiter=',')
X_B_pca = np.loadtxt('b_pca_train_feature.csv', delimiter=',')
feature_pca = np.concatenate((X_H_pca, X_R_pca, X_G_pca, X_B_pca), axis=1)
feature_pca

np.savetxt('pca_train_feature.csv', feature_pca, delimiter=',')

```


```python
feature_pca = np.loadtxt('pca_train_feature.csv', delimiter=',')
```


```python

Y_fold12=Y[(F==1)| (F==2)]
To_Y_fold12=Y_fold12.apply(lambda x:x.sum(),axis =1)
```


```python
table_coef_pca = [[' ','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']]
channel=['H-Channel','R-Channel','G-Channel','B-Channel']
# scatter plot
fig, axs = plt.subplots(4, 5,figsize=(23,25))
#, figsize=(12,12)
for j in range(len(channel)):
    dif_channel=component*j
    new=[[channel[j],
          round(np.corrcoef(feature_pca[:,dif_channel],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+1],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+2],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+3],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+4],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+5],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+6],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+7],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+8],To_Y_fold12)[0][1],5),
          round(np.corrcoef(feature_pca[:,dif_channel+9],To_Y_fold12)[0][1],5)
         ]]
    table_coef_pca=np.r_[table_coef_pca,new]
    for i in range(5):
        axs[j,i].scatter(To_Y_fold12,feature_pca[:,dif_channel+i],s=4,alpha=0.5)
        axs[j,i].set_xlabel('Image')
        axs[j,0].set_ylabel('coef of PCA')
        axs[j,i].set_title(channel[j])

    
print(tabulate(table_coef_pca))


```

    ---------  -------  -------  --------  --------  --------  -------  --------  --------  --------  -------
               PC1      PC2      PC3       PC4       PC5       PC6      PC7       PC8       PC9       PC10
    H-Channel  0.43362  0.02504  0.00541   -0.00763  -0.01065  0.02927  -0.00276  -0.00085  0.00847   0.00745
    R-Channel  0.48831  0.01601  0.02314   -0.0008   -0.01454  0.03154  0.00065   0.01088   -0.00394  0.01345
    G-Channel  0.58008  0.00857  -0.02209  0.00371   -0.01972  0.0341   -0.01082  -0.01175  0.00571   0.01236
    B-Channel  0.4679   0.01376  0.0285    -0.00471  -0.0177   0.03743  0.00771   -0.00321  -0.00139  0.0242
    ---------  -------  -------  --------  --------  --------  -------  --------  --------  --------  -------
    


    
![png](output_44_1.png)
    


In summary, only the correlation coefficient between the first principal component and the total number of cells exceeds 0.2 in PCA.

### GLCM


```python
def GLCM(data):
    # Converting RGB images to grayscale
    X_gray = np.array([rgb2gray(data[i]) for i in range(data.shape[0])])

    # Initialize feature vectors
    features = np.zeros((data.shape[0], 4))

    # For each image, GLCM features are calculated and stored in the feature vector
    for i in range(data.shape[0]):
        glcm = greycomatrix((X_gray[i]*255).astype(np.uint8), [5], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        features[i, 0] = greycoprops(glcm, 'contrast').mean()
        features[i, 1] = greycoprops(glcm, 'dissimilarity').mean()
        features[i, 2] = greycoprops(glcm, 'homogeneity').mean()
        features[i, 3] = greycoprops(glcm, 'energy').mean()

    # Normalisation of feature vectors
    features = (features - features.mean(axis=0)) / features.std(axis=0)
    return features

```


```python
glcm_feature=GLCM(X[(F==1)| (F==2)])
```


```python
table_coef_glcm = [[' ','contrast','dissimilarity','homogeneity','energy']]
# scatter plot
fig, axs = plt.subplots(1, 4,figsize=(15,5))

for i in range(4):
        axs[i].scatter(To_Y_fold12,glcm_feature[:,i],s=4,alpha=0.5)
        axs[i].set_xlabel('Image')
        axs[i].set_ylabel('value of'+table_coef_glcm[0][i+1])
        axs[i].set_title(table_coef_glcm[0][i+1]+'vs Total')


new_glcm=[['GLCM',
      round(np.corrcoef(glcm_feature[:,0],To_Y_fold12)[0][1],5),
      round(np.corrcoef(glcm_feature[:,1],To_Y_fold12)[0][1],5),
      round(np.corrcoef(glcm_feature[:,2],To_Y_fold12)[0][1],5),
      round(np.corrcoef(glcm_feature[:,3],To_Y_fold12)[0][1],5)
     ]]
table_coef_glcm=np.r_[table_coef_glcm,new_glcm]
    
print(tabulate(table_coef_glcm))

```

    ----  --------  -------------  -----------  --------
          contrast  dissimilarity  homogeneity  energy
    GLCM  0.58061   0.65338        -0.48899     -0.41972
    ----  --------  -------------  -----------  --------
    


    
![png](output_49_1.png)
    


In summary, the absolute value of the correlation coefficient for each feature exceeds 0.4

### ii. Try the following regression models with the features used in part-I. Plot the scatter plot between true and predicted counts for each type of regression model for the test data. Also, report your prediction performance in terms of RMSE, Pearson Correlation Coefficient, Spearman Correlation Coefficient and R2 score (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) on the test data.

generate the test dataset and train dataset 


```python
H_pca_test_feature=pca_H_feature(X[F==3],907)
np.savetxt('H_pca_test_feature.csv', H_pca_test_feature, delimiter=',')
del H_pca_test_feature

r_pca_test_feature=pca_rgb_feature(X[F==3],1,907)
np.savetxt('r_pca_test_feature.csv', r_pca_test_feature, delimiter=',')
del r_pca_test_feature

g_pca_test_feature=pca_rgb_feature(X[F==3],2,907)
np.savetxt('g_pca_test_feature.csv', g_pca_test_feature, delimiter=',')
del g_pca_test_feature

b_pca_test_feature=pca_rgb_feature(X[F==3],0,907)
np.savetxt('b_pca_test_feature.csv', b_pca_test_feature, delimiter=',')
del b_pca_test_feature

```


```python
X_H_pca_test = np.loadtxt('H_pca_test_feature.csv', delimiter=',')
X_R_pca_test = np.loadtxt('r_pca_test_feature.csv', delimiter=',')
X_G_pca_test = np.loadtxt('g_pca_test_feature.csv', delimiter=',')
X_B_pca_test = np.loadtxt('b_pca_test_feature.csv', delimiter=',')
feature_pca_test = np.concatenate((X_H_pca_test, X_R_pca_test, X_G_pca_test, X_B_pca_test), axis=1)
np.savetxt('feature_pca_test.csv', feature_pca_test, delimiter=',')
del X_H_pca_test,X_R_pca_test,X_G_pca_test,X_B_pca_test
```


```python
feature_pca_test = np.loadtxt('feature_pca_test.csv', delimiter=',')
```


```python
# test dataset 
component=907
h_avg_test, r_avg_test, g_avg_test, b_avg_test = avg_H_R_G_B(dataset_rgb=X[F==3])
h_var_test, r_var_test, g_var_test, b_var_test=var_H_R_G_B(dataset_rgb=X[F==3])

glcm_feature_test=GLCM(X[F==3])

X_test_avg_var_pca_glcm = DataFrame(np.vstack([h_avg_test, r_avg_test, g_avg_test, b_avg_test,
                     h_var_test, r_var_test, g_var_test, b_var_test,
                     glcm_feature_test[:,0],glcm_feature_test[:,1],glcm_feature_test[:,2],glcm_feature_test[:,3]
                                         ,feature_pca_test[:,0],feature_pca_test[:,component],feature_pca_test[:,component*2],feature_pca_test[:,component*3]
                                              ]).T,
                     index=None,
                     columns = ['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var',
                                'glcm_contrast','glcm_dissimilarity','glcm_homogeneity','glcm_energy'
                                ,'h_pca1','r_pca1','g_pca1','b_pca1'
                               ])
```


```python
X_test_avg_var_pca_glcm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h_avg</th>
      <th>r_avg</th>
      <th>g_avg</th>
      <th>b_avg</th>
      <th>h_var</th>
      <th>r_var</th>
      <th>g_var</th>
      <th>b_var</th>
      <th>glcm_contrast</th>
      <th>glcm_dissimilarity</th>
      <th>glcm_homogeneity</th>
      <th>glcm_energy</th>
      <th>h_pca1</th>
      <th>r_pca1</th>
      <th>g_pca1</th>
      <th>b_pca1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.028605</td>
      <td>197.698242</td>
      <td>170.029984</td>
      <td>220.461472</td>
      <td>0.000512</td>
      <td>1331.335242</td>
      <td>1876.016023</td>
      <td>481.998760</td>
      <td>0.810735</td>
      <td>0.965928</td>
      <td>-0.439103</td>
      <td>-0.362401</td>
      <td>-0.373034</td>
      <td>-4749.207900</td>
      <td>-5509.526716</td>
      <td>-1421.715926</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.026077</td>
      <td>202.473419</td>
      <td>176.373886</td>
      <td>222.857880</td>
      <td>0.000533</td>
      <td>1417.460078</td>
      <td>2178.975367</td>
      <td>509.681401</td>
      <td>0.857835</td>
      <td>0.828788</td>
      <td>-0.208924</td>
      <td>-0.183399</td>
      <td>-1.033533</td>
      <td>-6404.225984</td>
      <td>-6133.879506</td>
      <td>-2668.576156</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.029114</td>
      <td>197.348557</td>
      <td>170.161255</td>
      <td>220.063202</td>
      <td>0.000586</td>
      <td>1539.463942</td>
      <td>2307.646238</td>
      <td>535.724338</td>
      <td>0.507328</td>
      <td>0.578942</td>
      <td>0.009299</td>
      <td>0.063534</td>
      <td>-0.228566</td>
      <td>-4750.646593</td>
      <td>-5405.057624</td>
      <td>-1315.154987</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.023964</td>
      <td>205.994827</td>
      <td>180.647141</td>
      <td>223.980301</td>
      <td>0.000504</td>
      <td>1436.195576</td>
      <td>2438.515856</td>
      <td>513.623590</td>
      <td>0.575520</td>
      <td>0.579722</td>
      <td>-0.002403</td>
      <td>0.007661</td>
      <td>-1.549227</td>
      <td>-7497.239372</td>
      <td>-6410.665311</td>
      <td>-3547.594265</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.033372</td>
      <td>188.374313</td>
      <td>154.436249</td>
      <td>214.194946</td>
      <td>0.000475</td>
      <td>1233.496379</td>
      <td>1642.879389</td>
      <td>537.033895</td>
      <td>0.341959</td>
      <td>0.632068</td>
      <td>-0.475922</td>
      <td>-0.378003</td>
      <td>0.861196</td>
      <td>-738.945031</td>
      <td>-3901.417819</td>
      <td>978.760321</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1603</th>
      <td>0.017153</td>
      <td>212.572479</td>
      <td>172.512909</td>
      <td>217.049973</td>
      <td>0.000274</td>
      <td>964.245205</td>
      <td>1629.919114</td>
      <td>580.884420</td>
      <td>0.546894</td>
      <td>0.683091</td>
      <td>-0.399490</td>
      <td>-0.348863</td>
      <td>-3.301641</td>
      <td>-5396.706310</td>
      <td>-4640.199170</td>
      <td>-5238.272733</td>
    </tr>
    <tr>
      <th>1604</th>
      <td>0.020643</td>
      <td>206.418716</td>
      <td>165.679520</td>
      <td>209.284439</td>
      <td>0.000566</td>
      <td>1766.249466</td>
      <td>3407.630187</td>
      <td>1057.736126</td>
      <td>0.213481</td>
      <td>0.095125</td>
      <td>0.130653</td>
      <td>-0.015457</td>
      <td>-2.412801</td>
      <td>-3714.663917</td>
      <td>-2670.492359</td>
      <td>-3679.925367</td>
    </tr>
    <tr>
      <th>1605</th>
      <td>0.023952</td>
      <td>202.140808</td>
      <td>166.652145</td>
      <td>209.097931</td>
      <td>0.000644</td>
      <td>2090.342844</td>
      <td>4087.511367</td>
      <td>1211.674858</td>
      <td>-0.407248</td>
      <td>-0.393793</td>
      <td>0.284078</td>
      <td>0.079372</td>
      <td>-1.553795</td>
      <td>-3871.009463</td>
      <td>-2590.235828</td>
      <td>-2550.724070</td>
    </tr>
    <tr>
      <th>1606</th>
      <td>0.023798</td>
      <td>202.249878</td>
      <td>164.874191</td>
      <td>208.633499</td>
      <td>0.000661</td>
      <td>2105.959045</td>
      <td>4008.991847</td>
      <td>1247.109223</td>
      <td>0.660789</td>
      <td>0.373970</td>
      <td>0.327851</td>
      <td>0.132698</td>
      <td>-1.601612</td>
      <td>-3419.008046</td>
      <td>-2481.004566</td>
      <td>-2583.563919</td>
    </tr>
    <tr>
      <th>1607</th>
      <td>0.023468</td>
      <td>203.599594</td>
      <td>170.092133</td>
      <td>210.553970</td>
      <td>0.000671</td>
      <td>2160.485748</td>
      <td>4406.558559</td>
      <td>1264.357805</td>
      <td>0.001142</td>
      <td>-0.247558</td>
      <td>0.623192</td>
      <td>0.314351</td>
      <td>-1.698877</td>
      <td>-4836.224280</td>
      <td>-2991.607031</td>
      <td>-2960.663852</td>
    </tr>
  </tbody>
</table>
<p>1608 rows × 16 columns</p>
</div>




```python
np.savetxt('test.csv', X_test_avg_var_pca_glcm, delimiter=',')
```


```python
X_test_avg_var_pca_glcm=np.loadtxt('test.csv', delimiter=',')
```

### a. Ordinary Least Squares (OLS) regression


```python
h_var, r_var, g_var, b_var=var_H_R_G_B(dataset_rgb=X[(F==1)|(F==2)])
h_avg, r_avg, g_avg, b_avg=avg_H_R_G_B(dataset_rgb=X[(F==1)|(F==2)])
feature_pca = np.loadtxt('pca_train_feature.csv', delimiter=',')
glcm_feature=GLCM(X[(F==1)| (F==2)])

X_train_avg_var_pca_glcm=DataFrame(np.vstack([h_avg, r_avg, g_avg, b_avg,
                     h_var, r_var, g_var, b_var,
                     glcm_feature[:,0],glcm_feature[:,1],glcm_feature[:,2],glcm_feature[:,3]
                                 ,feature_pca[:,0],feature_pca[:,component],feature_pca[:,component*2],feature_pca[:,component*3]
                                             ]).T,
                     index=None,
                     columns = ['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var',
                                'glcm_contrast','glcm_dissimilarity','glcm_homogeneity','glcm_energy'
                                ,'h_pca1','r_pca1','g_pca1','b_pca1'
                               ])
```


```python
X_train_avg_var_pca_glcm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h_avg</th>
      <th>r_avg</th>
      <th>g_avg</th>
      <th>b_avg</th>
      <th>h_var</th>
      <th>r_var</th>
      <th>g_var</th>
      <th>b_var</th>
      <th>glcm_contrast</th>
      <th>glcm_dissimilarity</th>
      <th>glcm_homogeneity</th>
      <th>glcm_energy</th>
      <th>h_pca1</th>
      <th>r_pca1</th>
      <th>g_pca1</th>
      <th>b_pca1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.014908</td>
      <td>216.828201</td>
      <td>184.712616</td>
      <td>220.145798</td>
      <td>0.000166</td>
      <td>700.185283</td>
      <td>1525.001090</td>
      <td>464.010344</td>
      <td>-0.697989</td>
      <td>-0.429730</td>
      <td>-0.289784</td>
      <td>-0.271929</td>
      <td>-3.082497</td>
      <td>-5262.495331</td>
      <td>-7467.186806</td>
      <td>-4512.334671</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.018386</td>
      <td>210.432480</td>
      <td>175.731857</td>
      <td>216.199799</td>
      <td>0.000254</td>
      <td>968.860401</td>
      <td>1818.421676</td>
      <td>606.176359</td>
      <td>-0.131541</td>
      <td>0.176947</td>
      <td>-0.376335</td>
      <td>-0.293098</td>
      <td>-2.195719</td>
      <td>-3628.919399</td>
      <td>-5167.760808</td>
      <td>-3508.471849</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.022369</td>
      <td>203.084442</td>
      <td>163.588150</td>
      <td>211.209991</td>
      <td>0.000359</td>
      <td>1209.026744</td>
      <td>1786.483288</td>
      <td>673.088380</td>
      <td>0.496611</td>
      <td>0.693211</td>
      <td>-0.429394</td>
      <td>-0.305518</td>
      <td>-1.188094</td>
      <td>-1764.571364</td>
      <td>-2079.966308</td>
      <td>-2238.730017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011857</td>
      <td>224.299576</td>
      <td>196.641907</td>
      <td>225.250229</td>
      <td>0.000221</td>
      <td>851.487143</td>
      <td>1897.752842</td>
      <td>550.505211</td>
      <td>-0.401345</td>
      <td>-0.572601</td>
      <td>1.022201</td>
      <td>1.040711</td>
      <td>-3.856420</td>
      <td>-7154.484611</td>
      <td>-10470.614962</td>
      <td>-5805.932947</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005693</td>
      <td>238.598755</td>
      <td>228.551178</td>
      <td>237.590729</td>
      <td>0.000044</td>
      <td>246.522321</td>
      <td>616.419683</td>
      <td>224.905800</td>
      <td>-1.474067</td>
      <td>-1.883588</td>
      <td>1.673266</td>
      <td>1.280801</td>
      <td>-5.436792</td>
      <td>-10833.960645</td>
      <td>-18697.407448</td>
      <td>-8990.706671</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3368</th>
      <td>0.016086</td>
      <td>208.914917</td>
      <td>133.911880</td>
      <td>205.878021</td>
      <td>0.000217</td>
      <td>961.169152</td>
      <td>3388.763735</td>
      <td>645.759535</td>
      <td>-0.541020</td>
      <td>-0.500838</td>
      <td>-0.104642</td>
      <td>-0.222001</td>
      <td>-2.759941</td>
      <td>-3180.252887</td>
      <td>5660.253234</td>
      <td>-819.937110</td>
    </tr>
    <tr>
      <th>3369</th>
      <td>0.022559</td>
      <td>195.649033</td>
      <td>112.972305</td>
      <td>199.495331</td>
      <td>0.000215</td>
      <td>736.531561</td>
      <td>1125.877819</td>
      <td>376.756875</td>
      <td>-1.109030</td>
      <td>-0.896384</td>
      <td>-0.262346</td>
      <td>-0.260905</td>
      <td>-1.121683</td>
      <td>163.951473</td>
      <td>10908.493808</td>
      <td>769.842842</td>
    </tr>
    <tr>
      <th>3370</th>
      <td>0.023275</td>
      <td>196.148666</td>
      <td>118.573761</td>
      <td>202.451859</td>
      <td>0.000257</td>
      <td>777.945046</td>
      <td>1029.561729</td>
      <td>395.359560</td>
      <td>-0.777156</td>
      <td>-0.469580</td>
      <td>-0.334684</td>
      <td>-0.273128</td>
      <td>-0.948112</td>
      <td>20.769895</td>
      <td>9454.195954</td>
      <td>-0.582189</td>
    </tr>
    <tr>
      <th>3371</th>
      <td>0.022506</td>
      <td>195.817291</td>
      <td>113.379898</td>
      <td>199.627487</td>
      <td>0.000218</td>
      <td>746.081486</td>
      <td>1125.390361</td>
      <td>377.322553</td>
      <td>-1.117833</td>
      <td>-0.900360</td>
      <td>-0.264307</td>
      <td>-0.261779</td>
      <td>-1.139863</td>
      <td>116.113199</td>
      <td>10801.538070</td>
      <td>734.123672</td>
    </tr>
    <tr>
      <th>3372</th>
      <td>0.022524</td>
      <td>195.692642</td>
      <td>113.010788</td>
      <td>199.491989</td>
      <td>0.000214</td>
      <td>730.125578</td>
      <td>1098.273519</td>
      <td>371.541318</td>
      <td>-1.121249</td>
      <td>-0.910274</td>
      <td>-0.260635</td>
      <td>-0.260416</td>
      <td>-1.132089</td>
      <td>152.349591</td>
      <td>10900.090342</td>
      <td>771.122571</td>
    </tr>
  </tbody>
</table>
<p>3373 rows × 16 columns</p>
</div>




```python
y_train = Y.apply(lambda x:x.sum(),axis =1)[(F==1)|(F==2)]
y_test = Y.apply(lambda x:x.sum(),axis =1)[F==3]

# using var and avg feature

# fit model
reg_avg_var_glcm  = LinearRegression().fit(X_train_avg_var_pca_glcm[['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var']] , y_train)

# predict
y_pred_avg_var_glcm= reg_avg_var_glcm .predict(X_test_avg_var_pca_glcm[['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var']])
mse_avg_var_glcm= mean_squared_error(y_test, y_pred_avg_var_glcm)
r2_avg_var_glcm= r2_score(y_test, y_pred_avg_var_glcm)
pearson_avg_var_glcm=stats.pearsonr(y_test, y_pred_avg_var_glcm)
# performance matric
print("RMSE with avg and var: ", np.sqrt(mse_avg_var_glcm))
print("Pearson Correlation Coefficient with avg and var: ", pearson_avg_var_glcm[0])
print("R2 score with avg and var: ", r2_avg_var_glcm)
print("spearmanr Correlation Coefficient with avg and var: ", round(spearmanr(y_test, y_pred_avg_var_glcm)[0],5))
```

    RMSE with avg and var:  41.98806201221627
    Pearson Correlation Coefficient with avg and var:  0.6514203533364982
    R2 score with avg and var:  0.2738912063355402
    spearmanr Correlation Coefficient with avg and var:  0.68243
    


```python
y_train = Y.apply(lambda x:x.sum(),axis =1)[(F==1)|(F==2)]
y_test = Y.apply(lambda x:x.sum(),axis =1)[F==3]

# var and avg and glcm 


reg_avg_var_pca_glcm = LinearRegression().fit(X_train_avg_var_pca_glcm[['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var','glcm_contrast']] , y_train)


y_pred_avg_var_pca_glcm= reg_avg_var_pca_glcm .predict(X_test_avg_var_pca_glcm[['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var','glcm_contrast']])
mse_avg_var_pca_glcm= mean_squared_error(y_test, y_pred_avg_var_pca_glcm)
r2_avg_var_pca_glcm= r2_score(y_test, y_pred_avg_var_pca_glcm)
pearson_avg_var_pca_glcm=stats.pearsonr(y_test, y_pred_avg_var_pca_glcm)


print("RMSE with avg, var and glcm_contrast: ", np.sqrt(mse_avg_var_pca_glcm))
print("Pearson Correlation Coefficient with avg, var and glcm_contrast: ", pearson_avg_var_pca_glcm[0])
print("R2 score with avg, var and glcm_contrast: ", r2_avg_var_pca_glcm)
print("spearmanr Correlation Coefficient with avg, var and glcm_contrast:", round(spearmanr(y_test, y_pred_avg_var_pca_glcm)[0],5))
```

    RMSE with avg, var and glcm_contrast:  35.936294179379786
    Pearson Correlation Coefficient with avg, var and glcm_contrast:  0.7534273222199888
    R2 score with avg, var and glcm_contrast:  0.46811637977807063
    spearmanr Correlation Coefficient with avg, var and glcm_contrast: 0.75762
    


```python
H_pca_val_feature=pca_H_feature(X[F==2],907)
#将结果保存到CSV文件中
np.savetxt('H_pca_val_feature.csv', H_pca_val_feature, delimiter=',')
del H_pca_val_feature
print('1')
r_pca_val_feature=pca_rgb_feature(X[F==2],1,907)
# 将结果保存到CSV文件中
np.savetxt('r_pca_val_feature.csv', r_pca_val_feature, delimiter=',')
del r_pca_val_feature
print('1')
g_pca_val_feature=pca_rgb_feature(X[F==2],2,907)
# 将结果保存到CSV文件中
np.savetxt('g_pca_val_feature.csv', g_pca_val_feature, delimiter=',')
del g_pca_val_feature
print('1')
b_pca_val_feature=pca_rgb_feature(X[F==2],0,907)
# 将结果保存到CSV文件中
np.savetxt('b_pca_val_feature.csv', b_pca_val_feature, delimiter=',')
del b_pca_val_feature
print('1')
X_H_pca_val = np.loadtxt('H_pca_val_feature.csv', delimiter=',')
X_R_pca_val = np.loadtxt('r_pca_val_feature.csv', delimiter=',')
X_G_pca_val = np.loadtxt('g_pca_val_feature.csv', delimiter=',')
X_B_pca_val = np.loadtxt('b_pca_val_feature.csv', delimiter=',')
feature_pca_val = np.concatenate((X_H_pca_val, X_R_pca_val, X_G_pca_val, X_B_pca_val), axis=1)

np.savetxt('feature_pca_val.csv', feature_pca_val, delimiter=',')
del X_H_pca_val,X_R_pca_val,X_G_pca_val,X_B_pca_val
```

    1
    1
    1
    1
    


```python
feature_pca_val = np.loadtxt('feature_pca_val.csv', delimiter=',')
```


```python
h_avg_val, r_avg_val, g_avg_val, b_avg_val = avg_H_R_G_B(dataset_rgb=X[F==2])
h_var_val, r_var_val, g_var_val, b_var_val=var_H_R_G_B(dataset_rgb=X[F==2])


glcm_feature_val=GLCM(X[F==2])



X_val_avg_var_pca_glcm = DataFrame(np.vstack([h_avg_val, r_avg_val, g_avg_val, b_avg_val,
                     h_var_val, r_var_val, g_var_val, b_var_val,
                     glcm_feature_val[:,0],glcm_feature_val[:,1],glcm_feature_val[:,2],glcm_feature_val[:,3]
                                         ,feature_pca_val[:,0],feature_pca_val[:,component],feature_pca_val[:,component*2],feature_pca_val[:,component*3]
                                              ]).T,
                     index=None,
                     columns = ['h_avg', 'r_avg', 'g_avg','b_avg','h_var', 'r_var', 'g_var', 'b_var',
                                'glcm_contrast','glcm_dissimilarity','glcm_homogeneity','glcm_energy'
                                ,'h_pca1','r_pca1','g_pca1','b_pca1'
                               ])
```


```python
X_val_avg_var_pca_glcm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h_avg</th>
      <th>r_avg</th>
      <th>g_avg</th>
      <th>b_avg</th>
      <th>h_var</th>
      <th>r_var</th>
      <th>g_var</th>
      <th>b_var</th>
      <th>glcm_contrast</th>
      <th>glcm_dissimilarity</th>
      <th>glcm_homogeneity</th>
      <th>glcm_energy</th>
      <th>h_pca1</th>
      <th>r_pca1</th>
      <th>g_pca1</th>
      <th>b_pca1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.020428</td>
      <td>207.295578</td>
      <td>177.023788</td>
      <td>209.745300</td>
      <td>0.000538</td>
      <td>1835.419698</td>
      <td>3150.091399</td>
      <td>1406.434457</td>
      <td>0.434938</td>
      <td>0.258860</td>
      <td>0.027366</td>
      <td>-0.060358</td>
      <td>-1.266791</td>
      <td>-4642.081942</td>
      <td>-1617.212158</td>
      <td>-2180.936716</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.021255</td>
      <td>204.651840</td>
      <td>168.006271</td>
      <td>208.284653</td>
      <td>0.000468</td>
      <td>1449.368973</td>
      <td>2410.381873</td>
      <td>845.319898</td>
      <td>0.610392</td>
      <td>0.515831</td>
      <td>-0.203913</td>
      <td>-0.234798</td>
      <td>-1.045006</td>
      <td>-2340.828035</td>
      <td>-1244.298428</td>
      <td>-1487.514716</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.016852</td>
      <td>213.455811</td>
      <td>181.105698</td>
      <td>214.492783</td>
      <td>0.000404</td>
      <td>1351.265961</td>
      <td>2694.068921</td>
      <td>903.908822</td>
      <td>0.269018</td>
      <td>0.158243</td>
      <td>-0.091942</td>
      <td>-0.161125</td>
      <td>-2.186115</td>
      <td>-5712.999430</td>
      <td>-2842.229628</td>
      <td>-3759.758061</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.013158</td>
      <td>220.426193</td>
      <td>191.272064</td>
      <td>219.586090</td>
      <td>0.000258</td>
      <td>1042.492660</td>
      <td>2532.996690</td>
      <td>807.417607</td>
      <td>-0.474493</td>
      <td>-0.555833</td>
      <td>0.587690</td>
      <td>0.452193</td>
      <td>-3.126707</td>
      <td>-8350.767357</td>
      <td>-4160.333413</td>
      <td>-5553.234349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.009853</td>
      <td>228.749435</td>
      <td>210.292862</td>
      <td>228.040482</td>
      <td>0.000197</td>
      <td>754.511482</td>
      <td>1890.944734</td>
      <td>649.620813</td>
      <td>-0.810706</td>
      <td>-1.129172</td>
      <td>1.055033</td>
      <td>0.783435</td>
      <td>-3.972507</td>
      <td>-13223.036973</td>
      <td>-6330.177088</td>
      <td>-7680.139381</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1746</th>
      <td>0.016086</td>
      <td>208.914917</td>
      <td>133.911880</td>
      <td>205.878021</td>
      <td>0.000217</td>
      <td>961.169152</td>
      <td>3388.763735</td>
      <td>645.759535</td>
      <td>-0.590302</td>
      <td>-0.531446</td>
      <td>-0.120646</td>
      <td>-0.240584</td>
      <td>-2.366977</td>
      <td>6469.094913</td>
      <td>-611.331310</td>
      <td>-2562.332124</td>
    </tr>
    <tr>
      <th>1747</th>
      <td>0.022559</td>
      <td>195.649033</td>
      <td>112.972305</td>
      <td>199.495331</td>
      <td>0.000215</td>
      <td>736.531561</td>
      <td>1125.877819</td>
      <td>376.756875</td>
      <td>-1.172016</td>
      <td>-0.929111</td>
      <td>-0.272232</td>
      <td>-0.279789</td>
      <td>-0.725243</td>
      <td>11719.280705</td>
      <td>986.270563</td>
      <td>787.500642</td>
    </tr>
    <tr>
      <th>1748</th>
      <td>0.023275</td>
      <td>196.148666</td>
      <td>118.573761</td>
      <td>202.451859</td>
      <td>0.000257</td>
      <td>777.945046</td>
      <td>1029.561729</td>
      <td>395.359560</td>
      <td>-0.832135</td>
      <td>-0.500021</td>
      <td>-0.341763</td>
      <td>-0.292106</td>
      <td>-0.537483</td>
      <td>10264.584980</td>
      <td>215.092460</td>
      <td>658.451040</td>
    </tr>
    <tr>
      <th>1749</th>
      <td>0.022506</td>
      <td>195.817291</td>
      <td>113.379898</td>
      <td>199.627487</td>
      <td>0.000218</td>
      <td>746.081486</td>
      <td>1125.390361</td>
      <td>377.322553</td>
      <td>-1.181031</td>
      <td>-0.933108</td>
      <td>-0.274117</td>
      <td>-0.280670</td>
      <td>-0.742682</td>
      <td>11608.594374</td>
      <td>949.656735</td>
      <td>738.858270</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>0.022524</td>
      <td>195.692642</td>
      <td>113.010788</td>
      <td>199.491989</td>
      <td>0.000214</td>
      <td>730.125578</td>
      <td>1098.273519</td>
      <td>371.541318</td>
      <td>-1.184529</td>
      <td>-0.943076</td>
      <td>-0.270588</td>
      <td>-0.279295</td>
      <td>-0.733768</td>
      <td>11712.331682</td>
      <td>989.204175</td>
      <td>777.475722</td>
    </tr>
  </tbody>
</table>
<p>1751 rows × 16 columns</p>
</div>



### b. Support Vector Regression OR Multilayer Perceptron (MLP) OR Both


```python
a=F[(F==1)|(F==2)]
```


```python
X_fold1_avg_var=X_train_avg_var_pca_glcm[a==1]
```


```python
X_fold1_avg_var_glcm=X_fold1_avg_var.iloc[:,0:12]
X_fold1_avg_var_glcm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h_avg</th>
      <th>r_avg</th>
      <th>g_avg</th>
      <th>b_avg</th>
      <th>h_var</th>
      <th>r_var</th>
      <th>g_var</th>
      <th>b_var</th>
      <th>glcm_contrast</th>
      <th>glcm_dissimilarity</th>
      <th>glcm_homogeneity</th>
      <th>glcm_energy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.014908</td>
      <td>216.828201</td>
      <td>184.712616</td>
      <td>220.145798</td>
      <td>0.000166</td>
      <td>700.185283</td>
      <td>1525.001090</td>
      <td>464.010344</td>
      <td>-0.697989</td>
      <td>-0.429730</td>
      <td>-0.289784</td>
      <td>-0.271929</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.018386</td>
      <td>210.432480</td>
      <td>175.731857</td>
      <td>216.199799</td>
      <td>0.000254</td>
      <td>968.860401</td>
      <td>1818.421676</td>
      <td>606.176359</td>
      <td>-0.131541</td>
      <td>0.176947</td>
      <td>-0.376335</td>
      <td>-0.293098</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.022369</td>
      <td>203.084442</td>
      <td>163.588150</td>
      <td>211.209991</td>
      <td>0.000359</td>
      <td>1209.026744</td>
      <td>1786.483288</td>
      <td>673.088380</td>
      <td>0.496611</td>
      <td>0.693211</td>
      <td>-0.429394</td>
      <td>-0.305518</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011857</td>
      <td>224.299576</td>
      <td>196.641907</td>
      <td>225.250229</td>
      <td>0.000221</td>
      <td>851.487143</td>
      <td>1897.752842</td>
      <td>550.505211</td>
      <td>-0.401345</td>
      <td>-0.572601</td>
      <td>1.022201</td>
      <td>1.040711</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005693</td>
      <td>238.598755</td>
      <td>228.551178</td>
      <td>237.590729</td>
      <td>0.000044</td>
      <td>246.522321</td>
      <td>616.419683</td>
      <td>224.905800</td>
      <td>-1.474067</td>
      <td>-1.883588</td>
      <td>1.673266</td>
      <td>1.280801</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3344</th>
      <td>0.023362</td>
      <td>203.515259</td>
      <td>160.305359</td>
      <td>210.075638</td>
      <td>0.000723</td>
      <td>2019.279461</td>
      <td>3748.873522</td>
      <td>844.821534</td>
      <td>1.030406</td>
      <td>0.823593</td>
      <td>-0.343453</td>
      <td>-0.294803</td>
    </tr>
    <tr>
      <th>3345</th>
      <td>0.023167</td>
      <td>204.053421</td>
      <td>159.787094</td>
      <td>210.195587</td>
      <td>0.000743</td>
      <td>2078.308105</td>
      <td>3825.836827</td>
      <td>878.677139</td>
      <td>1.950368</td>
      <td>1.470824</td>
      <td>-0.388312</td>
      <td>-0.303971</td>
    </tr>
    <tr>
      <th>3346</th>
      <td>0.011381</td>
      <td>225.166473</td>
      <td>183.013504</td>
      <td>223.782059</td>
      <td>0.000336</td>
      <td>1028.205349</td>
      <td>2142.396073</td>
      <td>471.881960</td>
      <td>0.256597</td>
      <td>0.208119</td>
      <td>-0.335531</td>
      <td>-0.285099</td>
    </tr>
    <tr>
      <th>3347</th>
      <td>0.022770</td>
      <td>204.530136</td>
      <td>159.652847</td>
      <td>210.359131</td>
      <td>0.000719</td>
      <td>2019.715187</td>
      <td>3733.226027</td>
      <td>851.168297</td>
      <td>1.883133</td>
      <td>1.425582</td>
      <td>-0.386343</td>
      <td>-0.304430</td>
    </tr>
    <tr>
      <th>3348</th>
      <td>0.023147</td>
      <td>204.041397</td>
      <td>159.629395</td>
      <td>210.165939</td>
      <td>0.000739</td>
      <td>2071.619945</td>
      <td>3819.385631</td>
      <td>877.553626</td>
      <td>1.933327</td>
      <td>1.460801</td>
      <td>-0.388660</td>
      <td>-0.304377</td>
    </tr>
  </tbody>
</table>
<p>1622 rows × 12 columns</p>
</div>




```python
X_val_avg_var_glcm=X_val_avg_var_pca_glcm.iloc[:,0:12]
X_val_avg_var_glcm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h_avg</th>
      <th>r_avg</th>
      <th>g_avg</th>
      <th>b_avg</th>
      <th>h_var</th>
      <th>r_var</th>
      <th>g_var</th>
      <th>b_var</th>
      <th>glcm_contrast</th>
      <th>glcm_dissimilarity</th>
      <th>glcm_homogeneity</th>
      <th>glcm_energy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.020428</td>
      <td>207.295578</td>
      <td>177.023788</td>
      <td>209.745300</td>
      <td>0.000538</td>
      <td>1835.419698</td>
      <td>3150.091399</td>
      <td>1406.434457</td>
      <td>0.434938</td>
      <td>0.258860</td>
      <td>0.027366</td>
      <td>-0.060358</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.021255</td>
      <td>204.651840</td>
      <td>168.006271</td>
      <td>208.284653</td>
      <td>0.000468</td>
      <td>1449.368973</td>
      <td>2410.381873</td>
      <td>845.319898</td>
      <td>0.610392</td>
      <td>0.515831</td>
      <td>-0.203913</td>
      <td>-0.234798</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.016852</td>
      <td>213.455811</td>
      <td>181.105698</td>
      <td>214.492783</td>
      <td>0.000404</td>
      <td>1351.265961</td>
      <td>2694.068921</td>
      <td>903.908822</td>
      <td>0.269018</td>
      <td>0.158243</td>
      <td>-0.091942</td>
      <td>-0.161125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.013158</td>
      <td>220.426193</td>
      <td>191.272064</td>
      <td>219.586090</td>
      <td>0.000258</td>
      <td>1042.492660</td>
      <td>2532.996690</td>
      <td>807.417607</td>
      <td>-0.474493</td>
      <td>-0.555833</td>
      <td>0.587690</td>
      <td>0.452193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.009853</td>
      <td>228.749435</td>
      <td>210.292862</td>
      <td>228.040482</td>
      <td>0.000197</td>
      <td>754.511482</td>
      <td>1890.944734</td>
      <td>649.620813</td>
      <td>-0.810706</td>
      <td>-1.129172</td>
      <td>1.055033</td>
      <td>0.783435</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1746</th>
      <td>0.016086</td>
      <td>208.914917</td>
      <td>133.911880</td>
      <td>205.878021</td>
      <td>0.000217</td>
      <td>961.169152</td>
      <td>3388.763735</td>
      <td>645.759535</td>
      <td>-0.590302</td>
      <td>-0.531446</td>
      <td>-0.120646</td>
      <td>-0.240584</td>
    </tr>
    <tr>
      <th>1747</th>
      <td>0.022559</td>
      <td>195.649033</td>
      <td>112.972305</td>
      <td>199.495331</td>
      <td>0.000215</td>
      <td>736.531561</td>
      <td>1125.877819</td>
      <td>376.756875</td>
      <td>-1.172016</td>
      <td>-0.929111</td>
      <td>-0.272232</td>
      <td>-0.279789</td>
    </tr>
    <tr>
      <th>1748</th>
      <td>0.023275</td>
      <td>196.148666</td>
      <td>118.573761</td>
      <td>202.451859</td>
      <td>0.000257</td>
      <td>777.945046</td>
      <td>1029.561729</td>
      <td>395.359560</td>
      <td>-0.832135</td>
      <td>-0.500021</td>
      <td>-0.341763</td>
      <td>-0.292106</td>
    </tr>
    <tr>
      <th>1749</th>
      <td>0.022506</td>
      <td>195.817291</td>
      <td>113.379898</td>
      <td>199.627487</td>
      <td>0.000218</td>
      <td>746.081486</td>
      <td>1125.390361</td>
      <td>377.322553</td>
      <td>-1.181031</td>
      <td>-0.933108</td>
      <td>-0.274117</td>
      <td>-0.280670</td>
    </tr>
    <tr>
      <th>1750</th>
      <td>0.022524</td>
      <td>195.692642</td>
      <td>113.010788</td>
      <td>199.491989</td>
      <td>0.000214</td>
      <td>730.125578</td>
      <td>1098.273519</td>
      <td>371.541318</td>
      <td>-1.184529</td>
      <td>-0.943076</td>
      <td>-0.270588</td>
      <td>-0.279295</td>
    </tr>
  </tbody>
</table>
<p>1751 rows × 12 columns</p>
</div>




```python
X_test_avg_var_glcm=X_test_avg_var_pca_glcm.iloc[:,0:12]
X_test_avg_var_glcm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h_avg</th>
      <th>r_avg</th>
      <th>g_avg</th>
      <th>b_avg</th>
      <th>h_var</th>
      <th>r_var</th>
      <th>g_var</th>
      <th>b_var</th>
      <th>glcm_contrast</th>
      <th>glcm_dissimilarity</th>
      <th>glcm_homogeneity</th>
      <th>glcm_energy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.028605</td>
      <td>197.698242</td>
      <td>170.029984</td>
      <td>220.461472</td>
      <td>0.000512</td>
      <td>1331.335242</td>
      <td>1876.016023</td>
      <td>481.998760</td>
      <td>0.810735</td>
      <td>0.965928</td>
      <td>-0.439103</td>
      <td>-0.362401</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.026077</td>
      <td>202.473419</td>
      <td>176.373886</td>
      <td>222.857880</td>
      <td>0.000533</td>
      <td>1417.460078</td>
      <td>2178.975367</td>
      <td>509.681401</td>
      <td>0.857835</td>
      <td>0.828788</td>
      <td>-0.208924</td>
      <td>-0.183399</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.029114</td>
      <td>197.348557</td>
      <td>170.161255</td>
      <td>220.063202</td>
      <td>0.000586</td>
      <td>1539.463942</td>
      <td>2307.646238</td>
      <td>535.724338</td>
      <td>0.507328</td>
      <td>0.578942</td>
      <td>0.009299</td>
      <td>0.063534</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.023964</td>
      <td>205.994827</td>
      <td>180.647141</td>
      <td>223.980301</td>
      <td>0.000504</td>
      <td>1436.195576</td>
      <td>2438.515856</td>
      <td>513.623590</td>
      <td>0.575520</td>
      <td>0.579722</td>
      <td>-0.002403</td>
      <td>0.007661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.033372</td>
      <td>188.374313</td>
      <td>154.436249</td>
      <td>214.194946</td>
      <td>0.000475</td>
      <td>1233.496379</td>
      <td>1642.879389</td>
      <td>537.033895</td>
      <td>0.341959</td>
      <td>0.632068</td>
      <td>-0.475922</td>
      <td>-0.378003</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1603</th>
      <td>0.017153</td>
      <td>212.572479</td>
      <td>172.512909</td>
      <td>217.049973</td>
      <td>0.000274</td>
      <td>964.245205</td>
      <td>1629.919114</td>
      <td>580.884420</td>
      <td>0.546894</td>
      <td>0.683091</td>
      <td>-0.399490</td>
      <td>-0.348863</td>
    </tr>
    <tr>
      <th>1604</th>
      <td>0.020643</td>
      <td>206.418716</td>
      <td>165.679520</td>
      <td>209.284439</td>
      <td>0.000566</td>
      <td>1766.249466</td>
      <td>3407.630187</td>
      <td>1057.736126</td>
      <td>0.213481</td>
      <td>0.095125</td>
      <td>0.130653</td>
      <td>-0.015457</td>
    </tr>
    <tr>
      <th>1605</th>
      <td>0.023952</td>
      <td>202.140808</td>
      <td>166.652145</td>
      <td>209.097931</td>
      <td>0.000644</td>
      <td>2090.342844</td>
      <td>4087.511367</td>
      <td>1211.674858</td>
      <td>-0.407248</td>
      <td>-0.393793</td>
      <td>0.284078</td>
      <td>0.079372</td>
    </tr>
    <tr>
      <th>1606</th>
      <td>0.023798</td>
      <td>202.249878</td>
      <td>164.874191</td>
      <td>208.633499</td>
      <td>0.000661</td>
      <td>2105.959045</td>
      <td>4008.991847</td>
      <td>1247.109223</td>
      <td>0.660789</td>
      <td>0.373970</td>
      <td>0.327851</td>
      <td>0.132698</td>
    </tr>
    <tr>
      <th>1607</th>
      <td>0.023468</td>
      <td>203.599594</td>
      <td>170.092133</td>
      <td>210.553970</td>
      <td>0.000671</td>
      <td>2160.485748</td>
      <td>4406.558559</td>
      <td>1264.357805</td>
      <td>0.001142</td>
      <td>-0.247558</td>
      <td>0.623192</td>
      <td>0.314351</td>
    </tr>
  </tbody>
</table>
<p>1608 rows × 12 columns</p>
</div>




```python

scaler = StandardScaler()
X_fold1_avg_var_glcm = scaler.fit_transform(X_fold1_avg_var_glcm)
X_test_avg_var_glcm = scaler.transform(X_test_avg_var_glcm)
X_val_avg_var_glcm = scaler.transform(X_val_avg_var_glcm)
```


```python
y_test = Y.apply(lambda x:x.sum(),axis =1)[F==3]
validation_Y=Y.apply(lambda x:x.sum(),axis =1)[F==2]
y_fold_train=Y.apply(lambda x:x.sum(),axis =1)[F==1]

# tuning parameter
param_grid = {'C': [300,400,500]}
svr_linear = LinearSVR(random_state=5)
grid_search_linear = GridSearchCV(svr_linear, param_grid, refit=True)
# fit model
y_linear = grid_search_linear.fit(X_val_avg_var_glcm, validation_Y)
print(y_linear.best_params_)
# svr
linearsvr = LinearSVR(**y_linear.best_params_).fit(X_fold1_avg_var_glcm,y_fold_train)

y_pred = linearsvr.predict(X_test_avg_var_glcm)

# predict

linearsvr_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),5)
linearsvr_pearson = round(stats.pearsonr(y_test,y_pred)[0],5)
linearsvrspearman = round(stats.spearmanr(y_test,y_pred)[0],5)
linearsvr_r2 = round(r2_score(y_test,y_pred),5)

print("RMSE : ", linearsvr_rmse)
print("R2 score: ", linearsvr_r2)
print("pearson : ", linearsvr_pearson)
print("spearman: ", linearsvrspearman)
      
```

    {'C': 400}
    RMSE :  40.83638
    R2 score:  0.31318
    pearson :  0.67202
    spearman:  0.70905
    


```python
np.random.seed(90)

param_grid = {'C': [1,5,10,20,30,50,70]}
svr_linear = SVR(kernel='poly' )
# GridSearchCV 
grid_search_linear = GridSearchCV(svr_linear, param_grid, refit=True)

y_linear = grid_search_linear.fit(X_val_avg_var_glcm, validation_Y)
print(y_linear.best_params_)
# svr
linearsvr = LinearSVR(**y_linear.best_params_).fit(X_fold1_avg_var_glcm,y_fold_train)

y_pred = linearsvr.predict(X_test_avg_var_glcm)

# performance

linearsvr_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)),5)
linearsvr_pearson = round(stats.pearsonr(y_test,y_pred)[0],5)
linearsvrspearman = round(stats.spearmanr(y_test,y_pred)[0],5)
linearsvr_r2 = round(r2_score(y_test,y_pred),5)

print("RMSE : ", linearsvr_rmse)
print("R2 score: ", linearsvr_r2)
print("pearson : ", linearsvr_pearson)
print("spearman: ", linearsvrspearman)
```

    {'C': 20}
    RMSE :  38.60138
    R2 score:  0.3863
    pearson :  0.70174
    spearman:  0.73372
    


```python
# MLP
mlp = MLPClassifier()

param_grid = {
    'hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(mlp, param_grid,refit=True, verbose=1, n_jobs=-1)
grid_search.fit(X_val_avg_var_glcm, validation_Y)


best_params = grid_search.best_params_
print('Best Params:', best_params)

mlp_model = MLPClassifier(**grid_search.best_params_).fit(X_fold1_avg_var_glcm,y_fold_train)

y_pred_mlp = mlp_model.predict(X_test_avg_var_glcm)


mlp_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred_mlp)),5)
mlp_pearson = round(stats.pearsonr(y_test,y_pred_mlp)[0],5)
mlpspearman = round(stats.spearmanr(y_test,y_pred_mlp)[0],5)
mlp_r2 = round(r2_score(y_test,y_pred_mlp),5)

print("RMSE : ", mlp_rmse)
print("R2 score: ", mlp_r2)
print("pearson : ", mlp_pearson)
print("spearman: ", mlpspearman)
```

    Fitting 5 folds for each of 48 candidates, totalling 240 fits
    Best Params: {'activation': 'tanh', 'alpha': 0.01, 'hidden_layer_sizes': (256, 128), 'solver': 'adam'}
    RMSE :  39.18198
    R2 score:  0.3677
    pearson :  0.71067
    spearman:  0.72201
    

# Question No. 3 (Using Convolutional Neural Networks)

### a. Use a convolutional neural network (in PyTorch) to solve this problem in much the same was as in part (ii) of Question (2). You are to develop an architecture of the neural network that takes an image directly as input and produces a count as the output corresponding to the total number of cells. You are free to choose any network structure as long as you can show that it gives good performance. Report your results on the test examples by plotting the scatter plot between true and predicted counts on the test data. Also, report your results interms of RMSE, Pearson Correlation Coefficient, Spearman Correlation Coefficient and R2 score. You will be evaluated on the design of your machine learning model and final performance metrics. Try to get the best test performance you can. Please include convergence plots in your submission showing how does loss change over training epochs.


```python
X_train=X[F==1]
Y_train=np.array(Y[F==1].apply(lambda x:x.sum(),axis =1))

X_val=X[F==2]
Y_val=np.array(Y[F==2].apply(lambda x:x.sum(),axis =1))

X_test=X[F==3]
Y_test=np.array(Y[F==3].apply(lambda x:x.sum(),axis =1))

```


```python
USE_CUDA = torch.cuda.is_available() 
from torch.autograd import Variable
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):       
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()
def self_loss_function(model, loss_fn, val_loader, device):
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [],[],[]
        for X, Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            model.to(device)
            preds = model(toTensor(X.reshape(-1,3,256,256)))
            loss  = loss_fn(preds, Y)
            losses.append(loss.item())
            #Y_shuffled.append(Y)
            #Y_preds.append(preds.argmax(dim=-1))
        #Y_shuffled = torch.cat(Y_shuffled)
        #Y_preds    = torch.cat(Y_preds)
        valid_loss = torch.tensor(losses).mean()
        return valid_loss

print('Using CUDA:',USE_CUDA) 
```

    Using CUDA: True
    


```python
class MyDataset(Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label=label
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, item):
        img= self.data[item]
        img = torch.from_numpy(img).type(torch.FloatTensor)
        new_label = np.array(self.label[item])
        new_label = torch.from_numpy(new_label).type(torch.FloatTensor).expand(1)
        return img, new_label


```


```python
def transform_data(X_train,Y_train,X_val,Y_val,X_test,Y_test):
    train = MyDataset(X_train.astype(float)/255, Y_train.astype(float))
    val = MyDataset(X_val.astype(float)/255, Y_val.astype(float))
    test= MyDataset(X_test.astype(float)/255, Y_test.astype(float))

    train_loader = DataLoader(dataset=train, batch_size=200, shuffle=True)
    val_loader=DataLoader(dataset=val, batch_size=200, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=200, shuffle=True)
    return train_loader,val_loader,test_loader
```


```python
train_loader,val_loader,test_loader=transform_data(X_train,Y_train,X_val,Y_val,X_test,Y_test)
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
num_epochs = 50
learning_rate = 0.001
```


```python
class CellCountingCNN(nn.Module):
    def __init__(self):
        super(CellCountingCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        
        self.fc1 = nn.Linear(64,100)
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
```


```python

model = CellCountingCNN()
model.to(device)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=5*(1e-4))

# Train the model
total_step = len(train_loader)
Loss_value=[]
val_Loss_value=[]
val_loss_com=5000
for epoch in range(num_epochs):
    loss_list=[]
    for  i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(toTensor(images.reshape(-1,3,256,256)))
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    train_loss = torch.tensor(loss_list).mean()
    Loss_value.append(train_loss)

    print("Train Loss :{:.4f}".format(train_loss))
    val_loss=self_loss_function(model,criterion,val_loader,device)
    print("Valid Loss :{:.4f}".format(val_loss))
    val_Loss_value.append(val_loss)

    #Save the best parameters so far
    if (val_loss+train_loss)<val_loss_com:
      # Save the model checkpoint
        torch.save(model.state_dict(), 'model.ckpt')
        val_loss_com = val_loss+train_loss


```

    Train Loss :12286.9912
    Valid Loss :9803.9775
    Train Loss :10896.4971
    Valid Loss :8685.1924
    Train Loss :9801.3916
    Valid Loss :7069.8506
    Train Loss :7204.5718
    Valid Loss :4939.9609
    Train Loss :4964.3247
    Valid Loss :3043.8372
    Train Loss :3070.0264
    Valid Loss :1691.3086
    Train Loss :1531.8237
    Valid Loss :1234.2434
    Train Loss :948.7203
    Valid Loss :1373.9131
    Train Loss :930.8570
    Valid Loss :1504.0717
    Train Loss :925.3881
    Valid Loss :1258.4585
    Train Loss :800.5864
    Valid Loss :1107.7836
    Train Loss :728.9142
    Valid Loss :1365.3619
    Train Loss :606.1311
    Valid Loss :1331.4905
    Train Loss :554.3752
    Valid Loss :1105.2587
    Train Loss :544.3751
    Valid Loss :1028.7798
    Train Loss :502.1111
    Valid Loss :1220.0305
    Train Loss :387.4120
    Valid Loss :1083.3373
    Train Loss :366.6681
    Valid Loss :1165.0359
    Train Loss :272.8668
    Valid Loss :1255.3258
    Train Loss :303.8605
    Valid Loss :1304.9406
    Train Loss :263.8563
    Valid Loss :961.0500
    Train Loss :247.1232
    Valid Loss :1179.0732
    Train Loss :231.1799
    Valid Loss :1260.6843
    Train Loss :196.3404
    Valid Loss :1076.7340
    Train Loss :167.2694
    Valid Loss :1158.3534
    Train Loss :217.1888
    Valid Loss :1117.8833
    Train Loss :150.1918
    Valid Loss :1078.3545
    Train Loss :126.3455
    Valid Loss :1170.8247
    Train Loss :101.8822
    Valid Loss :1138.2800
    Train Loss :91.0054
    Valid Loss :1194.6458
    Train Loss :121.8547
    Valid Loss :1211.0028
    Train Loss :105.8811
    Valid Loss :1175.2450
    Train Loss :62.2677
    Valid Loss :1133.4409
    Train Loss :57.9130
    Valid Loss :1136.8413
    Train Loss :67.5798
    Valid Loss :1292.9912
    Train Loss :95.1061
    Valid Loss :1046.7925
    Train Loss :71.2762
    Valid Loss :1245.4274
    Train Loss :62.8693
    Valid Loss :1215.4402
    Train Loss :51.4584
    Valid Loss :1153.8167
    Train Loss :68.3907
    Valid Loss :1161.5161
    Train Loss :67.1277
    Valid Loss :1222.3154
    Train Loss :46.9026
    Valid Loss :1102.9030
    Train Loss :44.9595
    Valid Loss :1175.1011
    Train Loss :41.8032
    Valid Loss :1232.6340
    Train Loss :43.5625
    Valid Loss :1119.5381
    Train Loss :31.8866
    Valid Loss :1269.8796
    Train Loss :37.4059
    Valid Loss :1228.6127
    Train Loss :33.9948
    Valid Loss :1233.2611
    Train Loss :29.3784
    Valid Loss :1120.1033
    Train Loss :28.4105
    Valid Loss :1268.6351
    


```python

```


```python
plt.plot(Loss_value,label='Train Loss')
plt.plot(val_Loss_value,label='Validation Loss')
plt.legend()
plt.title('Convergence Plots')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.grid()
```


    
![png](output_89_0.png)
    



```python
# get model
model.load_state_dict(torch.load('model.ckpt'))
```




    <All keys matched successfully>




```python
def test( model,test_loader, device):
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load('model.ckpt'))
    with torch.no_grad():
        test_loss = 0
        pred_y_value,true_y_value = [],[]
        for X, Y in test_loader:
            X,Y= X.to(device),Y.to(device)
            model.to(device)
            predict = model(toTensor(X.reshape(-1,3,256,256)))
            loss = criterion(predict, Y)
            test_loss += loss.item() * X.size(0)

            true_y_value.extend(Y.cpu().detach().numpy().flatten())
            pred_y_value.extend(predict.cpu().detach().numpy().flatten())
    
    rmse_test = round(np.sqrt(mean_squared_error(true_y_value, pred_y_value)),5)
    pearson_test = round(pearsonr(true_y_value, pred_y_value)[0],5)
    spearman_test = round(spearmanr(true_y_value, pred_y_value)[0],5)
    r2_test = round(r2_score(true_y_value, pred_y_value),5)
    
    return true_y_value, pred_y_value, rmse_test, pearson_test, spearman_test, r2_test
```


```python
true_y_value, pred_y_value, rmse_test, pearson_test, spearman_test, r2_test= test(model,test_loader, device)
```


```python
print('Performance matrix of CNN: ')
print('RMSE:',rmse_test)
print('R^2:',r2_test)
print('Pearson Correlation Coefficient:',pearson_test)
print('Spearman Correlation Coefficient:',spearman_test)
```

    Performance matrix of CNN: 
    RMSE: 37.98692
    R^2: 0.40568
    Pearson Correlation Coefficient: 0.70192
    Spearman Correlation Coefficient: 0.72546
    


```python
plt.scatter(true_y_value,pred_y_value,s=2,alpha=0.8)
plt.title('CNN True vs. Predict Y')
plt.xlabel('True Y')
plt.ylabel('Predict Y')
```




    Text(0, 0.5, 'Predict Y')




    
![png](output_94_1.png)
    


### b. Use a convolutional neural network (in Pytorch) to predict the counts of 6 types of cells simultaneously given the image patch as input as well as the total number of cells (7 outputs in total). You are free to choose any network structure as long as you can show that it gives good cross-validation performance. Report the results for the test fold for each cell type in the form of separate predicted-vs-actual count scatter plots (3 folds, 6 cell types and 1 as the total number of cells so 21 plots in total) using your optimal machine learning model and report your results in terms of RMSE, Pearson Correlation Coefficient, Spearman Correlation Coefficient and R2 score for each cell type and the total number of cells. [20 Marks]

# Note：the plots are at the end of output area


```python
New_Y=Y
```


```python
New_Y["total"] = New_Y[:].sum(axis =1)
New_Y.neutrophil
```




    0       0
    1       0
    2       0
    3       0
    4       0
           ..
    4976    0
    4977    0
    4978    0
    4979    0
    4980    0
    Name: neutrophil, Length: 4981, dtype: int64




```python
New_Y_fold1=New_Y[F==1]
New_Y_fold2=New_Y[F==2]
New_Y_fold3=New_Y[F==3]
```


```python
#fold3 test is train
#fold2 val is test
#fold1 train is val
num_epochs=10
Type=['neutrophil','epithelial','lymphocyte','plasma','eosinophil','connective','total']
model_P_M=[['','rmse_test', 'pearson_test', 'spearman_test', 'r2_test']]

#train_loader,val_loader,test_loader=transform_data(X_train,Y_train,X_val,Y_val,X_test,Y_test)
fig, axs = plt.subplots(1, 7,figsize=(30,4))

for i in range(7):

    y_train3=np.array(New_Y_fold3[Type[i]])
    y_val1=np.array(New_Y_fold1[Type[i]])
    y_test2=np.array(New_Y_fold2[Type[i]])
    
    # process data
    train_loader,val_loader,test_loader=transform_data(X_test,y_train3,X_train,y_val1,X_val,y_test2)

    model1 = CellCountingCNN()
    model1.to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate,weight_decay=5*(1e-4))
    
    # Train the model
    total_step = len(train_loader)

    val_loss_com=5000
    for epoch in range(num_epochs):
        loss_list=[]
        for  images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model1(toTensor(images.reshape(-1,3,256,256)))
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        train_loss = torch.tensor(loss_list).mean()

        print("Train Loss :{:.4f}".format(train_loss))
        val_loss=self_loss_function(model1,criterion,val_loader,device)
        print("Valid Loss :{:.4f}".format(val_loss))


        #Save the best parameters so far
        if (val_loss+train_loss)<val_loss_com:
          # Save the model checkpoint
            torch.save(model1.state_dict(), 'model1.ckpt')
            val_loss_com = val_loss+train_loss

    # get model
    model1.load_state_dict(torch.load('model1.ckpt'))
    true_y_value, pred_y_value, rmse_test, pearson_test, spearman_test, r2_test= test(model1,test_loader, device)
    model_P_M.append([Type[i],rmse_test, pearson_test, spearman_test, r2_test])
    axs[i].scatter(true_y_value, pred_y_value,s=4,alpha=0.8)
    axs[i].set_xlabel('True Y')
    axs[i].set_ylabel('Predict Y')
    axs[i].set_title(Type[i]+':True vs Predict')

```


```python


#fold3 test is val 
#fold2 val is train
#fold1 train is test

num_epochs=10
Type=['neutrophil','epithelial','lymphocyte','plasma','eosinophil','connective','total']
model_P_M2=[['','rmse_test', 'pearson_test', 'spearman_test', 'r2_test']]
#train_loader,val_loader,test_loader=transform_data(X_train,Y_train,X_val,Y_val,X_test,Y_test)
fig, axs = plt.subplots(1, 7,figsize=(30,4))

for i in range(7):

    y_train3=np.array(New_Y_fold3[Type[i]])
    y_val1=np.array(New_Y_fold1[Type[i]])
    y_test2=np.array(New_Y_fold2[Type[i]])
    
    # process data
    train_loader,val_loader,test_loader=transform_data(X_val,y_test2,X_test,y_train3,X_train,y_val1)

    model2 = CellCountingCNN()
    model2.to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate,weight_decay=5*(1e-4))

    # Train the model2
    total_step = len(train_loader)

    val_loss_com=5000
    for epoch in range(num_epochs):
        loss_list=[]
        for  images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model2(toTensor(images.reshape(-1,3,256,256)))
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        train_loss = torch.tensor(loss_list).mean()

        print("Train Loss :{:.4f}".format(train_loss))
        val_loss=self_loss_function(model2,criterion,val_loader,device)
        print("Valid Loss :{:.4f}".format(val_loss))


        #Save the best parameters so far
        if (val_loss+train_loss)<val_loss_com:
          # Save the model2 checkpoint
            torch.save(model2.state_dict(), 'model2.ckpt')
            val_loss_com = val_loss+train_loss

    # get model
    model2.load_state_dict(torch.load('model2.ckpt'))
    true_y_value, pred_y_value, rmse_test, pearson_test, spearman_test, r2_test= test(model2,test_loader, device)
    
    model_P_M2.append([Type[i],rmse_test, pearson_test, spearman_test, r2_test])
    axs[i].scatter(true_y_value, pred_y_value,s=4,alpha=0.8)
    axs[i].set_xlabel('True Y')
    axs[i].set_ylabel('Predict Y')
    axs[i].set_title(Type[i]+':True vs Predict')

```


```python

```


```python
#fold1 train is train X_train,y_val1
#fold2 val is val     X_val,y_test2
#fold 3 test is test  X_test,y_train3


num_epochs=10
Type=['neutrophil','epithelial','lymphocyte','plasma','eosinophil','connective','total']

#train_loader,val_loader,test_loader=transform_data(X_train,Y_train,X_val,Y_val,X_test,Y_test)
fig, axs = plt.subplots(1, 7,figsize=(30,4))

model_P_M3=[['','rmse_test', 'pearson_test', 'spearman_test', 'r2_test']]
for i in range(7):

    y_train3=np.array(New_Y_fold3[Type[i]])
    y_val1=np.array(New_Y_fold1[Type[i]])
    y_test2=np.array(New_Y_fold2[Type[i]])
    
    # process data
    train_loader,val_loader,test_loader=transform_data(X_train,y_val1,X_val,y_test2,X_test,y_train3)

    model3 = CellCountingCNN()
    model3.to(device)
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate,weight_decay=5*(1e-4))

    # Train the model
    total_step = len(train_loader)

    val_loss_com=5000
    for epoch in range(num_epochs):
        loss_list=[]
        for  images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model3(toTensor(images.reshape(-1,3,256,256)))
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        train_loss = torch.tensor(loss_list).mean()

        print("Train Loss :{:.4f}".format(train_loss))
        val_loss=self_loss_function(model3,criterion,val_loader,device)
        print("Valid Loss :{:.4f}".format(val_loss))


        #Save the best parameters so far
        if (val_loss)<val_loss_com:
          # Save the model checkpoint
            torch.save(model3.state_dict(), 'model3.ckpt')
            val_loss_com = val_loss

    # get model
    model3.load_state_dict(torch.load('model3.ckpt'))
    true_y_value, pred_y_value, rmse_test, pearson_test, spearman_test, r2_test= test(model3,test_loader, device)
    model_P_M3.append([Type[i],rmse_test, pearson_test, spearman_test, r2_test])
    axs[i].scatter(true_y_value, pred_y_value,s=4,alpha=0.8)
    axs[i].set_xlabel('True Y')
    axs[i].set_ylabel('Predict Y')
    axs[i].set_title(Type[i]+':True vs Predict')

```


```python
print('fold 3 as train dataset,fold 2 as test dataset and fold 1 as validation dataset')
print(tabulate(model_P_M))
print('fold 2 as train dataset,fold 1 as test dataset and fold 3 as validation dataset')
print(tabulate(model_P_M2))
print('fold 1 as train dataset,fold 3 as test dataset and fold 2 as validation dataset')
print(tabulate(model_P_M3))
```

    fold 3 as train dataset,fold 2 as test dataset and fold 1 as validation dataset
    ----------  ------------------  ------------  -------------  -----------
                rmse_test           pearson_test  spearman_test  r2_test
    neutrophil  101.18834686279297  0.08328       0.0836         -1021.28971
    epithelial  60.888671875        0.52295       0.5042         -2.34161
    lymphocyte  85.3473129272461    0.4008        0.54387        -5.71069
    plasma      97.23710632324219   0.33485       0.41472        -165.74172
    eosinophil  101.4283218383789   0.21741       0.27468        -3748.50265
    connective  84.12727355957031   0.29073       0.34544        -25.90799
    total       34.051151275634766  0.7541        0.77062        0.55556
    ----------  ------------------  ------------  -------------  -----------
    fold 2 as train dataset,fold 1 as test dataset and fold 3 as validation dataset
    ----------  ------------------  ------------  -------------  -----------
                rmse_test           pearson_test  spearman_test  r2_test
    neutrophil  106.423828125       -0.01491      -0.00199       -952.43819
    epithelial  62.00762176513672   0.55447       0.55868        -1.71552
    lymphocyte  83.3658218383789    0.65435       0.66149        -7.40922
    plasma      100.8328628540039   0.37235       0.50148        -150.05577
    eosinophil  106.35932159423828  0.3103        0.30767        -6401.30848
    connective  86.27129364013672   0.39761       0.42176        -24.40514
    total       9.27322006225586    0.98682       0.98609        0.96625
    ----------  ------------------  ------------  -------------  -----------
    fold 1 as train dataset,fold 3 as test dataset and fold 2 as validation dataset
    ----------  ------------------  ------------  -------------  -----------
                rmse_test           pearson_test  spearman_test  r2_test
    neutrophil  103.21112060546875  0.12931       0.17862        -4018.35369
    epithelial  69.00921630859375   0.45171       0.43824        -3.09771
    lymphocyte  88.42082977294922   0.3735        0.58704        -7.95444
    plasma      99.08627319335938   0.18966       0.31506        -131.03655
    eosinophil  102.92286682128906  0.26063       0.30308        -4185.98924
    connective  86.38395690917969   0.27566       0.29348        -21.86095
    total       39.928890228271484  0.68055       0.7105         0.34336
    ----------  ------------------  ------------  -------------  -----------
    


```python

```
