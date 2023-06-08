<div align="center">
      <h1> <br/>IBM Automating visual inspection with Machine Learning (ML)</h1>
     </div>

### Abstract
In this lab, we will study how the quality of agricultural products can be classified based on photographs. Through the example of evaluating the quality of lemons, we will learn to form a DataSet from photographs and compare different types of classifiers. At the end, we will create a report that forms a DataSet of the lemons quality.

### Introduction
Today, various production lines and grocery supermarkets use video and photo images to evaluate product quality or find defects. To do this, special information systems are created that allow you to do it automatically. This lab will show you how to upload images, transform them and identify the key features that underlie the classification of goods or agricultural products.
Various classifiers will be analyzed and a function will be created that automatically regulates the product quality data set.

### Materials and methods
In this lab, we will learn the basic methods of images classification. The laboratory consists of four stages:
Download and preliminary transformation of images
Create image features
Compare different classical classification methods
Create function for lemon quality classification
The statistical data was obtained from https://www.kaggle.com/qiujiahui/lemondata under GPL 2 license.
### Prerequisites
- Python - basic level
- numpy - middle level
- SeaBorn - basic level
- Matplotlib - basic level
- mahotas - middle level
- scikit-learn - middle level
- pandas - basic level

### Objectives
* After completing this lab, you will be able to:
* Download and transform images
* Create features of images
* Build different classification models
* Build a DataSet with quality level of agricultural products.
* Download and preliminary transformation of images

### Required libraries installation
We need to install additional libraries and upgrade the existing ones in the lab.

``` python
# conda install matplotlib >= 3.4
# conda config --add channels conda-forge
pip install mahotas
conda install scikit-learn

```

### Required libraries import
Here we will use Mahotas for image processing, Keras library for creating our CNN model and its training. We will also use Matplotlib and Seaborn for visualizing our dataset to gain a better understanding of the images we are going to be handling. We will also use libraries os and glob to work with files and folders. NumPy - for arrays of images. Scikit-Learn - for classical classification models. Pandas - for present comparison of classifiers.

``` python
import mahotas as mh
import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

#Classifiers
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix

```

### Data loading
For convenience, let's create a function that downloads all and displays first 50 pictures from a specified directory. All training pictures should be placed in one directory. Separate a csv file that consists of two columns: image file name and its class, must be located in the parent directory. The name of the directory with images must match the csv file name.
The function has to work in the following way:
* Download a csv file into a DataSet.
* Download all the images described in the csv file: [mahotas.imread()] (https://mahotas.readthedocs.io/en/latest/io.html).
* Display first 50 images.
* Form and return a DataSet that has to be an array of tuples [image, class code].


``` python 
import skillsnetwork

await skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0UEREN/hiroshima-lemon.zip", overwrite=True)


def get_data(folder, file):
    plt.rcParams["axes.grid"] = False
    data = []
    print(folder + '/' + file)
    ds = pd.read_csv(folder + '/' + file + '.csv') # Form a DataSet from a csv
    print(ds)
    i = 0
    r = 0
    for im, c in ds.values:
        if i==0 and r < 50:
            fig = plt.figure(figsize = (20,20)) # Display an image
        image = mh.imread(folder + '/' + file + '/' + im) #Download an image
        data.append([image, c]) # Append finaly DataSet
        if r < 50:
            plt.subplot(1, 5, i+1) # Create a table of images
            plt.imshow(image)
        i += 1
        if i==5:         
            i = 0
        r += 1
    plt.show()
   
    return np.array(data)   

```


``` python
d = "hiroshima-lemon"
f = "train_images"
train = get_data(d, f)
```


<img width="801" alt="image" src="https://github.com/ahammadmejbah/IBM-Automating-visual-inspection-with-Machine-Learning-ML/assets/56669333/a5e03fc2-77d4-44c0-a030-5b41bfc2b211">

### Comparing different classical classification methods

If we want to compare some classifiers, we should use a Pipeline. A Pipeline can be used to chain multiple estimators into one. This is useful as there is often a fixed sequence of steps in data processing, for example, feature selection, normalization and classification. A pipeline serves multiple purposes here:
You only have to call fit() once to evaluate a whole sequence of estimators. You can grid search over parameters of all estimators in the pipeline at once. Pipelines help to avoid leaking statistics from your test data into the trained model in cross-validation by ensuring that the same samples are used to train the transformers and predictors. All estimators in a pipeline, except the last one, should be transformers (i.e. should have a transform method). The last estimator may be of any type (transformer, classifier, etc.).
The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms and estimators.
The fit() and score() functions are used for training and evaluating the accuracy, this is an example of OOP polymorphism.
In order to test how it works we will use LogisticRegression.
We will use function plot_confusion_matrix() for the analysis.


``` python
clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000))])
clf.fit(features_train, c_train)
scores_train = clf.score(features_train, c_train)
scores_val = clf.score(features_val, c_val)
print('Training DataSet accuracy: {: .1%}'.format(scores_train), 'Test DataSet accuracy: {: .1%}'.format(scores_val))
plot_confusion_matrix(clf, features_val, c_val)  
plt.show() 

```


<img width="421" alt="image" src="https://github.com/ahammadmejbah/IBM-Automating-visual-inspection-with-Machine-Learning-ML/assets/56669333/a9e74ca8-32a8-4363-ba8f-fc212fdc650f">

As you can see, the results are not bad. Conflusion matrix shows us how many mistaken predictions we got. It allows us to check other classifiers and compare the results. We will test:

1. Logistic Regression
2. Nearest Neighbors
3. Linear SVM
4. RBF SVM
5. Gaussian Process
6. Decision Tree
7. Random Forest
8. Multi-layer Perceptron classifier
9. Ada Boost
10. Naive Bayes
11. Quadratic Discriminant Analysis


``` python

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    LogisticRegression(max_iter=1000),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
scores_train = []
scores_val = []
for name, clf in zip(names, classifiers):
    print("Fitting:", name)
    clf = Pipeline([('preproc', StandardScaler()), ('classifier', clf)])
    clf.fit(features_train, c_train)
    score_train = clf.score(features_train, c_train)
    score_val = clf.score(features_val, c_val)
    scores_train.append(score_train)
    scores_val.append(score_val)
```

<h4>Copyright © 2020 IBM Corporation. This notebook and its source code are released under the terms of the MIT License. </h4>
