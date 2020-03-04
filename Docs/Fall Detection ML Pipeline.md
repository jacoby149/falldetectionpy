# Fall Detection ML Pipeline

Our fall classification pipeline is comprised of the following components.



**Data Labeling**

1. Data labeled in Excel by manual inspection of MATPLOTLIB graphs.



**Features**

1. Segmented XYZ acceleration data from accelerometer.

   * These features can possibly then be PCA compressed for better results.

2. Frequency domain representations of XYZ acceleration data.

   * STFT, and Wavelet transform.

   * These features can also be PCA compressed for better results.

These features have tunable parameters, and the PCA compression is also tunable. How many dimensions are we reducing?



**Algorithm**

Features are then fed into an SVM, and the performance of this SVM will be benchmarked using a  test and validation dataset. After doing multiple benchmarks on different parameter values of the features,  we will select the optimal parameter values for our features.



