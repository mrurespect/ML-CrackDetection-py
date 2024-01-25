# Feature Extraction and Classification using FDA and PCA

This project aims to perform feature extraction on time and frequency domain signals and subsequently classify them into three categories: Tensile, Shear, and Mixed. The features extracted from the signals include both time and frequency domain characteristics.

## Dependencies
- NumPy
- SciPy
- Matplotlib
- Scikit-Learn
- Pandas

## Dataset
The training and testing datasets are loaded from MAT files (`x_train.mat` and `y_train.mat` for training, and `Testing_dataset_1650AE_1000_samples.mat` for testing) using SciPy's `loadmat` function.

## Feature Extraction
The `features_extraction` function is defined to extract various statistical features from both time and frequency domains. These features include minimum, maximum, mean, RMS (Root Mean Square), variance, standard deviation, power, peak value, peak-to-peak value, crest factor, skewness, kurtosis, and various statistical measures from the frequency domain.

## Data Processing
The extracted features are computed for both training and testing datasets. The results are stored in Pandas DataFrames for better visualization and organization.

## Standardization and Dimensionality Reduction
The training data is standardized using Scikit-Learn's `StandardScaler`. Principal Component Analysis (PCA) is then applied to reduce the dimensionality of the feature space while retaining important information.

## Linear Discriminant Analysis (LDA) Model
A Linear Discriminant Analysis (LDA) model is trained on the transformed feature space using the training data. The model is trained to classify signals into three classes: Tensile, Shear, and Mixed.

## Prediction and Visualization
The testing data is standardized and transformed using the same scaler and PCA model. The trained LDA model is then used to predict the classes of the testing data. The results are visualized by plotting the first two principal components against each other, with different colors indicating different classes. Additionally, predicted classes for the testing data are marked with 'x' symbols on the plot.
