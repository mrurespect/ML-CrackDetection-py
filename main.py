
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scipy.stats as stats
from scipy.fft import fft
from scipy.io import loadmat
# Load training data
train_data = sio.loadmat('DataSet\\x_train.mat')
X_train = train_data['XTrain']
ytrain_data = sio.loadmat('DataSet\\y_train.mat')
y_train = ytrain_data['X']

# Load testing data
test_data = sio.loadmat('DataSet\\Testing_dataset_1650AE_1000_samples.mat')
X_test = test_data['XTest']
df=pd.DataFrame(X_train)
dt=pd.DataFrame(X_test)
print(df)
x=df.values
te=dt.values
def features_extraction(x):
    ## TIME DOMAIN ##
    Min = np.min(x)
    Max = np.max(x)
    Mean = np.mean(x)
    Rms = np.sqrt(np.mean(x ** 2))
    Var = np.var(x)
    Std = np.std(x)
    Power = np.mean(x ** 2)
    Peak = np.max(np.abs(x))
    P2p = np.ptp(x)
    CrestFactor = np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2))
    Skew = stats.skew(x)
    Kurtosis = stats.kurtosis(x)
    ## FREQ DOMAIN ##
    ft = fft(x)
    S = np.abs(ft ** 2) / len(df)
    Max_f = np.max(S)
    Sum_f = np.sum(S)
    Mean_f = np.mean(S)
    Var_f = np.var(S)
    Peak_f = np.max(np.abs(S))
    Skew_f = stats.skew(x)
    Kurtosis_f = stats.kurtosis(x)
    return [Min, Max, Mean, Rms, Var, Std, Power, Peak, P2p, CrestFactor, Skew, Kurtosis, Max_f, Sum_f, Mean_f, Var_f,
            Peak_f, Skew_f, Kurtosis_f]



y = []
for i in range(len(df)):
    y.append(features_extraction(x[i]))

t = []
for i in range(len(dt)):
    t.append(features_extraction(te[i]))
X_test=t

FEATURES = ['MIN', 'MAX', 'MEAN', 'RMS', 'VAR', 'STD', 'POWER', 'PEAK', 'P2P', 'CREST FACTOR', 'SKEW', 'KURTOSIS',
            'MAX_f', 'SUM_f', 'MEAN_f', 'VAR_f', 'PEAK_f', 'SKEW_f', 'KURTOSIS_f']

features_MAt =pd.DataFrame(y,columns=[FEATURES])
print(features_MAt)
X_train=y
# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Perform PCA
pca = PCA(n_components=7)
X_train_pca = pca.fit_transform(X_train_scaled)

# Train FDA model
fda = LinearDiscriminantAnalysis()
fda.fit(X_train_pca, y_train.ravel())

# Standardize testing data and perform PCA
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Predict classes of testing data using the trained FDA model
y_pred = fda.predict(X_test_pca)

# Plot first two principal components against each other with different colors indicating different classes
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(['Tensile', 'Shear', 'Mixed']):
    plt.scatter(X_train_pca[y_train.ravel()==i+1, 0], X_train_pca[y_train.ravel()==i+1, 1], label=class_name)

plt.title('Factorial Plan')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Add predicted classes of testing data to the plot
for i, class_name in enumerate(['Tensile', 'Shear', 'Mixed']):
    plt.scatter(X_test_pca[y_pred==i+1, 0], X_test_pca[y_pred==i+1, 1], label=f'{class_name} (Predicted)', marker='x')

plt.show()