import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_wave(data, ndx=0, sr=22050, label='waveform', color='blue', figsize=(6, 3), ax=None):
    
    # Extract the record's data
    if isinstance(data, list):
        w = data
    if isinstance(data, pd.DataFrame):
        w = data.loc[ndx, 'wave']

    # Create a time array
    time = np.arange(0, len(w)) / sr

    # Check if an axis object is provided
    if ax is None:
        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        own_plot = True
    else:
        own_plot = False

    # Plot the waveform
    ax.plot(time, w, label=label, color=color)
    ax.set_title(f'Wave {ndx}')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.grid()
    ax.legend()

    # If no external axis was passed, show the plot
    if own_plot:
        plt.tight_layout()
        plt.show()
            
    return ax 

def analyze_corr(data, th=0.3, figsize=(14, 10), annot=False):
    correlation_matrix = data.corr()

    cm = correlation_matrix.loc['age', :]
    print(cm[cm > th].index)

    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', center=0)
    plt.show()

def analyze_loadings(pca, features, releveant_features=5):
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    most_descriptive_features = {}
    for i in range(loadings.shape[1]):
        abs_loadings = np.abs(loadings[:, i])
        top_indices = np.argsort(abs_loadings)[-releveant_features:]
        most_descriptive_features[f'PC{i+1}'] = [features[top_indices[0]], features[top_indices[1]], features[top_indices[2]], features[top_indices[3]], features[top_indices[4]]]

    print("*** Most Descriptive Features for Each Principal Component***\n")
    for pc, desc_features in most_descriptive_features.items():
        print(f"{pc}:\t{desc_features}")