 # Baylor College of Medicine - Russell Ray Lab Experimental Data Mining and Analysis
s
 ## About

The Russell Ray Lab is conducting experiments on mouse pups to investigate Sudden Infant Death Syndrome (SIDS) and collecting ECG and breath flow velocity data from the mice in order to garner information and insight into the mechanisms and/or underlying pathophysiologies that are responsible for SIDS in human infants.

So far in this repository we have developed a data mining pipeline to identify or uncover classes of breaths that the are associated with particular characteristics, which may be informative and ultimately useful for the lab's mission. As we are still in the process of investigating for said classes, we have not identified them yet. Upon identification of such classes, we can then construct a machine learning algorithm that can diagnose the mice's suscpetibilities to SIDS-like phenomena which can, if successful, be very informative when we analyze the classified breaths from a neurological perspective and then consider the parallel of those neurological features in humans. More information on the methodologies and background research underwent throughout the development of this repository [is laid out in the follow summary report.](https://drive.google.com/file/d/1JE6vpFgYof3KJZhZym1y52igWQB82wtk/view?usp=sharing)

Upon uploading the raw lab data to a **data** folder in the root directory, the user should then run the *preprocessing.py* file which reproduces the raw data in a Python-friendly format. The user can then proceed to use the jupyter notebooks or to-be develped full data mining pipeline.

## Repository Details

The following is a description of the files and folders in this repository.

* **data_mining_analysis** folder: Contains the following files
    - *breath_extraction_pca_plots.ipynb*: Includes the full respiratory data mining pipeline
        * Breath data preprocessing
        * Principal Component Analysis (PCA)-based Plotting and Clustering
        * Cluster Anlysis
    - *frequency_analysis.ipynb*: this file includes some code used to analyze frequency content of breathing data using fourier transforms
    - *frequency_analysis.ipynb*: this file includes some code used to analyze frequency content of breathing data using short-term fourier transform (STFT)-based spectrograms
    - *report_visuals.ipynb*: this file includes some code used to produce some visuals used in the summary report

* **under_development** folder: Contains the following files
    - *data_mining.py*: putting the entire data mining code into one file that can be run from terminal including inputting parameters (underdevelopment)
    - *gradient_based_breatlists_extraction.py*: all of the breath extraction code to be used by data_mining.py
    - *preprocessing_germless.py*: code used to preprocess raw germless mice data straight from the lab to a useable format before further preprocessing
    - *preprocessing.py*: code used to preprocess raw hM3D and hM4D mouse data straight from the lab to a useable format before further preprocessing
