## Description

This repository provides sample code; running steps 1-8 sequentially will help you understand the algorithm logic in the paper "Spectro-Temporal Deep Learning Integrated with Real-Time Near-Infrared Spectroscopy for Dynamic Tracking of Bioactive Constituents in Chrysanthemum Extraction."

## Installation

Spectro-Temporal-Deep-Learning requires following dependencies to be installed:

- numpy == 2.1.3
- matplotlib == 3.10.1
- matplotlib-inline == 0.1.6
- scikit-learn == 1.6.1
- scipy == 1.15.2
- joblib == 1.4.2
- torch == 2.7.0
- torchvision == 0.22.0
- shap == 0.47.2

If you intend to introduce new functionalities or make code modifications, we suggest duplicating the repository and setting up a local installation:

```bash
[git clone https://github.com/CongCong-Bang/Spectro-Temporal-Deep-Learning-for-near-infrared-spectroscopy](https://github.com/CongCong-Bang/Spectro-Temporal-Deep-Learning-for-near-infrared-spectroscopy)
cd Spectro-Temporal-Deep-Learning-for-near-infrared-spectroscopy

# [OPTIONAL] create conda environment
conda create -n STDL python=3.10
conda activate STDL

# install package
pip install -e .
```

## Data

Refer to the step0_raw_data folder. During extraction, two types of data are collected: spectral and chromatographic. The spectral data include transmission intensity (spec_data.npy), wavelength information (band.npy), and spectral acquistion time points (time_stamp_1.npy). The chromatographic data include Apigenin-7-O-glucuronide concentration (content_data.npy) and chromatographic acquistion time points (time_stamp_2.npy).

## Step-by-step tutorial

This is a foolproof, step-by-step tutorial.

(1) Refer to the step1_intra_batch_and_inter_batch_analysis. run plot.py;

(2) Refer to the step2_training_crossval_test_split. Enter the subfolder raw_spec and run dataset_split.py. Then enter the subfolder time_resolved_spec and run B-splined_or_time_windowing.py and training_crossval_split.py in sequence. After these runs are completed, the data sources required for the four data configurations — raw spectra, B-spline-augmented spectra, time-resolved spectra, and time-resolved spectra with time-stamp incorporation — have been generated, and the training, cross-validation, and test splits have been completed;

(3) Refer to the step3_PLS_model. Run pls.py and pls4bspline.py in sequence;

(4) Refer to the step4_1D_deeplearning. Enter the subfolder data_preparation; if raw_spectra are selected as input, run dataset_split_raw.py; if B-spline-augmented spectra are selected as input, run dataset_split_bspline.py. Then enter the subfolder CNN/GRU/LSTM and run CNN.py, GRU.py, and LSTM.py, respectively;

(5) Refer to the step5_2D_deeplearning. Enter the subfolder data_preparation; if raw spectra are selected as input, run dataset_split_raw.py; if B-spline_augmented spectra or time-resolved spectra are selected as input, run dataset_split_time_window.py. The window length can be adjusted by setting the t_idx variable in the .py file. Then enter the subfolder CNN_CNN/CNN_GRU/CNN_LSTM and run CNN_CNN.py, CNN_GRU.py, and CNN_LSTM.py, respectively;

(6) Refer to the step6_2D_deeplearning_with_timestamp. Enter the subfolder data_preparation; if raw_spectra are selected as input, run dataset_split_raw.py; if time-resolved spectra with time-stamp incorporation are selected as input, run dataset_split_time_window.py. The window length can be adjusted by setting the t_idx variable in the .py file. Then enter the subfolder CNN_CNN_timestamp/CNN_GRU_timestamp/CNN_LSTM_timestamp and run CNN_CNN.py, CNN_GRU.py, and CNN_LSTM.py, respectively;

(7) Refer to the step7_2D_deeplearning_with_timestamp&spectral_downsampling. Enter the subfolder data_preparation, run dataset_split_time_window.py. The t_idx was set to 4, with the window length set to 9. Then enter the subfolder CNN_CNN_timestamp&spectral_downsampling and run CNN_CNN_.25.py, CNN_CNN_.50.py, and CNN_CNN_.100.py in sequence to evaluate the model performance at spectral down-sampling fractions of 0.25, 0.5, and 1, respectively;

(8) Refer to the step8_DeepSHAP. Enter CNN_CNN_timestamp and run CNN_CNN.py and data_for_temporal_analysis.py in sequence. Then enter DeepSHAP and run deepshap.py to visualize the most influential features spanning both the spectral and temporal domains.

## Contributing

As this is the first version of our package, there might be scope for enhancements and bug fixes. We highly value community contributions. 
If you find any issues, missing features, or unusual behavior during your usage of this library, please feel free to open an issue or submit a pull request on GitHub. 
For any queries, suggestions, or feedback, please send them to [ru2689385126@163.com].

## License

Distributed under the terms of the \[MIT\] license, "Spectro-Temporal-Deep-Learning" is free and open source software.

## Citation
These algorithms come from the unpublished paper "Spectro-Temporal Deep Learning Integrated with Real-Time Near-Infrared Spectroscopy for Dynamic
Tracking of Bioactive Constituents in Chrysanthemum Extraction."
