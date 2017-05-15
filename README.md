# merck
Merck Molecular Activity Challenge code

re-implementation of the paper (the recommended model): 

`Ma, J., Sheridan, R.P., Liaw, A., Dahl, G.E. and Svetnik, V., 2015. Deep neural nets as a method for quantitative structure–activity relationships. Journal of chemical information and modeling, 55(2), pp.263-274.`

## Installation
The code was tested in _Keras_ with _Tensorflow_ backend. 
The packages needed are listed in the `requirements.txt`

### Installing python virtual environment and requirements
 ```
 pip install virtualenv
 virtualenv --no-site-packages vkeras
 source vkeras/bin/activate
 pip install -r path/to/requirements.txt

 ```
 

## Running the Code
* Download the training and test data-set from: [Paper supplementary materials](http://pubs.acs.org/doi/suppl/10.1021/ci500747n/suppl_file/ci500747n_si_002.zip)

* Set `data_root` and `save_root` variables in `data_preprocessing.py` and run it (This will remove the features that are not common to both training and test sets and, rescale features and activations).

    * Currently the features are rescaled to 0-1 by dividing each column by its max and the activations are rescaled to their z-score  

* point the `data_root` in `main.py` to where the pre-processed training and test files are located.

* `python main.py`


## Results

The Standered Error of Prediction (SEP) on the test set

| Dataset  | [merk paper](http://www.cs.toronto.edu/~gdahl/papers/deepQSARJChemInfModel2015.pdf) | This implementation |
|----------|------------|---------------------|
| 3A4      | 0.48       | 0.50                |
| CB1      | 1.25       | 1.21                |
| DPP4     | 1.30       | 1.68                |
| HIVINT   | 0.44       | 0.47                |
| HIVPROT  | 1.66       | 1.60                |
| LOGD     | 0.51       | 0.51                |
| METAB    | 21.78      | 23.19               |
| NK1      | 0.76       | 0.76                |
| OX1      | 0.73       | 0.81                |
| OX2      | 0.95       | 0.93                |
| PGP      | 0.36       | 0.38                |
| PPB      | 0.56       | 0.57                |
| RAT_F    | 0.54       | 0.55                |
| TDI      | 0.40       | 0.41                |
| THROMBIN | 2.04       | 2.10                |

