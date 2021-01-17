# Excess-risk-bounds-in-robust-empirical-risk-minimization

Code of robust classification and robust regression. 
This code is an illustration of the methods described in the article "Excess risk bounds in robust empirical risk minimization".

A simplified but maybe more stable version of this algorithm has  been implemented as part of scikit-learn-extra python package (see [Here](https://scikit-learn-extra.readthedocs.io/en/latest/modules/robust.html)).

## Dependencies 
python >=3, numpy, scipy, joblib, scikit-learn and tqdm

## Usage
This package follows the basic scikit-learn API

    from estimators import classifier
    clf = classifier
    clf.fit(X, y)
    predictions = clf.predict(X)

    
The two main classes are `estimators.classifier` and `estimators.regressor`.

To obtain the figures in the article, please execute the script named after the figure, e.g. `figure_10.py` will compute figure 10. Please note that in these script, you may have to change the number of monte-carlo sample used (usually denoted by `M`) in order to either have a small computation time or a better approximation of the error.

## License
This package is released under the 3-Clause BSD license.
