# Disaster Response Framework and Web App


<a name="dependencies"></a>
## Dependencies
* Python 3.7+ 
* Data handle and machine learning libraries: NumPy, SciPy, pandas, Scikit-Learn, pickle, joblib
* Natural Language Process libraries: NLTK, re
* Database libraries: SQLalchemy
* (optional) Libraries for deep learning classification: transformers, torch
* Visualization libraries: Flask, Plotly

<a name="exec"></a>
## Executing Program:
1. To train/evaluate the SVC machine learning model for multiclassification, run the following command in this directory **DisasterResponseApp**:
       
	  ` python model/train-multiclassifier.py`

2. To execute the web app in your localhost, run the following command from the directory **app** :

    `python run.py`

3. To visualise and use your application to http://localhost:5000/. Please note you will need to train the deep learning models and locate them in the directory **models** to use the deep classifier.

<a name="authors"></a>
## Authors

* [VPL](https://github.com/vponcelo): New models, functionalities and updated interface
	* [LN](https://github.com/lng15): First version

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

<a name="acknowledgement"></a>
## Acknowledgements

* [Belmont Forum](https://www.belmontforum.org/archives/projects/re-energize-governance-of-disaster-risk-reduction-and-resilience-for-sustainable-development) first disaster-focused funding Call Belmont Collaborative Research Action 2019: Disaster Risk, Reduction and Resilience (DR32019).

<a name="citation"></a>
## Citation

For any use of this code, please cite our paper as:

<cite>Ponce-López, Víctor and Spataru, Catalina. Social Media Data Analysis Framework for Disaster Response. Research Square preprint, Under Review at Discover Artificial Intelligence, Springer Nature, 2022. [DOI: 10.21203/rs.3.rs-1370942/v1](https://doi.org/10.21203/rs.3.rs-1370942/v1) </cite>


