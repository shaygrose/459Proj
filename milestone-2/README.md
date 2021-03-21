## CMPT 459 Milestone 2

### Shayna Grose, Patrick Nguyen, Argenis Chang


First, install the requirements listed in requirements.txt by doing `pip install -r requirements.txt` from the main directory.  
Our code can be run using `python src/main.py`. It cannot be run directly using `python main.py` since it will break the file access paths used.  
If the 3 model files in the models folder exist, the code does not retrain the models, it simply loads the .pkl files and evaluates them. If you wish to retrain the models and produce new .pkl files you have to first delete the old ones.  
Finally, the results are printed to the terminal, and the confusion matrices for KNN and Random Forest are outputted to the plots folder.  
XGBoost did not support a graphical confusion matrix, so the values of its confusion matrix are simply printed in the terminal at the end of all other results.
