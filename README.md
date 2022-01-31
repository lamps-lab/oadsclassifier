# oadsclassifier

A language model based classifier to extract and classify URLs in scientific papers into two categories: URLs linking to open access datasets (OA D/S) or software and other URLs. The repository contains both training dataset, test dataset and the model. 

Scripts
---------------
1. language_model.py:
    A script for loading transformer language model. This module has been called within the script "classification_report.py"
    
2. pub_url_cleaner.py:
    A script for eliminating publisher urls from dataset. This module has been called within the script "classification_report.py"

3. url_extract.py:
    A script for extracting sentences incorporating urls from text file. 
    
4. classification_report.py:
    A script for implementing OADSClassifier.
    

How to use the script
---------------------
    python3 classification_report.py

Input
---------------------

"classification_report.py" script uses "dataset.csv" file of "oadsclassifier-Data" folder for both training and test data. This dataset contains 500 labeled samples. 400 samples are used for training purpose and the remaining 100 samples are used as test data.

Output
---------------------

The output of this script is the test accuracy, f1 score, precision and recall.


      
