# oadsclassifier

A language model based classifier to extract and classify URLs in scientific papers into two categories: URLs linking to open access datasets (OA D/S) or software and other URLs. The repository contains both training dataset, test dataset and the model. 

Scripts and sav file
---------------
1. language_model.py:
    A script for loading transformer language model. This module has been called within the script "classification_report.py"
    
2. pub_url_cleaner.py:
    A script for eliminating publisher urls from dataset. This module has been called within the script "classification_report.py"

3. model_weight.sav
   This file contains model weights which is loaded in the "OADS.py" script.
    
4. OADS.py:
    A script for implementing OADSClassifier.
    
Requirements
---------------
    pip install --upgrade transformers==4.2

How to use the script
---------------------
    python3 classification_report.py

Input
---------------------

"OADS.py" script uses takes a text file as input.

Output
---------------------

The output of this script is the list of sentences that contains URLs and their predicted class. The classes are - "OADS", "Not-OADS"
1. "OADS" means Open access dataset/software
2. "Not-OADS" means not open access dataset/software


      
