#B3P2Augur
B3P2Augur is a blood-brain barrier penetrating peptide(B3PPs) classification prediction tool that can determine whether an input amino acid sequence belongs to B3PPs.

# Data
The original B3PPs sequences and non-B3PPs data are stored in ./B3PPs/B3P2Augur Data/Raw data/Training.txt(2851 sequences) and ./B3PPs/B3P2Augur Data/Raw data/Validing.txt(108 sequences).

The extracted features data is stored in ./B3PPs/B3P2Augur Data/Extracted features.

The results of multiple ML methods are stored in ./B3PPs/B3P2Augur Data/Comparison of multiple ML methods.

The data of data augmentation with different ratios is stored in ./B3PPs/B3P2Augur Data/Data Augmentation with different ratios.

The training set and independent test set data used for modeling are stored in ./B3PPs/B3P2Augur Data/Training.csv and ./B3PPs/B3P2Augur Data/Independent test set.csv.

# Working environment
The work was built in Microsoft Visual Studio Code 1.85.0
encoding : utf-8  
environment : Python 3.9.16
The pandas, matplotlib, joblib, numpy, scikit-learn packages are used for model training.
The packages requirement is stored in : requirements.txt


# How to use
It is necessary to install the dependent packages before running B3P2Augur.py, which is stored in ./B3PPs/B3P2Augur

Environment configuration: 
pip3 install lightgbm
pip3 install xgboost
pip install -r requirements.txt

The user needs to confirm that the input peptide sequence is in FASTA format.

Details are available in B3P2Augur_Manual.pdf that stored in ./B3PPs/B3P2Augur

# View results
Finally, you can view the prediction results in the result box.