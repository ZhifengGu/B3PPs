# Data
The blood-brain barrier penetrating peptide and non blood-brain barrier penetrating peptide data are stored in ./Benchmark dataset/Raw data/Training.txt(2851 sequences) and ./Benchmark dataset/Raw data/Validing.txt(108 sequences).

# Working environment
The work was built in Microsoft Visual Studio Code 1.85.0
encoding : utf-8  
environment : Python 3.9.16
The pandas, matplotlib, joblib, numpy, scikit-learn packages are used for model training.
The packages requirement is stored in : requirements.txt


# How to use
B3P2Augur.py is the tool to predict blood-brain barrier penetrating peptides. 

Environment configuration: 
pip3 install lightgbm
pip3 install xgboost
pip install -r requirements.txt

The user needs to confirm that the input peptide sequence is in FASTA format.

# View results
Finally, you can view the prediction results in the result box.
