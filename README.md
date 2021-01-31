# Mara Tatari - MSc project - Birckbek University of London 

## Hypothesis/Aim

Prolines are found in cis conformation in a higher proportion compared to other amino acids due to their side chain which is energetically equivalent between cis and trans conformations. This can be determined on the basis of the omega torsion angle which in the case of cis conformations has values between 30o and -30o and in trans conformations between 180o and -180o. 

## Methodology

Using the omega torsion angles from proteins from PDB use suitable ML methods to predict the conformation of Prolines in various proteins. Using the Proline as the central aa in windows of different sizes we will aim to predict its conformation on the basis of its neighbouring aas. A seond method will be to replace the aas with their aa-group (e.g. aromatic, acidic, aliphatic etc) and perform the prediction based on the aa-type. 

## Data to use

- Immunoglobulins are a class of proteins particularly rich in cis-Prolines and as such are a suitable training dataset for the ML training and testing 
ML approaches: SVM, KNN and random forests
- 2 ways of encoding the protein sequence: one-hot encoding, BLOSUM-encoding (based on protein sequence conservation score)
- 2 ways of representing the data: using the actual aas or their aa-groups (aromatic, basic, acidic etc)

### Files included and description

| File | Description |
|-------------|-------|
|ohe_aas_win5 | One-hot-encoding of a aa-window size 5 (middle-PRO) and SVM for cis/trans prediction
|ohe_aas_win11 | One-hot-encoding of a aa-window size 11 (middle-PRO) and SVM for cis/trans prediction
|ohe_aaTYPE_win5 | One-hot-encoding of a aa-window size 5 (middle-PRO) using the aa-TYPES and SVM for cis/trans prediction
|ohe_aaTYPE_win11 | One-hot-encoding of a aa-window size 11 (middle-PRO) using the aa-TYPES and SVM for cis/trans prediction
|blosum62_win5 | Blosum62-encoding of a aa-window size 5 (middle-PRO) and SVM for cis/trans prediction
|blosum62_win11 | Blosum62-encoding of a aa-window size 11 (middle-PRO) and SVM for cis/trans prediction
|knn_blosum62_win11 | Blosum62-encoding of a aa-window size 11 (middle-PRO) and KNN for cis/trans prediction
|knn_ohe_win11 | One-hot-encoding of a aa-window size 11 (middle-PRO) and KNN for cis/trans prediction
|pdbtors.sh | bash script for the extraction of torsions using the torsion function (external function, not included here)
|BLOSUM62 | matrix used for the Blosum-encoding of the proteins
|IgG_IDs | the IDs of the IgG proteins that were used for testing and analysis

### Conditions to keep in mind

When looking at the omega torsions for the Prolines values of 9999.000 are found. For the purposes of this analysis these values have been eliminated 

In proteins non-standard amino acids are often found. Initially I am excluding them from the analysis for 2 reasons: their single-letter symbols are interfering with the standard aas and also I am not sure how to include them in the Blosum matrix 

In the data there are some None values which are excluded from the analysis at the moment because I don’t know what they are (beginning or end of protein?)
