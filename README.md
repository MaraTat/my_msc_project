# Mara Tatari - MSc project - Birkbeck University of London - Bioinformatics with Systems Biology

## Hypothesis/Aim

Prolines are found in cis conformation in a higher proportion compared to other amino acids due to their side chain which is energetically equivalent between cis and trans conformations. This can be determined on the basis of the omega torsion angle which in the case of cis conformations has values between 30o and -30o and in trans conformations between 180o and -180o. 

## Methodology

Using the omega torsion angles from proteins from PDB use suitable ML methods to predict the conformation of Prolines in various proteins. Using the Proline as the central aa in windows of different sizes we will aim to predict its conformation on the basis of its neighbouring aas. Other than using OHE 2 additional matrices will be used in order to try and incorporate amino acid biochemical properties (BLOSUM62 and LiKo)

## Data to use

- PISCES offers preculled datasets and for this work proteins with resolution better than 3A will be used. The protein IDs can be found in the repository. 
- 3 ways of encoding the protein sequence: one-hot encoding, BLOSUM-encoding (based on protein sequence conservation score) and LiKo (a reduced version of the BLOSUM62 matrix)
- 2 decision tree ensemble ML methods will be compared: Random Forest Classifier and XGBoost (a boosted tree ensemble method)


### Conditions to keep in mind

When looking at the omega torsions for the Prolines values of 9999.000 are found. For the purposes of this analysis these values have been eliminated 

In proteins non-standard amino acids are often found. Where possible these will be replaced with the parental amino acid. Chemical modifications will be removed 

In the data there are some None values which are excluded from the analysis at the moment because I don’t know what they are (beginning or end of protein?)
