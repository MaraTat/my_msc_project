import pandas as pd

def cleanupmydata(protdf):
    """
    Parameters
    ----------
    protdf : TYPE .txt file which has been generated through another function pdbtorsions outside of this function
        DESCRIPTION. used to eliminate unnecessary information such as modified amino acids 
        or amino acids at the ends of proteins where the torsion angles are not useful for downstream analysi

    Returns
    -------
    Cleaned up dataframe with the amino acid residues encoded as single letters and the omega torsion angles

    """
    try:
        if not isinstance(protdf, pd.DataFrame):
            with open (protdf, 'rt') as file:
                dftest = pd.read_csv(file, skiprows=(0,1), header=None, delimiter='\s+')
        else:
            dftest = protdf
            
        dftest.columns = ['Resnum', 'Resnam', 'PHI', 'PSI', 'OMEGA'] # eliminating unecessary information --> Resnum, PHI, PSI from the df
        prot_df = dftest[['Resnam', 'OMEGA']] # eliminating unecessary information --> Resnum, PHI, PSI from the df
        prot_df = prot_df.loc[(prot_df['OMEGA'] >= -360) & (prot_df['OMEGA'] <= 360)] # getting rid of weird values like 9999.000
        
        aas =['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
              'PHE','PRO','SER','THR','TRP','TYR','VAL','5HP','ABA','AIB','BAL','BMT','CEA','CGU','CME','CSD','CSO',
                      'CSS','CSW','CSX','CXM','DAL','DAR','DCY','DGL','DGN','DHI','DIL',
                      'DIV','DLE','DLY','DPN','DPR','DSG','DSN','DSP','DTH','DTR','DTY',
                      'DVA','FME','HYP','KCX','LLP','MLE','MVA','NLE','OCS','ORN',
                      'PCA','PTR','SAR','SEP','STY','TPO','TPQ','TYS','MSE']
    
        aaletters = ['A','R','N','D','C','Q','E','G','H','I',
                  'L','K','M','F','P','S','T','W','Y','V',
                  'E','A','A','A','T','C','E','C','C','C','C','C','C','M','A','R','C',
                            'H','I','V','L','K','F','P','N','S','D','W','Y','T','V','M','P','K',
                            'K','L','C','A','E','Y','G','S','Y','T','F','Y','T','A','A','Y','M']
    
        
        prot_df = prot_df[prot_df['Resnam'].isin(aas)] # getting rid of residues that are not aas but just active groups
    
        prot_df.replace(to_replace=aas, value= aaletters, inplace=True) # replacing aas with their single letters
    
        prot_df = prot_df.reset_index(drop=True)
        return prot_df
    
    except:
        return print('Please use a dataframe with your data or run the function pdbtorsions on the raw .ent pdb file and generate a .txt file')
      
