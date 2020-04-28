import pandas as pd

def extract_smiles(path='ChEMBL25/chembl25_smiles'):
    file = 'data/' + path + '.csv'
    print("file:", file)
    df = pd.read_csv(file, delimiter=';')
    data = df['Smiles']
    data.to_csv(path='data/TFL/chembl25_smiles.csv', header=False, index=False)
    return

if __name__ == '__main__':
    extract_smiles()
