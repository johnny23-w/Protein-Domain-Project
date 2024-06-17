import gzip
import csv
import pandas as pd

def sort_domains(domain_list):
    # For a list of ('PFAM000X', (start, end)), sort by start index
    idx = sorted(range(len(domain_list)), key=lambda k: domain_list[k][1][0])
    return [domain_list[i] for i in idx]


# The following code writes to a csv where index is UniProt ID and single column containing lists of Pfam domain annotations and start,end boundaries for each protein
with gzip.open('protein2ipr.dat.gz', 'rt', encoding='utf-8') as f:
    
    iprreader = csv.reader(f, delimiter='\t', quotechar='|')

    df = pd.DataFrame(columns=['pfam_domains'])
    df.to_csv('pfam_domains.csv')

    # Count number of proteins in dataframe before writing to csv and starting new dataframe
    counter = 0
    prev_protein = ''
    for idx, line in enumerate(iprreader):
        cur_protein = line[0]
        # Check if current line corresponds to a new protein sequence
        if (cur_protein != prev_protein):
            counter += 1
            # If more than 5000 proteins in dataframe, write to csv and start new dataframe
            if counter > 5000:
                # Sort domains by start index
                df['pfam_domains'] = df['pfam_domains'].apply(sort_domains)
                df.to_csv('pfam_domains.csv', mode='a', header=False)
                df = pd.DataFrame(columns=['pfam_domains'])
                counter = 1
            # New row for new protein
            df.loc[line[0]] = [[]]
        if line[3][:2] == 'PF':
            # Add domain annotation (if current line corresponds to Pfam domain) in form (PF0000, (start, end))
            df.loc[line[0], 'pfam_domains'].append((line[3], (int(line[4]), int(line[5]))))
        if idx%1000000 == 0:
            clear_output()
            print('Processing line: ', idx)
        prev_protein = cur_protein
    
    df['pfam_domains'] = df['pfam_domains'].apply(sort_domains)
    df.to_csv('pfam_domains.csv', mode='a', header=False)
