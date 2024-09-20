import json
import pandas as pd
import os
from tqdm import tqdm

TT_labels = json.load(open(os.path.dirname(__file__)+'/../TT_labels.json'))
changes = json.load(open(os.path.dirname(__file__)+'/changes.json'))
items = TT_labels['table columns']
TT_answers = TT_labels['T-T score']

# ===================================================================
# Load ecrf and create annotation table
def get(cohort_folder, annotator="Florence"):

    # Initialisation
    eye_ecrf = {'OD Vasculitis V2': 'R', 'OS Vasculitis V2': 'L'}
    items_all = ['key', 'PID', 'sex', 'birthyear', 'eye', 'date', 'url']+items
    csv_path = f'intergrader/table_{annotator}.csv' if annotator!="Florence" else 'table.csv'
    csv_path = os.path.dirname(__file__)+'/'+csv_path
    annotators = [annotator, 'Victor'] if annotator=='Shalini' else [annotator]

    # Load current annotation table into d
    d = {} 
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        for item in df.columns:
            d[item] = list(df[item])
    else:
        for item in items_all:
            d[item] = []
    n_0 = len(d['key'])

    # Iterate over ecrf to fill d
    patients = [thing for thing in os.listdir(cohort_folder) if os.path.isdir(cohort_folder+'/'+thing)]
    for patient in tqdm(patients):
        for date_id in os.listdir(cohort_folder+'/'+patient):
            for eye_side in os.listdir(cohort_folder+'/'+patient+'/'+date_id):

                # Load ecrf and metadata
                folder = cohort_folder+'/'+patient+'/'+date_id+'/'+eye_side
                info_file = open(folder+'/info.json')
                ecrf_file = open(folder+'/ecrf/ecrf_01.json')
                info = json.load(info_file)
                ecrf = json.load(ecrf_file)

                # Check if ecrf is valid and filled by annotator
                if ecrf['title'] in eye_ecrf and info['owner']['firstName'] in annotators:

                    # Get eye key and check if it is already done
                    pid = info['patient']['surname']
                    date = info['study']['studyDatetime'][:10]
                    side = eye_ecrf[ecrf['title']]
                    key = pid+'_'+date+'_'+side
                    if not any([key_ in d['key'] for key_ in [key, key+'_Ex']]):

                        # Fill d with corrected ecrf answers
                        fill_d(d, ecrf['questions'])

                        # Fill d with metadata
                        d['key'][-1] = key+d['key'][-1]*'_Ex'
                        d['eye'].append(side)
                        d['PID'].append(pid)
                        d['sex'].append(info['patient']['sex'])
                        d['birthyear'].append(int(info['patient']['birthdate'][:4]))
                        d['date'].append(date)
                        d['url'].append(info['url'])

                        # Complete existing columns (splits) with 'new'
                        for item in d:
                            if item not in items_all:
                                d[item].append('new')

                # Close ecrf and metadata
                info_file.close()
                ecrf_file.close()

    # Add custom items
    add_custom_items(d)

    # Patch changes
    if annotator == 'Florence':
        for key, change in changes.items():
            if key in d['key']: # may not be True if key already patched
                i = d['key'].index(key)
                if change == 'excluded':
                    d['key'][i] = d['key'][i]+'_Ex'
                else:
                    for item, new_label in change.items():
                        d[item][i] = new_label

    # Save to csv
    pd.DataFrame(d).to_csv(csv_path)
    print(f"table updated: {n_0} -> {len(d['key'])} entries")

# ===================================================================
# Fill d with corrected ecrf answers           
def fill_d(d, content):

    # Changes to do
    caracts = {'é': 'e', 'ê': 'e', 'è': 'e', 'à': 'a'}
    to_replace = {'poor': 'bad', 'leakage at': 'margin blurring', 'faint hyperfluo': 'faint hyperfluorescence',
                'not': 'not assessable', 'complete ring': 'complete ring of leakage', 'na': 'not assessable',
                'more extended': 'multifocal', '': 'not assessable'}
    to_replace_k = {'diffuse': (6, '(capillar)'), 'extensive': (12, '(staining)')}
    word_limit = {0: 1, 2: 2, 4: 2, 8: 2, 11: 1, 12: 1}

    # Get the corrected answer k
    def get(k):
        if k==13 and not 'value' in content[k]['answer']:
            return ''
        answer = content[k]['answer']['value']
        if type(answer) == type(''):
            answer = answer.lower()
            while answer.startswith(' '):
                answer = answer[1:]
            while answer.endswith(' '):
                answer = answer[:-1]
            if k in word_limit:
                answer = ' '.join(answer.split(' ')[:word_limit[k]])
            if '(' in answer:
                answer =  answer.split('(')[1].split(')')[0]
            if answer in to_replace:
                answer = to_replace[answer]
            if answer in to_replace_k and k==to_replace_k[answer][0]:
                answer += ' '+to_replace_k[answer][1]
        return answer
    
    factors = get(1)
    for i, factor in enumerate(factors):
        factor = factor.lower()
        if factor.startswith('vitritis'):
            factors[i] = 'vitritis'

    comment = get(13)
    for caract in caracts:
        comment = comment.replace(caract,caracts[caract])

    answers = [factors]+[get(k) for k in [0]+[i for i in range(2, 13)]]+[comment]
    
    # Compute Tugal-Tuktun score
    TT_score = 0
    for i,answer in enumerate(answers):
        if type(answer)==type(0):
            answer = str(answer)
        if type(answer)==type(''):
            for k, TT_answer in enumerate(TT_answers):
                TT_score += (k+1)*(answer in TT_answer)
        d[items[i]].append(answer)
    d['T-T score'].append(TT_score)

    # Exclude eye if needed
    d['key'].append(('exclure' in comment) or (d['image quality']=='bad'))
    
# ===================================================================
# Add custom items to d
def add_custom_items(d):
    d['vascular leakage (significant)'] = []
    significant = {'not assessable':'not assessable', 'none':'not significant', 
                   'focal':'not significant', 'multifocal':'significant', 'diffuse':'significant'}
    d['vascular leakage (significant)'] = [significant[leakage] for leakage in d['vascular leakage']]
    to_merge = ['incomplete ring of leakage', 'complete ring of leakage']
    d['macular edema (ring)'] = ['ring of leakage' if macula in to_merge else macula 
                                 for macula in d['macular edema']]
    grouped_macula = {"incomplete ring of leakage":"faint leakage", 
                    "faint hyperfluorescence":"faint leakage",
                    "complete ring of leakage":"complete leakage", 
                    "pooling of dye in cystic spaces":"pooling of dye"}
    d['macular edema (grouped)'] = [grouped_macula[macula] if macula in grouped_macula else macula
                                    for macula in d['macular edema']
                                    ]
    d['vascular leakage (periphery 2)'] = ['2' if str(leakage) in ['1', '2', '3'] else leakage 
                                           for leakage in d['vascular leakage (periphery)']]
    scores = {'1': '0', '2': '4', '3': '4', '5': '4', '6': '8', '7': '8'}
    d['capillary leakage (score 4)'] = [scores[str(leakage)] if str(leakage) in scores else leakage 
                                        for leakage in d['capillary leakage (score)']]
    nothing = ['not assessable', 'none', '0']
    for item in ['optic disk hyperfluorescence', 'macular edema', 'vascular leakage',
                 'capillary leakage (score)', 'vascular leakage (periphery)', 'capillary leakage']:
        d[item+' (any)'] = ['none' if answer in nothing else 'any' for answer in d[item]]