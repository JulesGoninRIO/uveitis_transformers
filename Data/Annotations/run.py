import os
import Get, Split

if os.path.exists('T:/Studies'):
    cohort_folder = "T:/Studies/Uveitis/data/Cohorts/Vasculitis_ECRFs_231030/Vasculitis Grading Collage"
    cohort_folder_2 = "T:/Studies/Uveitis/data\Cohorts/Uveitis/Vasculitis Grading Collage"
else:
    cohort_folder = "/data/soin/cohortbuilder/cohorts/ch_amiotv/ecrf/Vasculitis Grading 2 Collage"

# Download ecrf with Cohort Builder 
# ssh cohortbuilder@sfhvcohortbuilder01
# password: **************
# conda activate cb
# cb-dev download --configs Uveitis-eCRF --cohorts_dir /mnt/vfhvnas01/HOJG/Studies/Uveitis/data/Cohorts/ -i fhv_jugo -p Uveitis -w 'Vasculitis Grading Collage'

# Update annotations.csv
# Get.get(cohort_folder)
if os.path.exists('T:/Studies'):
    Get.get(cohort_folder_2)
    Get.get(cohort_folder_2, "Muriel")
else:
    Get.get(cohort_folder, "Teodora")
    Get.get(cohort_folder, "Shalini")

import pandas as pd
import numpy as np
todo = pd.read_csv(os.path.dirname(__file__)+'/intergrader/Shalini_list.csv')
pid_todo = np.unique([int(pid) for pid in todo['PID'] if not '.' in pid])
ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
Split.add('Shalini', ratios, pid_test=pid_todo)