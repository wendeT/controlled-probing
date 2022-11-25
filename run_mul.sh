#!/usr/bin/env bash
python3 prob_exper_single.py 'data/de_B-EVENT.csv' &&
python3 prob_exper_single.py 'data/de_B-FAC.csv' &&
python3 prob_exper_single.py 'data/de_B-GPE.csv' &&
python3 prob_exper_single.py 'data/de_B-LANGUAGE.csv' &&
python3 prob_exper_single.py 'data/de_B-LOC.csv' &&
python3 prob_exper_single.py 'data/de_B-NORP.csv' &&
python3 prob_exper_single.py 'data/de_B-ORG.csv' &&
python3 prob_exper_single.py 'data/de_B-PERSON.csv' &&
python3 prob_exper_single.py 'data/de_B-PRODUCT.csv' &&
python3 prob_exper_single.py 'data/de_B-WORK_OF_ART.csv'


python3 prob_exper_single.py 'data/tf_df_en_ambig_shared_deviate.csv'

python3 prob_exper_single.py 'data/4_df_ent_org_per_loc.csv'


python3 prob_exper_single.py 'data/batch_22_shared_org_per_loc.csv'

python3 prob_exper_single.py 'data/sampled_batch_batch_22_shared_org_per_loc.csv'
