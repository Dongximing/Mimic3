import csv
import pandas as pd
import numpy as np
import os
import sys
# note_events
import argparse
from mimic3benchmark.util import dataframe_from_csv

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('--mimic3_path', type=str,default="/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4")
parser.add_argument('--output_path', type=str,default="/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4")
args = parser.parse_args()


def read_icd_diagnoses_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'LONG_TITLE']]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]]
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID']] = diagnoses[['SUBJECT_ID', 'HADM_ID']].astype(int)
    return diagnoses
def read_icd_procedures_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_PROCEDURES.csv'))
    codes = codes[['ICD9_CODE', 'LONG_TITLE']]
    procedures = dataframe_from_csv(os.path.join(mimic3_path, 'PROCEDURES_ICD.csv'))
    procedures = procedures[["SUBJECT_ID","HADM_ID","ICD9_CODE"]]
    procedures = procedures.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    procedures[['SUBJECT_ID', 'HADM_ID']] = procedures[['SUBJECT_ID', 'HADM_ID']].astype(int)
    return procedures


note_events = dataframe_from_csv(os.path.join(args.mimic3_path, 'NOTEEVENTS.csv'))
note_event_text = note_events[['SUBJECT_ID', 'HADM_ID', 'TEXT']].to_csv(os.path.join(args.output_path, 'all_note_event.csv'), index=False)
diagnoses = read_icd_diagnoses_table(args.mimic3_path)
procedures = read_icd_procedures_table(args.mimic3_path)
diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
procedures.to_csv(os.path.join(args.output_path, 'all_procedures.csv'), index=False)
