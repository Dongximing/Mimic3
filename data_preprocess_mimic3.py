import csv
import pandas as pd
import numpy as np
import os
import sys
note_events = pd.read_csv("/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4/NOTEEVENTS.csv",sep=',')
note_event_text = note_events[['SUBJECT_ID', 'HADM_ID', 'TEXT']].to_csv('/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4/NOTEEVENTS_TEXT.csv',index=False)



