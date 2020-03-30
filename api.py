#%%
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from summary_handler import Summarizer

import os
file_name = 'family_violence.txt'
input_file = os.path.join('data',file_name)
summary_ouput_file = os.path.join('data',"summary_"+file_name)
with open(input_file) as fp:
    text = fp.read()
    loaded_model = pickle.load(open('small_saved_model.pkl', 'rb'))
    summary = loaded_model(text,ratio=0.2,algorithm='kmeans',use_first=True,min_length=10,max_length=500)
    with open(summary_ouput_file,"w+") as fw: 
        #fw.write(summary)
        pass

# %%
