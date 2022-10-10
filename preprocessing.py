#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 22:39:36 2022

@author: jinnieshin
"""

import pandas as pd
import numpy as np 
import nltk
from collections import Counter
#nltk.download('punkt')
#nltk.download('wordnet')
nltk.download('omw-1.4')
def data_import(data='Book2'):
    dat = pd.read_csv(data+'.csv')
    dat = dat[dat.vid.apply(lambda x: len(str(x))>4)]

    def clean(string):
        string = str(string)
        string  = string.replace('- [Narrator]', '')
        string = string.replace('\n', ' ')
        string = string.replace('-', ' ')
        string = string.replace("don't", "do not")
        string = string.replace("can't", "cannot")
        string = string.replace("'ll'", " will")
        string = string.replace("she's", "she is")
        string = string.replace("he's", "he is")
        string = string.replace("'re'", " are")
        string = string.replace("it's", "it is")
        string = string.replace("I'm", "I am")
        string = string.strip()
        return string 

    dat['text'] = dat.text.apply(clean)
    #dat['start_sent'] = dat.text.apply(nltk.word_tokenize).apply(lambda x: x[-1] in ['.', '?'])
    #dat['start_sent'] = dat.start_sent.replace({False:'', True:'//end//'})
    #dat['text']  = dat.text + ' ' + dat.start_sent

    #df = pd.DataFrame(
    #    ''.join(dat.text.values)
    #    .split('//end//'), columns=['text']
    # )

    #df_short = dat[dat.start_sent == '//end//']    
    #df_short['sentence'] = df.text[:-1].tolist()

    df_short = dat
    df_short = df_short[['vid', 
                         'Elaboration', 'Originality', 'Aesthetics ',
                         'Surprise', 'Humor', 'Complexity', 
                         'Novel use of materials', 'Realism/ recreation', 
                         'text']]
    return df_short


df_short = data_import(data='Book1')
df_test = data_import(data='Book2')
