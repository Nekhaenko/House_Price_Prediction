from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json
import os
import torch


app = Flask(__name__)
client = app.test_client()


PATH = 'torch_model.bin'
mlp = torch.load(PATH)


def clean(x):
  return x[:10]


def new_address(x):
  x = x.split(', ')
  len_x = len(x)

  if len_x < 7:
    x = ['']*(8 - len_x) + x

  return x[-7:]


addr_cols = ['address1', 'address2', 'address3', 'address4', 'address5', 
             'address6', 'address7']


@app.route('/predict', methods=['POST'])
def udate_list():
    x = request.json
    
    df = pd.DataFrame(x)
    df['date'] = df['date'].apply(clean)
    df['date'] = pd.to_datetime(df['date'])
    
    df[addr_cols] = ''

    for i in df.index:
      address = new_address(df.loc[i, 'address'])
      df.loc[i, addr_cols] = address
      
    if 'price' in df.columns:
        df.drop(["price"],axis=1,inplace=True)
        
    df.drop(["address"],axis=1,inplace=True)
    
    cat_cols = ['type', 'bedrooms', 'area', 'tenure', 'is_newbuild'] + addr_cols
    
    for feature in cat_cols:
        with open(f'encode/{feature}', 'rb') as fp:
            lbl_encoders = pickle.load(fp)
            df[feature]= lbl_encoders.fit_transform(df[feature].astype(str))
  
    inp =df.loc[0].to_numpy()
    inp = torch.from_numpy(inp)
    inp = inp.float()
    y = mlp(x).tolist()
    
    return jsonify({"prediction":y})


if __name__=='__main__':
    app.run()