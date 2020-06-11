"""
Downloads pretrained models for nowcast and synrad
"""
import pandas as pd
import urllib.request
import os

def main():
    model_info = pd.read_csv('model_urls.csv')
    for i,r in model_info.iterrows():
        print(f'Downloading {r.model}...')
        download_file(r.url,f'{r.application}/{r.model}')

def download_file(url,filename):
    os.system(f'wget -O {filename} {url}')

if __name__=='__main__':
    main()



