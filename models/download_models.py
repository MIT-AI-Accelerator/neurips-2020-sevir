"""
Downloads pretrained models for nowcast and synrad
"""
import pandas as pd
import requests

def main():
	model_info = pd.read_csv('model_urls.csv')
	for i,r in model_info.iterrows():
		print(f'Downloading {r.model}...')
		req = requests.get(r.url,allow_redirects=True)
		with open(f'{r.application}/{r.model}','wb') as f:
			f.write(req.content)

if __name__=='__main__':
	main()



