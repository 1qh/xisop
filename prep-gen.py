import sys
from pathlib import Path

from pandas import read_json

if len(sys.argv) < 2:
  print(f'Usage: {sys.argv[0]} <file.json>')
  sys.exit(1)

BASE_OUTPUT_DIR = 'gen-clean'
filepath = sys.argv[1]
filename = Path(filepath).stem
Path(f'{BASE_OUTPUT_DIR}/{filename}').mkdir(parents=True, exist_ok=True)


df = (
  read_json(
    filepath,
    lines=True,
    dtype={'query_id': 'str'},
  )
  .drop(columns=['model', 'generated_on'])
  .rename(
    columns={
      'query_id': 'set',
      'query_text': 'prompt',
    }
  )
)

df['set'] = df['set'].str.split('_').str[0]


for set_id, data in df.groupby('set'):
  data.drop(columns=['set']).to_parquet(f'{BASE_OUTPUT_DIR}/{filename}/{set_id}.parquet')
