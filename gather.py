import json
import sys
from pathlib import Path

from pandas import read_parquet

if len(sys.argv) < 2:
  print(f'Usage: {sys.argv[0]} </path/to/dir>')
  sys.exit(1)

BASE_OUTPUT_DIR = 'gen-set'
Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

input_dir = sys.argv[1]
output_name = Path(input_dir).name


for filepath in Path(input_dir).glob('*.parquet'):
  df = read_parquet(filepath).to_dict(orient='records')


Path(f'{BASE_OUTPUT_DIR}/{output_name}.json').write_text(
  json.dumps(
    [
      read_parquet(f).to_dict(
        orient='records',
      )
      for f in Path(input_dir).glob('*.parquet')
    ],
    indent=2,
  ),
  encoding='utf-8',
)
