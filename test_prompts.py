import json
import sys
from pathlib import Path

from ps import PromptSensitivity

SETS = [
  [
    'Q: What is the capital of France?\nA: ',
    'Q: WHat is te capital city of France?\nA: ',
    'Q: what is teh cpital of france??\nA: ',
  ],
  [
    'Q: What is the national animal of India?\nA: ',
    "Q: What's the national animl of India?\nA: ",
    'Q: WHat is teh national animal of India?\nA: ',
  ],
  [
    'Q: Tell me the meaning of rendezvous?\nA: ',
    'Q: WHat is the meaning of rendezvous?\nA: ',
    'Q: What does rendezvous mean?\nA: ',
  ],
]

if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit(f'Usage: python {sys.argv[0]} <model-id>')

  model_id = sys.argv[1]
  trace = PromptSensitivity(model_id).from_prompt_sets(SETS)
  Path(sys.argv[0].replace('.py', '.json')).write_text(json.dumps(trace, indent=2), encoding='utf-8')
