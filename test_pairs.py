import json
import sys
from pathlib import Path

from ps import PromptSensitivity

SETS = [
  [
    {
      'prompt': 'Q: What is the capital of France?\nA: ',
      'answer': '\nParis\n\n<end_of_turn>',
    },
    {
      'prompt': 'Q: WHat is te capital city of France?\nA: ',
      'answer': '\nParis\n\nLet me',
    },
    {
      'prompt': 'Q: what is teh cpital of france??\nA: ',
      'answer': '\nParis\n\n<end_of_turn>',
    },
  ],
  [
    {
      'prompt': 'Q: What is the national animal of India?\nA: ',
      'answer': '\nThe national animal of',
    },
    {
      'prompt': "Q: What's the national animl of India?\nA: ",
      'answer': '\nThe national animal of',
    },
    {
      'prompt': 'Q: WHat is teh national animal of India?\nA: ',
      'answer': '\nThe national animal of',
    },
  ],
  [
    {
      'prompt': 'Q: Tell me the meaning of rendezvous?\nA: ',
      'answer': '\n(A) A',
    },
    {
      'prompt': 'Q: WHat is the meaning of rendezvous?\nA: ',
      'answer': '\n(1) A',
    },
    {
      'prompt': 'Q: What does rendezvous mean?\nA: ',
      'answer': '\nIt means a meeting',
    },
  ],
]


if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit(f'Usage: python {sys.argv[0]} <model-id>')

  model_id = sys.argv[1]
  trace = PromptSensitivity(model_id).from_pair_sets(SETS)
  Path(sys.argv[0].replace('.py', '.json')).write_text(json.dumps(trace, indent=2), encoding='utf-8')
