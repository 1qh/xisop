### Verify examples

Expect

```sh
python test_prompts.py google/gemma-3-1b-it
```

and

```sh
python test_pairs.py google/gemma-3-1b-it
```

give same results

## Data preparation

```sh
for f in gen-raw/*.jsonl ; python prep-gen.py $f ; end
```

## Gathering sets

```sh
for f in gen-clean/* ; python gather.py $f ; end
```
