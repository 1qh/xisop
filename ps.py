from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from os import environ
from time import perf_counter

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Embedding
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

environ['TOKENIZERS_PARALLELISM'] = 'true'

MAX_NEW_TOKENS = 5
EMBED_MODEL = 'embeddinggemma:300m-qat-q4_0'
load_dotenv()
oai = OpenAI()


def embed(a: list[str]) -> list[Embedding]:
  return oai.embeddings.create(model=EMBED_MODEL, input=a).data


def norm_str(s: str) -> str:
  return ' '.join((s or '').strip().split())


class PromptSensitivity:
  def __init__(self, model_id: str) -> None:
    self.tok = AutoTokenizer.from_pretrained(model_id)
    self.lm = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    self.device = self.lm.device
    self.pad = self.tok.pad_token_id

  @torch.inference_mode()
  def gen(self, prompt: str) -> tuple[list[int], str]:
    batch = self.tok(prompt, return_tensors='pt').to(self.device)
    tokens = self.lm.generate(**batch, pad_token_id=self.pad, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)[
      0, batch.input_ids.size(1) :
    ]
    return tokens.tolist(), self.tok.decode(tokens, skip_special_tokens=False)

  @torch.inference_mode()
  def log_prob(self, prompt_tokens: list[int], an_tokens: list[int]) -> float:
    an = torch.tensor(an_tokens, device=self.device)
    return (
      log_softmax(self.lm(torch.tensor([prompt_tokens + an_tokens], device=self.device)).logits[0], dim=-1)[
        torch.arange(an.numel(), device=self.device) + len(prompt_tokens) - 1, an
      ]
      .sum()
      .item()
    )

  def response_distribution_entropy(self, texts: Iterable[str]) -> dict:
    normed = [norm_str(t) for t in texts]
    counts = Counter(normed)
    n = sum(counts.values())
    if n == 0:
      return {'entropy': 0.0, 'entropy_norm': 0.0, 'unique': 0, 'n': 0, 'counts': {}}
    p = np.array([c / n for c in counts.values()], dtype=float)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    k = len(p)
    entropy_norm = float(entropy / math.log(k)) if k > 1 else 0.0
    return {
      'entropy': entropy,
      'entropy_norm': entropy_norm,
      'unique': k,
      'n': n,
      'counts': dict(counts),
    }

  def avg_cos_sim(self, texts: Iterable[str]) -> float:
    texts = [norm_str(t) for t in texts]
    if len(texts) <= 1:
      return 1.0
    X = np.asarray([d.embedding for d in sorted(embed(texts), key=lambda d: d.index)], dtype=float)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    S = X @ X.T
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(S[iu].mean()) if iu[0].size else 1.0

  def score_with_gen(self, prompts: list[str]) -> tuple[list[str], float, dict, float]:
    t0 = perf_counter()
    tokens, ans = zip(*(self.gen(p) for p in prompts), strict=True)
    mat = [[self.log_prob(p, r) for r in tokens] for p in self.tok(prompts).input_ids]
    n = len(prompts)
    sen = sum(abs(value - row[j]) / len(tokens[j]) for j, row in enumerate(mat) for value in row) / (n * (n - 1))
    rde = self.response_distribution_entropy(ans)
    avg_cos = self.avg_cos_sim(ans)
    print(f'{sen:.4f} | {perf_counter() - t0:.3f}s')
    return list(ans), sen, rde, avg_cos

  def from_prompt_sets(self, prompt_sets: list[list[str]]) -> dict:
    out_data = []
    posix_vals, rde_norm_vals, avg_cos_vals = [], [], []
    for prompts in prompt_sets:
      answers, sen, rde, avg_cos = self.score_with_gen(prompts)
      out_data.append({
        'pairs': [{'prompt': p, 'answer': a} for p, a in zip(prompts, answers, strict=True)],
        'sen': sen,
        'rde': rde,
        'avg_cosine': avg_cos,
      })
      posix_vals.append(sen)
      rde_norm_vals.append(rde['entropy_norm'])
      avg_cos_vals.append(avg_cos)
    return {
      'data': out_data,
      'posix': float(sum(posix_vals) / len(posix_vals)) if posix_vals else 0.0,
      'rde_norm': float(sum(rde_norm_vals) / len(rde_norm_vals)) if rde_norm_vals else 0.0,
      'avg_cosine': float(sum(avg_cos_vals) / len(avg_cos_vals)) if avg_cos_vals else 0.0,
    }

  def from_pair_sets(self, pair_sets: list[list[dict]]) -> dict:
    results = []
    posix_vals, rde_norm_vals, avg_cos_vals = [], [], []
    for p_idx, pairs in enumerate(pair_sets):
      t0 = perf_counter()
      prompts = [p['prompt'] for p in pairs]
      answers_txt = [p['answer'] for p in pairs]
      prompt_tok = self.tok(prompts).input_ids
      ans_tok = [self.tok(a, add_special_tokens=False)['input_ids'] for a in answers_txt]
      mat = [[self.log_prob(p_t, r_t) for r_t in ans_tok] for p_t in prompt_tok]
      n = len(prompts)
      sen = sum(abs(value - row[j]) / max(1, len(ans_tok[j])) for j, row in enumerate(mat) for value in row) / (
        n * (n - 1)
      )
      rde = self.response_distribution_entropy(answers_txt)
      avg_cos = self.avg_cos_sim(answers_txt)
      results.append({
        'pairs': pairs,
        'sen': sen,
        'rde': rde,
        'avg_cosine': avg_cos,
      })
      posix_vals.append(sen)
      rde_norm_vals.append(rde['entropy_norm'])
      avg_cos_vals.append(avg_cos)
      print(f'{p_idx:03} | {sen:.4f} | {perf_counter() - t0:.3f}s')
    return {
      'data': results,
      'posix': float(sum(posix_vals) / len(posix_vals)) if posix_vals else 0.0,
      'rde_norm': float(sum(rde_norm_vals) / len(rde_norm_vals)) if rde_norm_vals else 0.0,
      'avg_cosine': float(sum(avg_cos_vals) / len(avg_cos_vals)) if avg_cos_vals else 0.0,
    }
