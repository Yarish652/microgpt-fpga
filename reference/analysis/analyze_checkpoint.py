"""
Feasibility measurements on an exported microgpt checkpoint (see run_upstream.py).

1. NumPy forward pass, cross-checked against logits computed by the upstream code.
2. Held-out evaluation (teacher forcing): NLL and top-1 accuracy, FP64 vs FP32.
3. Per-stage value ranges (empirical) and weight-norm range bounds (static).
4. Preliminary weight-only quantization sensitivity (NOT a hardware result).
5. Preliminary fake-quantized 16/12-bit fixed-point activations (NOT bit-accurate to HLS).

usage: python analyze_checkpoint.py <checkpoint.json> [out.json]
"""
import json, sys
import numpy as np

ck = json.load(open(sys.argv[1]))
meta, sd = ck['meta'], {k: np.array(v, dtype=np.float64) for k, v in ck['state_dict'].items()}
E, H, BS, V, BOS = meta['n_embd'], meta['n_head'], meta['block_size'], meta['vocab_size'], meta['BOS']
HD = E // H
uchars = meta['uchars']
EPS = 1e-5

def rmsnorm(x):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)

def forward(W, tok, dtype=np.float64, taps=None, qa=None):
    """Batched teacher-forced forward. tok: (N, T) int. Mirrors upstream gpt() op order per position.
    qa(name, x) optionally fake-quantizes each intermediate tensor (accumulators stay exact)."""
    Q = (lambda name, x: qa(name, x)) if qa else (lambda name, x: x)
    W = {k: v.astype(dtype) for k, v in W.items()}
    N, T = tok.shape
    x = Q('emb_sum', W['wte'][tok] + W['wpe'][np.arange(T)][None])  # (N,T,E)
    if taps is not None: taps['emb_sum'] = x
    x = Q('rmsnorm0', rmsnorm(x)); x_res = x
    if taps is not None: taps['rmsnorm0'] = x
    xn = Q('rmsnorm1', rmsnorm(x))
    if taps is not None: taps['rmsnorm1'] = xn
    q, k, v = (Q(n, xn @ W[f'layer0.attn_w{n}'].T) for n in 'qkv')  # (N,T,E)
    qh, kh, vh = (a.reshape(N, T, H, HD).transpose(0, 2, 1, 3) for a in (q, k, v))  # (N,H,T,HD)
    s = Q('attn_scores', (qh @ kh.transpose(0, 1, 3, 2)) / dtype(HD ** 0.5))  # (N,H,T,T)
    mask = np.triu(np.ones((T, T), bool), 1)
    s_masked = np.where(mask, -np.inf, s)
    p = np.exp(s_masked - s_masked.max(-1, keepdims=True)); p = Q('attn_probs', p / p.sum(-1, keepdims=True))
    o = Q('attn_out', (p @ vh).transpose(0, 2, 1, 3).reshape(N, T, E))
    a = Q('attn_proj', o @ W['layer0.attn_wo'].T)
    x = Q('resid1', a + x_res); x_res = x
    xn2 = Q('rmsnorm2', rmsnorm(x))
    h = Q('fc1', xn2 @ W['layer0.mlp_fc1'].T)
    hr = np.maximum(h, 0)
    f = Q('fc2', hr @ W['layer0.mlp_fc2'].T)
    x = Q('resid2', f + x_res)
    logits = Q('logits', x @ W['lm_head'].T)
    if taps is not None:
        taps.update(q=q, k=k, v=v, attn_scores=np.where(mask, 0, s), attn_probs=p, attn_out=o, attn_proj=a,
                    resid1=x_res, rmsnorm2=xn2, fc1=h, relu=hr, fc2=f, resid2=x, logits=logits)
    return logits

def encode(doc):
    t = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n = min(BS, len(t) - 1)
    return t[:n], t[1:n + 1]

def batch(docs):
    X = np.full((len(docs), BS), BOS, np.int64); Y = np.zeros_like(X); M = np.zeros(X.shape, bool)
    for i, d in enumerate(docs):
        x, y = encode(d); X[i, :len(x)] = x; Y[i, :len(y)] = y; M[i, :len(x)] = True
    return X, Y, M

def nll_stats(logits, Y, M):
    lp = logits - logits.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    tok_nll = -np.take_along_axis(lp, Y[..., None], -1)[..., 0]
    per_doc = (tok_nll * M).sum(1) / M.sum(1)
    return {'token_mean_nll': float(tok_nll[M].mean()), 'doc_mean_nll': float(per_doc.mean()),
            'top1_acc': float((logits.argmax(-1) == Y)[M].mean())}

R = {'meta': meta}

# 1) cross-check against upstream-computed logits (causality: later padding cannot affect earlier positions)
errs = []
for pr in ck['probes']:
    n = len(pr['logits'])
    out = forward(sd, np.array([pr['tokens'][:n]]))[0]
    errs.append(np.abs(out - np.array(pr['logits'])).max())
R['crosscheck_max_abs_err_fp64_vs_upstream'] = float(max(errs))

# 2) held-out evaluation set: post-shuffle docs never trained on (and not duplicates of trained names)
docs = ck['docs_order']; ns = meta['num_steps']
trained = set(docs[:ns])
heldout = [d for d in docs[ns:] if d not in trained]
X, Y, M = batch(heldout)
R['eval'] = {'n_docs': len(heldout), 'n_tokens': int(M.sum())}

taps64 = {}
L64 = forward(sd, X, np.float64, taps64)
L32 = forward(sd, X, np.float32).astype(np.float64)
R['fp64'] = nll_stats(L64, Y, M)
R['fp32'] = nll_stats(L32, Y, M)
R['fp32_vs_fp64'] = {'max_abs_logit_err': float(np.abs(L32 - L64)[M].max()),
                     'top1_agreement': float((L32.argmax(-1) == L64.argmax(-1))[M].mean())}

# 3) ranges: empirical over held-out set, and static bounds from weight norms (|rmsnorm out|_2 <= sqrt(E))
R['weights'] = {k: {'shape': list(v.shape), 'max_abs': float(np.abs(v).max()), 'std': float(v.std()),
                    'max_row_l2': float(np.linalg.norm(v, axis=1).max())} for k, v in sd.items()}
R['activations_max_abs'] = {}
for k, v in taps64.items():
    vm = v[M] if v.shape[:2] == M.shape else v.transpose(0, 2, 1, 3)[M]  # (N,T,...) or (N,H,T,T)
    R['activations_max_abs'][k] = float(np.abs(vm).max())
R['attn_probs_min_nonzero'] = float(taps64['attn_probs'][taps64['attn_probs'] > 0].min())
r = np.sqrt(E)
R['static_bounds'] = {'rmsnorm_out_elem': float(r),
                      **{f'{n}': float(np.linalg.norm(sd[f'layer0.attn_w{n}'], axis=1).max() * r) for n in 'qkv'},
                      'fc1': float(np.linalg.norm(sd['layer0.mlp_fc1'], axis=1).max() * r)}
R['double_rmsnorm_max_abs_diff'] = float(np.abs(taps64['rmsnorm1'] - taps64['rmsnorm0'])[M].max())

# 4) weight-only symmetric per-tensor quantization (round-to-nearest), FP64 compute. Preliminary.
def quant_w(W, bits):
    qmax = 2 ** (bits - 1) - 1
    return {k: np.round(v / (np.abs(v).max() / qmax)) * (np.abs(v).max() / qmax) for k, v in W.items()}

R['weight_only_quant'] = {}
for bits in (16, 12, 8, 6, 4):
    Lq = forward(quant_w(sd, bits), X)
    R['weight_only_quant'][f'int{bits}'] = {**nll_stats(Lq, Y, M),
                                            'top1_agreement_vs_fp64': float((Lq.argmax(-1) == L64.argmax(-1))[M].mean()),
                                            'max_abs_logit_err': float(np.abs(Lq - L64)[M].max())}

# 5) fake-quantized activations: 16-bit signed fixed point, per-tensor integer bits chosen from the
#    measured ranges above (I includes the sign bit). Nonlinear functions in FP64. Preliminary.
A16_INT_BITS = {'emb_sum': 2, 'rmsnorm0': 3, 'rmsnorm1': 3, 'q': 3, 'k': 3, 'v': 3, 'attn_scores': 4,
                'attn_probs': 1, 'attn_out': 3, 'attn_proj': 2, 'resid1': 3, 'rmsnorm2': 3, 'fc1': 3,
                'fc2': 4, 'resid2': 4, 'logits': 6}

def make_qa(width, rounding):
    def qa(name, x):
        F = width - A16_INT_BITS[name]  # attn_probs is unsigned: no sign bit, so range is [0, 2)
        scaled = x * 2.0 ** F
        y = np.floor(scaled + 0.5) if rounding == 'round' else np.floor(scaled)  # floor == ap_fixed AP_TRN
        lo, hi = (0, 2 ** width - 1) if name == 'attn_probs' else (-2 ** (width - 1), 2 ** (width - 1) - 1)
        return np.clip(y, lo, hi) / 2.0 ** F                                    # saturating
    return qa

top2 = np.sort(L64, -1)[..., -2:]
fp64_margin = top2[..., 1] - top2[..., 0]  # top-1 minus top-2 logit; small margin = near-tie
R['fake_quant_activations'] = {}
for label, wbits, abits, rnd in (('W16A16_round', 16, 16, 'round'), ('W16A16_trunc', 16, 16, 'trunc'),
                                 ('W8A16_round', 8, 16, 'round'), ('W16A12_round', 16, 12, 'round')):
    Lq = forward(quant_w(sd, wbits), X, qa=make_qa(abits, rnd))
    disagree = (Lq.argmax(-1) != L64.argmax(-1)) & M
    R['fake_quant_activations'][label] = {**nll_stats(Lq, Y, M),
                                          'top1_agreement_vs_fp64': float(1 - disagree.sum() / M.sum()),
                                          'n_disagree': int(disagree.sum()),
                                          'max_fp64_top2_margin_at_disagreement': float(fp64_margin[disagree].max()) if disagree.any() else 0.0,
                                          'max_abs_logit_err': float(np.abs(Lq - L64)[M].max()),
                                          'mean_logit_err': float((Lq - L64)[M].mean())}

# greedy generation from BOS (deterministic demo target for hardware)
def greedy(W, prefix=''):
    toks = [BOS] + [uchars.index(c) for c in prefix]
    while len(toks) < BS:
        nxt = int(forward(W, np.array([toks]))[0, -1].argmax())
        if nxt == BOS: break
        toks.append(nxt)
    return ''.join(uchars[t] for t in toks[1:])
R['greedy_from_BOS'] = greedy(sd)
R['greedy_prefixes'] = {p: greedy(sd, p) for p in ('a', 'b', 'ka', 'mar', 'zo')}

out = json.dumps(R, indent=1)
print(out)
if len(sys.argv) > 2: open(sys.argv[2], 'w').write(out)
