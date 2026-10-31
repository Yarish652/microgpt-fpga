"""
Stage 3 export (FP32 build): weight image, manifest and test vectors from the frozen checkpoint.

usage: python export_fp32.py <checkpoint.json> <out_dir>

Outputs in <out_dir>:
  weights_fp32.bin     little-endian float32, tensors concatenated row-major [out][in], in MANIFEST order
  manifest.json        per-tensor name/shape/offset/sha256, plus whole-file sha256 and checkpoint provenance
  vectors/             test vectors (see vectors/vectors.json for the exact record layout):
      <case>.tokens.i32      input token ids, one per position (T <= 16)
      <case>.taps_fp32.f32   per-position records of all taps, float32 (incremental/KV-cache order)
      <case>.taps_fp64.f32   same taps computed in FP64, stored as float32 (reference for tolerance checks)
      <case>.argmax.i32      FP32 argmax of the logits per position (first maximum wins ties)

The step function below is the hardware dataflow (one token at a time, KV cache), NOT the batched
teacher-forced model in analyze_checkpoint.py. The script checks the two agree before writing anything.
"""
import hashlib, json, os, sys
import numpy as np

ck_path, out_dir = sys.argv[1], sys.argv[2]
raw = open(ck_path, 'rb').read()
ck = json.loads(raw)
meta = ck['meta']
E, H, BS, V, BOS = meta['n_embd'], meta['n_head'], meta['block_size'], meta['vocab_size'], meta['BOS']
HD = E // H
EPS = 1e-5
uchars = meta['uchars']

# Order from plan §4.3. Do not reorder: the manifest offsets and the HLS LOAD code depend on it.
ORDER = ['wte', 'wpe', 'layer0.attn_wq', 'layer0.attn_wk', 'layer0.attn_wv', 'layer0.attn_wo',
         'layer0.mlp_fc1', 'layer0.mlp_fc2', 'lm_head']
sd64 = {k: np.array(ck['state_dict'][k], dtype=np.float64) for k in ORDER}
assert sum(v.size for v in sd64.values()) == meta['num_params'] == 4192
assert set(sd64) == set(ck['state_dict'])

# Per-position tap record: (name, size). Matches plan §9.3. s and p are [H][BS], zero-padded beyond T.
TAPS = [('x0', E), ('x1', E), ('a_in', E), ('q', E), ('k', E), ('v', E), ('s', H * BS), ('p', H * BS),
        ('o', E), ('attn_proj', E), ('x2', E), ('m_in', E), ('hdn_pre', 64), ('hdn', 64),
        ('ffn_out', E), ('x3', E), ('logits', V)]
REC = sum(n for _, n in TAPS)


def rms(x, dt):
    return x / np.sqrt((x * x).mean() + dt(EPS))


def step(W, kc, vc, tok, pos, dt):
    """One token through the model with a KV cache. kc, vc: lists of per-position (E,) arrays, appended in place."""
    t = {}
    t['x0'] = W['wte'][tok] + W['wpe'][pos]
    t['x1'] = x = rms(t['x0'], dt)
    t['a_in'] = a = rms(x, dt)                                   # second, back-to-back norm (as upstream)
    t['q'] = q = W['layer0.attn_wq'] @ a
    t['k'] = k = W['layer0.attn_wk'] @ a
    t['v'] = v = W['layer0.attn_wv'] @ a
    kc.append(k); vc.append(v)
    K, Vv = np.stack(kc), np.stack(vc)                           # (T, E)
    T = K.shape[0]
    s_all = np.zeros((H, BS), dt); p_all = np.zeros((H, BS), dt); o = np.zeros(E, dt)
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        s = (K[:, sl] @ q[sl]) / dt(HD ** 0.5)                   # exact *0.5
        e = np.exp(s - s.max())
        p = e / e.sum()
        s_all[h, :T], p_all[h, :T] = s, p
        o[sl] = p @ Vv[:, sl]
    t['s'], t['p'], t['o'] = s_all.reshape(-1), p_all.reshape(-1), o
    t['attn_proj'] = ap = W['layer0.attn_wo'] @ o
    t['x2'] = x2 = ap + x
    t['m_in'] = m = rms(x2, dt)
    t['hdn_pre'] = hp = W['layer0.mlp_fc1'] @ m
    t['hdn'] = hd = np.maximum(hp, 0)
    t['ffn_out'] = f = W['layer0.mlp_fc2'] @ hd
    t['x3'] = x3 = f + x2
    t['logits'] = W['lm_head'] @ x3
    return t


def run(W, tokens, dt):
    kc, vc, recs, am = [], [], [], []
    for pos, tok in enumerate(tokens):
        t = step(W, kc, vc, tok, pos, dt)
        recs.append(np.concatenate([np.asarray(t[n], dt).reshape(-1) for n, _ in TAPS]))
        am.append(int(np.argmax(t['logits'])))                   # first maximum = lowest index on ties
    return np.stack(recs), np.array(am, np.int32)


def greedy(W, prefix, dt):
    toks = [BOS] + [uchars.index(c) for c in prefix]
    kc, vc = [], []
    for pos, tok in enumerate(toks):
        t = step(W, kc, vc, tok, pos, dt)
    while len(toks) < BS:
        nxt = int(np.argmax(t['logits']))
        if nxt == BOS:
            break
        toks.append(nxt)
        t = step(W, kc, vc, nxt, len(toks) - 1, dt)
    return ''.join(uchars[x] for x in toks[1:])


W64 = sd64
W32 = {k: v.astype(np.float32) for k, v in sd64.items()}

# ---- checks before writing anything ----
# (1) FP64 incremental step vs the logits the upstream code produced (probes in the checkpoint)
err = 0.0
for pr in ck['probes']:
    n = len(pr['logits'])
    recs, _ = run(W64, pr['tokens'][:n], np.float64)
    err = max(err, np.abs(recs[:, REC - V:] - np.array(pr['logits'])).max())
print(f'check 1: FP64 incremental vs upstream logits, max abs err = {err:.3e}')
assert err < 1e-12, err

# (2) greedy demo strings must equal the FP64 reference recorded in PROJECT_CONTEXT.md
EXPECT = {'': 'anan', 'a': 'anan', 'b': 'bina', 'ka': 'karian', 'mar': 'marian', 'zo': 'zona'}
greedy_out = {}
for pre, want in EXPECT.items():
    g64, g32 = greedy(W64, pre, np.float64), greedy(W32, pre, np.float32)
    print(f'check 2: prefix {pre!r:6} fp64={g64!r} fp32={g32!r} expected={want!r}')
    assert g64 == want and g32 == want
    greedy_out[pre] = want

# ---- weight image ----
os.makedirs(os.path.join(out_dir, 'vectors'), exist_ok=True)
blob, tensors, off = b'', [], 0
for k in ORDER:
    b = W32[k].astype('<f4').tobytes()
    tensors.append({'name': k, 'shape': list(W32[k].shape), 'dtype': 'float32', 'offset_elems': off // 4,
                    'offset_bytes': off, 'count': int(W32[k].size), 'sha256': hashlib.sha256(b).hexdigest()})
    blob += b; off += len(b)
open(os.path.join(out_dir, 'weights_fp32.bin'), 'wb').write(blob)

# (3) round trip: reload the .bin per the manifest -> identical logits to the in-memory FP32 weights
rt = {}
flat = np.frombuffer(open(os.path.join(out_dir, 'weights_fp32.bin'), 'rb').read(), dtype='<f4')
for t in tensors:
    rt[t['name']] = flat[t['offset_elems']:t['offset_elems'] + t['count']].reshape(t['shape']).astype(np.float32)
seq = [BOS, 3, 4, 13, 19, 14, 13]
a, _ = run(W32, seq, np.float32); b, _ = run(rt, seq, np.float32)
assert np.array_equal(a, b)
print('check 3: .bin round trip -> bit-identical FP32 logits')

manifest = {
    'format': 'float32 little-endian, row-major [out][in]',
    'checkpoint_file': os.path.basename(ck_path),
    'checkpoint_file_sha256': hashlib.sha256(raw).hexdigest(),
    'state_dict_sha256_expected': '1761e472e0c66d17274ded4fbd62a6512df9a7f86f8d09a1202375cc0f422e8a',
    'weights_file': 'weights_fp32.bin', 'weights_file_sha256': hashlib.sha256(blob).hexdigest(),
    'weights_total_bytes': len(blob), 'num_params': int(flat.size), 'tensors': tensors,
}
json.dump(manifest, open(os.path.join(out_dir, 'manifest.json'), 'w'), indent=1)

# ---- test vectors ----
docs = ck['docs_order']; ns = meta['num_steps']
trained = set(docs[:ns])
heldout = [d for d in docs[ns:] if d not in trained]


def pick(length, skip=0):
    return [d for d in heldout if len(d) == length][skip]


def enc(prefix):
    return [BOS] + [uchars.index(c) for c in prefix]


cases = {'bos_only': [BOS]}                                    # T=1: softmax over one element, p = 1.0
for L in (1, 2, 3, 6, 7, 15):
    name = pick(L) if L > 1 else None
    if name is None:                                           # dataset has no 1-char names: use 'a' directly
        name = 'a'
    cases[f'len{L:02d}_{name}'] = enc(name)[:BS]               # 15-char name -> 16 inputs = full context
for pre in ('a', 'b', 'ka', 'mar', 'zo'):
    cases[f'prefix_{pre}'] = enc(pre)
# a position/token sweep: every token id at every position needs history, so use a fixed 16-token ramp
cases['ramp_all_pos'] = [BOS] + [(i * 7) % 26 for i in range(BS - 1)]
cases['bos_repeated'] = [BOS] * BS                             # BOS at every position (BOS is also the end token)

index = {}
for name, toks in cases.items():
    assert 1 <= len(toks) <= BS and all(0 <= t < V for t in toks)
    r32, am = run(W32, toks, np.float32)
    r64, _ = run(W64, toks, np.float64)
    vd = os.path.join(out_dir, 'vectors', name)
    np.array(toks, '<i4').tofile(vd + '.tokens.i32')
    r32.astype('<f4').tofile(vd + '.taps_fp32.f32')
    r64.astype('<f4').tofile(vd + '.taps_fp64.f32')
    am.astype('<i4').tofile(vd + '.argmax.i32')
    index[name] = {'positions': len(toks), 'tokens': toks,
                   'max_abs_fp32_vs_fp64_logits': float(np.abs(r32[:, REC - V:].astype(np.float64) - r64[:, REC - V:]).max())}

offs, o = {}, 0
for n, sz in TAPS:
    offs[n] = {'offset_floats': o, 'size_floats': sz}; o += sz
json.dump({'record_floats': REC, 'taps': offs, 'dtype': 'float32 little-endian',
           'note': 's and p are [4 heads][16], zero beyond T=pos+1. One record per position, positions in order.',
           'greedy_expected': greedy_out, 'cases': index},
          open(os.path.join(out_dir, 'vectors', 'vectors.json'), 'w'), indent=1)
print(f'wrote {len(blob)} bytes of weights, {len(cases)} test cases, record = {REC} floats')
print('weights sha256', manifest['weights_file_sha256'])
