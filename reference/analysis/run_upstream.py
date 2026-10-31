"""
Run the unmodified upstream microgpt.py and export its trained checkpoint.

The upstream file is exec'd as-is (training + its own sampling printout), then
the trained state_dict and a few upstream-computed logit vectors are written to
JSON so other tools can check themselves against the original implementation.

usage: python run_upstream.py <microgpt.py> <names.txt> <out_dir>
"""
import hashlib, json, os, platform, shutil, sys, time

src_path, names_path, out_dir = (os.path.abspath(p) for p in sys.argv[1:4])
os.makedirs(out_dir, exist_ok=True)
shutil.copyfile(names_path, os.path.join(out_dir, 'input.txt'))  # upstream reads ./input.txt
os.chdir(out_dir)

src = open(src_path, encoding='utf-8').read()
ns = {'__name__': '__main__'}
t0 = time.time()
exec(compile(src, src_path, 'exec'), ns)
train_s = time.time() - t0

# upstream-computed logits for a few never-trained-on documents (cross-check targets)
docs, uchars, BOS, n_layer, block_size = ns['docs'], ns['uchars'], ns['BOS'], ns['n_layer'], ns['block_size']
trained = set(docs[:ns['num_steps']])
probe_docs = [d for d in docs[ns['num_steps']:] if d not in trained][:20]
probes = []
for doc in probe_docs:
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    logits = [[v.data for v in ns['gpt'](tokens[p], p, keys, values)] for p in range(min(block_size, len(tokens) - 1))]
    probes.append({'doc': doc, 'tokens': tokens, 'logits': logits})

ckpt = {
    'meta': {
        'source_sha256': hashlib.sha256(src.encode('utf-8')).hexdigest(),
        'names_sha256': hashlib.sha256(open('input.txt', 'rb').read()).hexdigest(),
        'python': sys.version, 'platform': platform.platform(), 'train_seconds': round(train_s, 1),
        'n_layer': n_layer, 'n_embd': ns['n_embd'], 'n_head': ns['n_head'], 'block_size': block_size,
        'vocab_size': ns['vocab_size'], 'BOS': BOS, 'uchars': ''.join(uchars), 'num_steps': ns['num_steps'],
        'num_params': len(ns['params']),
    },
    'state_dict': {k: [[p.data for p in row] for row in mat] for k, mat in ns['state_dict'].items()},
    'docs_order': docs,  # post-shuffle order; docs[:num_steps] were trained on
    'probes': probes,
}
with open('checkpoint.json', 'w') as f:
    json.dump(ckpt, f)
print(f"\nwrote {os.path.join(out_dir, 'checkpoint.json')} ({train_s:.1f}s)")
