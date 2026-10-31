# PROJECT_CONTEXT: microGPT FPGA inference accelerator

> Living document. Update **Status**, **Decisions**, **Open questions** and **Results** at the end of
> every work session. The full rationale is in [docs/01_feasibility_and_architecture_plan.md](docs/01_feasibility_and_architecture_plan.md)
> (§ references below point there). Evidence labels: [SRC] source-verified, [MEAS] measured here,
> [DER] derived, [EST] estimate, [PROP] proposed, [OPEN] unresolved.

Last updated: 2026-10-16. Phase: **Stage 1: tool install pending; Stage 3 (FP32 export) done**.
**Minor-project deadline: 2026-11-21 [user-stated]**, ~5 weeks. Team: 2 people, "12 hours" each-or-total and per week
not yet clarified [OPEN]. The plan's 12–22 week critical path [EST] does not fit, so scope is cut to **FP32 only** (§5).

## 1. Goal and non-goals

**Goal (minor project):** run inference for Karpathy's microGPT (4,192 parameters) on a physical AMD/Xilinx
FPGA. Verify it against a software golden reference (bit-exact vs emulator/co-sim, quantified error vs FP64),
and report measured latency, resources and numerical quality with a reproducible method.

**Goal (major project):** a controlled study of precision (16/8/4-bit) × MAC parallelism (P) on the
latency–area–accuracy trade-off (§14).

**Non-goals:** on-FPGA training; claiming speedups that were not measured; on-device temperature sampling;
hardware tokenizer; INT8/INT4 or wide MAC arrays before the baseline exists.

## 2. Verified model specification

| Item | Value |
|---|---|
| Source | gist `karpathy/8627fe009c40f57531cb18360106ce95` @ `14fb038816c7aae0bb9342c2dbf1a51dd134a5ff` (2026-02-16); `microgpt.py` sha256 `d47d88c2fd432c8ebdc1048beab7f7eb64ea7e0e664e11b812d72a6d95ebccee` [SRC] |
| ⚠ Older revisions | Before `cf7331b` (2026-02-12 10:43 PST): block_size 8 (4,064 params), squared ReLU, different init/optimizer. **Do not use.** |
| Dataset | makemore `names.txt` @ `988aa59`, sha256 `0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d`; 32,033 names (29,494 unique), a–z, length 2–15 |
| Vocab | 27 = a..z (ids 0–25) + BOS (id 26); BOS also ends a sequence |
| Dims | n_embd 16, n_head 4, head_dim 4, n_layer 1, block_size 16, MLP hidden 64 |
| Params | **4,192 exactly** [SRC][MEAS]: wte 27×16, wpe 16×16, Wq/Wk/Wv/Wo 16×16, fc1 64×16, fc2 16×64, lm_head 27×16; row-major `[out][in]`; no biases; untied lm_head |
| Norm | RMSNorm without gain, `x·(mean(x²)+1e-5)^-0.5`; applied after embedding, again pre-attention (back-to-back), pre-MLP; **no final norm** |
| Positions | learned absolute `wpe`; pos ∈ [0, 15] |
| Attention | incremental with KV cache (the reference itself); causal by construction; scale 0.5 (exact shift) |
| MLP | ReLU |
| Reference decode | temperature 0.5 sampling (Python RNG). **Hardware demo uses greedy argmax** (lowest index on ties). |
| Checkpoint | `reference/checkpoints/microgpt_14fb038_seed42.json`; state_dict sha256 `1761e472e0c66d17274ded4fbd62a6512df9a7f86f8d09a1202375cc0f422e8a`. Training is bit-reproducible on this machine (2 identical runs) but **not guaranteed across OSes**, so always use this file. |
| Held-out set | post-shuffle `docs[1000:]` minus names in `docs[:1000]`: 30,869 names, 219,838 predicted tokens |

Forward pass per token (§4.2):
`x0=wte[tok]+wpe[pos]; x1=rms(x0); a=rms(x1); q,k,v=W·a; cache k,v; per head: s=0.5·q·K, p=softmax(s), o=p·V;`
`x2=Wo·o+x1; m=rms(x2); h=relu(W1·m); x3=W2·h+x2; logits=Wlm·x3`

## 3. Hardware and toolchain

| Item | Status |
|---|---|
| Board / part | **PYNQ-Z2 (XC7Z020)** [user-stated 2026-10-16] (Q1 answered). Kria KV260 (XCK26) remains a possible major-project upgrade |
| Host | [OPEN] Q2. Assumed: on-chip ARM (PYNQ/Python or bare-metal C) |
| Vivado / Vitis version | **2025.x chosen by user** (exact minor, 2025.1 or 2025.2, [OPEN]; record after install). Install must be the full **Vitis** product (it includes Vivado + Vitis HLS). **Risk:** PYNQ v3.1 (PYNQ-Z2 image `pynq_z2_v3.1.0`) is built with 2024.1 tools and its docs list 2024.1 as the matching version [SRC: pynq.readthedocs.io v3.1 changelog / SD-card docs]; whether a 2025.x-built .bit/.hwh loads on that image is [OPEN] and is exactly what the vadd test checks. PYNQ-Z2 board files must be installed separately in Vivado [SRC] |
| Tool host requirements | AMD UG1742 (2025.2) lists **32 GB RAM minimum (64 GB recommended)** for embedded + HLS, 35 GB disk (Vitis Embedded) to 200 GB (full), and Windows 10 Pro/Enterprise 22H2 or Windows 11 Enterprise 23H2/24H2/25H2 [SRC]. This machine has 13.8 GB RAM and Windows 11 **Home**, i.e. outside the supported list [MEAS]. Whether small-design HLS/Vivado runs here is [OPEN]; use the command-line flow (no IDE) to lower memory use [PROP] |
| HLS flow | Vitis Unified IDE or `v++ -c --mode hls` + `vitis-run` (standalone `vitis_hls` GUI deprecated) [SRC] |
| PL clock target | 100 MHz initially [PROP] |
| Dev machine | Windows 11; Python 3.14.5 venv at `.venv` (numpy 2.5.3) |

Device facts [SRC]:
- **XC7Z010:** 17.6k LUT, 60 BRAM36, 80 DSP.
- **XC7Z020:** 53.2k LUT, 140 BRAM36, 220 DSP.
- **XCK26:** 117k LUT, 144 BRAM36, 64 URAM, 1,248 DSP.
- **DSP:** DSP48E1 18×25, 48-bit accumulator.

## 4. Architecture decisions (with rationale)

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D1 | All weights, KV cache and activations **on-chip** | Total state is < 20 KB FP32 vs ≥ 2.1 Mb BRAM [DER] | PROP |
| D2 | **Single reusable mat-vec engine** + SFU + fixed sequencer, not per-layer engines | Batch-1 autoregressive ops are sequentially dependent; dedicated engines would idle [DER] | PROP |
| D3 | Start with **P = 1** MAC lane; P is a template parameter | Simplest to verify; P becomes the major-project sweep axis | PROP |
| D4 | **KV cache is the baseline**; full recompute is an ablation only | The reference is incremental; recompute costs 7.5× the MACs [DER] | PROP |
| D5 | One HLS source **templated on data type** (FP32 build and 16-bit fixed build) | FP32 build = integration vehicle; fixed build = numerical baseline | PROP |
| D6 | Explicit rounding (round-to-nearest) + saturation at every requantization point; wide accumulators | `ap_fixed` defaults are AP_TRN/AP_WRAP [SRC]; truncation bias is 27× larger [MEAS] | PROP |
| D7 | Weights loaded at runtime (`LOAD` via m_axi), not baked into the bitstream | Swap checkpoints and quantizations without re-synthesis | PROP |
| D8 | Greedy decoding on device; sampling/loss on host in FP64 | Deterministic, bit-exact verification | PROP (Q6) |
| D9 | Enforce one engine instance (`#pragma HLS allocation function … limit=1` or single call site) | HLS may duplicate the engine otherwise [SRC] | PROP |

## 5. Open questions

- **Q1** board/part: **ANSWERED**, PYNQ-Z2
- **Q2** ARM host allowed? (PYNQ-Z2 has one; confirm with examiner)
- **Q3** tool version + OS: Vivado candidate; exact version and OS [OPEN]
- **Q4** examiner's required demo/metrics (power?)
- **Q5** team size, hours, deadlines: 2 people, deadline **2026-11-21**; "12 hours" needs clarifying (per person per week?) [OPEN]
- **Q6** greedy decoding OK?
- **Q7** model size frozen? (Proposed answer: yes, 4,192 parameters, no retraining.)

**Scope under the deadline [user-accepted 2026-10-16: "focus on FP32"]:** 2026-11-21 leaves ~5 weeks vs the plan's 12–22 [EST].
Build the **FP32** datapath only (P = 1), STEP mode, greedy demo. Defer 16-bit fixed (and the bit-accurate emulator, §7 formats),
GENERATE mode, P = 4 and power to the major project. FP32 acceptance = csim ≡ cosim ≡ board, and within tolerance of NumPy FP32.
Fallback if the board flow slips: report csim/cosim results plus the HLS synthesis report, and say clearly that nothing
was measured on the board.
- **[OPEN]** σ_seed of held-out NLL. Needed for the ΔNLL acceptance threshold: train ≥ 5 seeds, ~25 min CPU total.
- **[OPEN]** Whether `hls::rsqrt` / `hls::exp` fixed-point accuracy is adequate, or a LUT is needed (Stage 5).

## 6. Interfaces and tensor shapes (proposed core `microgpt_core`)

- **AXI4-Lite registers:** `ctrl` (ap_start/done/idle), `mode` {LOAD, RESET, STEP, GENERATE}, `token` (u8 < 27),
  `pos` (u8 < 16), `argmax` (out), `cycles` (out), `error` (out), buffer addresses for weights/logits/debug.
- **m_axi:** weight image (packed per manifest), logits[27], debug taps (debug build only).
- **Internal buffers:** `K[16][16]`, `V[16][16]`, `x1/x2/x3/a/m[16]`, `q[16]`, `s/p[4][16]`, `o[16]`, `h[64]`, `logits[27]`.
- **Per-token cost:** 3,504 + 32·T MACs (T = pos+1); 4T exp, 4 recip, 3 rsqrt [DER].

## 7. Numerical formats and acceptance criteria

**Proposed 16-bit formats** (from measured ranges; freeze at the Stage 5 exit):

| Tensor | Format |
|---|---|
| weights | `ap_fixed<16,2>` |
| x0 | `<16,2>` |
| rmsnorm outputs | `<16,3>` + saturation |
| q, k, v | `<16,3>` |
| scores | `<16,4>` |
| probs | `ap_ufixed<16,1>` |
| o | `<16,3>` |
| Wo·o | `<16,2>` |
| x2 | `<16,3>` |
| fc1 | `<16,3>` |
| fc2 out | `<16,4>` |
| x3 | `<16,4>` |
| logits | `<16,6>` |
| MAC accumulator | `<40,12>` |

**Acceptance (framework fixed; numbers set at the Stage 5 exit):**
1. Emulator (L3) ≡ HLS csim ≡ cosim ≡ board: **bit-exact** (fixed build). csim ≡ cosim ≡ board for the FP32 build.
2. Per-tap error ≤ k LSB (k from the emulator plus margin).
3. Held-out argmax agreement ≥ emulator value, with **no** disagreement where the FP64 top-2 margin exceeds the max
   logit error; |ΔNLL| ≤ 0.1·σ_seed.

**Measured so far** (fake-quant emulation, not HLS) [MEAS]:

| Configuration | NLL / ΔNLL | Argmax agreement | Max \|Δlogit\| |
|---|---|---|---|
| FP64 | NLL 2.367208 | — | — |
| FP32 | — | 100% | 3.5e-6 |
| W16A16 round+sat | ΔNLL +4e-6 | 99.976% (53 near-ties, all with margin < 8.8e-4) | 2.5e-3 |
| W8 weights | ΔNLL −4.8e-4 | 99.05% | — |
| W4 weights | ΔNLL +0.036 | 77.3% | — |

Full table: report §9.7 and `reference/analysis/results_14fb038_seed42.json`.

**Greedy reference outputs** (FP64): BOS → `anan`; prefixes `a`→`anan`, `b`→`bina`, `ka`→`karian`, `mar`→`marian`, `zo`→`zona`.

## 8. Milestones and status

| Stage | Description | Status |
|---|---|---|
| 1 | Confirm model / HW / requirements; hello-world IP on board | **In progress**: model confirmed; HW pending Q1–Q3 |
| 2 | Python reference | **Mostly done**: FP64/FP32 NumPy model verified vs upstream (1.8e-15). TODO: bit-accurate emulator, σ_seed. |
| 3 | Export weights / layouts / test vectors | **Done for FP32** (2026-10-16): `export/fp32/`; 16-bit export deferred |
| 4 | Mat-vec HLS kernel | Not started |
| 5 | Supporting operators + frozen formats | Not started |
| 6 | Full forward pass in HLS | Not started |
| 7 | Host integration + on-board demo | Not started |
| 8 | Measurement | Not started |
| 9 | Baseline documentation + major-project proposal | Not started |

**Stage 3 FP32 export results [MEAS]** (`reference/export/export_fp32.py`, output `export/fp32/`):
- `weights_fp32.bin`: 16,768 bytes = 4,192 float32, sha256 `0ed67a9dbbb0c85b2fc6ab0f3d2ef664a4789457b175f6ae87f836be551cc9ed`;
  tensor order, offsets and per-tensor SHA-256 in `manifest.json`.
- 14 test cases in `export/fp32/vectors/` (lengths 1, 2, 3, 4, 7, 8, 16 positions, the five demo prefixes, a 16-position ramp,
  BOS×16). Per-position record = 475 floats (layout in `vectors.json`), FP32 and FP64 taps plus FP32 argmax.
- Checks passed before writing: FP64 incremental (KV-cache) step vs upstream probe logits 2.7e-15; greedy FP64 and FP32 strings
  equal the reference (`anan`, `bina`, `karian`, `marian`, `zona`); `.bin` round trip gives bit-identical FP32 logits;
  re-running the script reproduces identical files. Max |FP32 − FP64| logit over the test cases: 9.0e-7.
- Caveats: the dataset has no 1-character names, so the length-1 case uses a synthetic `a`. The "27 tokens × 16 positions single-step"
  test from §9.4 is not generated. The unit tests for masking, invalid inputs and reset are not generated either.

**Next actions:**
1. **User:** install Vivado/Vitis 2025.x (needs an AMD account; Claude cannot download it) on a machine that meets or approaches
   the RAM requirement; record the exact version in §3. Install the PYNQ-Z2 board files in Vivado.
2. Get the vector-add IP running on the PYNQ-Z2 (Stage 1 exit). Files prepared 2026-10-16, **all untested**:
   `hls/vadd/` (vadd.cpp, vadd_tb.cpp, run_hls.tcl), `vivado/vadd_pynqz2/build.tcl`, `host/vadd/vadd_pynq.py`.
   Only the testbench logic was checked (plain g++, pragmas ignored: 6 cases pass); that is not an HLS csim.
3. Write the FP32 `matvec` HLS kernel + testbench reading `export/fp32/` (Stage 4), then the remaining operators and the full STEP core.
4. Clarify Q2, Q4, Q5 (hours), Q6 with the supervisor.

## 9. Reproducible commands (PowerShell, from repo root)

```powershell
# clone (private repo; the pinned upstream gist is a submodule at 14fb038)
git clone --recursive https://github.com/Yarish652/microgpt-fpga.git

# environment (once)
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# reproduce the checkpoint from upstream (~4.5 min); compare its state_dict sha256 with §2
.venv\Scripts\python -I reference\analysis\run_upstream.py reference\upstream\microgpt_gist\microgpt.py reference\data\names.txt reference\runs\run_a

# numerical analysis on the frozen checkpoint (~30 s)
.venv\Scripts\python -I reference\analysis\analyze_checkpoint.py reference\checkpoints\microgpt_14fb038_seed42.json reference\analysis\results_14fb038_seed42.json
```

The upstream source is a git clone of the gist, pinned at `14fb038`:
`git -C reference\upstream\microgpt_gist checkout 14fb038`.
```powershell
# FP32 weight image + test vectors (seconds; asserts the checks listed in §8 before writing)
.venv\Scripts\python -I reference\export\export_fp32.py reference\checkpoints\microgpt_14fb038_seed42.json export\fp32
```

HLS/Vivado build and benchmark commands will be added here in Stages 4–8. The benchmark procedure is report §11.

## 10. Repository layout

```
PROJECT_CONTEXT.md        <- this file (update every session)
CLAUDE.md                 <- instructions for AI-assisted sessions
requirements.txt
docs/01_feasibility_and_architecture_plan.md
reference/upstream/microgpt_gist/   pinned upstream gist (own .git)
reference/data/names.txt            pinned dataset
reference/checkpoints/              frozen checkpoint(s)
reference/analysis/                 run_upstream.py, analyze_checkpoint.py, results JSON
reference/export/                   export_fp32.py (Stage 3)
export/fp32/                        weights_fp32.bin, manifest.json, vectors/ (inputs for HLS testbenches)
hls/vadd/                           Stage 1 hello-world HLS source, testbench, run_hls.tcl
vivado/vadd_pynqz2/                 build.tcl (block design + bitstream for PYNQ-Z2)
host/vadd/                          vadd_pynq.py (runs on the board)
reference/runs/                     scratch training runs (reproducible, not canonical)
```

## 11. Key references

- microGPT gist and blog (karpathy.github.io/2026/02/12/microgpt)
- AMD DS190 (Zynq-7000), DS987 (K26), UG973 (supported devices), UG1399 (Vitis HLS: ap_fixed, hls_math,
  allocation), UG1448 (pipeline II)
- PYNQ boards page
- Softermax (arXiv:2103.09301), I-BERT (2101.01321), online softmax (1805.02867), FTRANS (2007.08563),
  Lu et al. (2009.08605)

Full links: report References section.
