# microGPT on FPGA: Feasibility Diagnosis and Architecture Plan

**Version:** 0.1 (2026-10-09). **Status:** model verified and measured; board and toolchain not yet selected.
**Companion files:** [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) (persistent state, updated every session) ·
[analysis scripts](../reference/analysis/) · [measured results](../reference/analysis/results_14fb038_seed42.json)

Each claim carries an evidence label:

| Label | Meaning |
|---|---|
| **[SRC]** | Verified against a primary source (code, datasheet, user guide). Link given. |
| **[MEAS]** | Measured in this repository. Command in §15 / PROJECT_CONTEXT.md. |
| **[DER]** | Derived arithmetically from SRC/MEAS facts. Derivation shown. |
| **[EST]** | Estimate. Must be replaced by a measurement before it is reported. |
| **[PROP]** | Proposed design decision. Not implemented. |
| **[OPEN]** | Unresolved. Listed in §3. |

---

## 1. Executive feasibility assessment

**Verdict: feasible.** Confidence is high that the model fits and that 16-bit arithmetic is sufficient.
Confidence in the schedule is medium. The schedule risk comes from board access and toolchain bring-up,
not from the model.

1. **The provisional spec is internally consistent and matches the current upstream code exactly.**
   Gist revision `14fb038` (2026-02-16) defines a 1-layer, 16-wide, 4-head, 16-context model with
   exactly **4,192 parameters**. The upstream code prints this count when run [SRC][MEAS].
   *However*, gist revisions before 2026-02-12 10:43 PST define a **different model**: `block_size=8`
   (4,064 parameters), squared ReLU, and different init and optimizer settings [SRC]. Pin the revision.
2. **The checkpoint is reproducible on one machine.** Two independent training runs here produced
   bit-identical weights [MEAS]. Bit-exactness across operating systems is not guaranteed, because Python's
   `math.exp`/`math.log` come from the platform C library [EST]. The project therefore freezes one checkpoint
   file and its SHA-256 instead of relying on retraining.
3. **Everything fits on-chip with large margin.** FP32 weights take 16.4 KiB, and weights + KV cache +
   activations take under 20 KB [DER]. The smallest common Zynq (XC7Z010) has 60 × 36 Kb block RAMs
   (2.1 Mb) [SRC]. External DRAM is not needed for model data.
4. **Compute is tiny and purely matrix-vector.** Each token needs 3,504 weight MACs plus 32·T attention
   MACs, where T = 1…16 is the number of cached positions. That is 3,536–4,016 MACs per token [DER].
   Each weight is read exactly once per token (arithmetic intensity: 1 MAC per weight read). Parallelism is
   therefore limited by **on-chip memory ports**, not by DSP count.
5. **Numerics are benign at 16 bits.** Evidence comes from 219,838 held-out next-token predictions:
   - **FP32 vs FP64:** argmax agreement is 100% and the largest logit error is 3.5×10⁻⁶ [MEAS].
   - **16-bit fixed point** (weights and every activation, round-to-nearest, saturating): argmax agreement is
     **99.976%**. All 53 disagreements are near-ties, where the FP64 top-1/top-2 logit margin is below
     9×10⁻⁴ [MEAS, preliminary emulation, not bit-accurate to HLS].
   - **INT8 weights:** 99.1% agreement [MEAS, preliminary].
   - **INT4 weights:** 77% agreement and +1.5% NLL [MEAS, preliminary].

   INT16 is a low-risk baseline. INT8/INT4 is a genuine research question for the major project.
6. **Do not expect a speedup.** The kernel is a few thousand cycles per token, roughly 40–50 µs at
   100 MHz with one MAC lane [EST]. An optimized C loop on the board's own ARM core will plausibly be in
   the same range, and host-call overhead will likely dominate end-to-end latency [EST].
   The minor project's claim should be: *correct, verified, measured baseline with a reproducible method*.
7. **KV cache is not optional.** The reference implementation itself is incremental: it caches K and V and
   processes one token at a time, even during training [SRC]. Full-context recomputation would be the
   *alternative*, costing about 7.5× more MACs for a 16-token generation [DER].
8. **Recommended architecture (provisional until the board is chosen)** [PROP]:
   - one reusable, time-multiplexed matrix-vector engine (1 MAC lane initially, parameterized to P lanes)
   - a small special-function unit for rsqrt, exp, reciprocal and max
   - a fixed sequencer
   - all weights, the KV cache and activations in on-chip RAM
   - a Zynq ARM host using AXI4-Lite for control and a one-time AXI-master weight load
   - one HLS C++ source templated on data type, so that the FP32 and 16-bit fixed-point datapaths share code.
9. **The largest risks are integration risks, not model risks:**
   - board availability
   - Vivado / Vitis / PYNQ version alignment
   - the first HLS → bitstream → board round-trip.

   Mitigate these with a "hello-world" HLS IP running on the chosen board within the first 2–3 weeks,
   in parallel with the software work.

---

## 2. Confirmed facts, assumptions and missing information

### 2.1 Confirmed facts: model and data

| Fact | Evidence |
|---|---|
| Source: `karpathy/8627fe009c40f57531cb18360106ce95`, revision `14fb038816c7…` (2026-02-16 08:30 PST), file SHA-256 `d47d88c2…ebccee` | [SRC] [gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), cloned at `reference/upstream/microgpt_gist` |
| Blog / walkthrough: karpathy.github.io/2026/02/12/microgpt | [SRC] [blog](https://karpathy.github.io/2026/02/12/microgpt/) |
| 4,192 parameters; vocab 27 (26 letters + BOS id 26); n_embd 16; n_head 4; head_dim 4; n_layer 1; block_size 16 | [SRC] code lines 75–90; [MEAS] `num params: 4192` |
| RMSNorm **without learnable gain**, ε = 1e-5 inside the root; applied 3×: after embedding, pre-attention, pre-MLP. **No final norm before lm_head.** | [SRC] lines 103–106, 112, 117, 137, 143 |
| Learned absolute position embedding `wpe` (16×16), added to token embedding | [SRC] lines 81, 109–111 |
| No biases anywhere; ReLU (not GeLU, not squared ReLU); lm_head **not** tied to wte | [SRC] lines 81–88, 139 |
| Attention scale 1/√head_dim = 1/√4 = **0.5 exactly** | [SRC] line 129 |
| Causality is implicit: K/V are appended to a cache and each step attends only to cached positions. No mask exists. | [SRC] lines 121–131 |
| Inference: temperature 0.5 sampling with Python `random.choices`; stops at BOS or 16 tokens | [SRC] lines 187–199 |
| Training: `random.seed(42)`, 1,000 steps × 1 document, Adam (lr 0.01, β 0.85/0.99), linear LR decay | [SRC] lines 12, 147–182 |
| Dataset: makemore `names.txt` @ commit `988aa59`, SHA-256 `0a30b555…8cc4d`; 32,033 names (29,494 unique), a–z only, length 2–15 | [SRC] line 17; [MEAS] |
| Longest name has 15 chars, so BOS + 15 chars = 16 input positions. `block_size = 16` covers every name. | [MEAS] |
| Training sees only `docs[0:1000]`. 30,869 never-seen, de-duplicated names remain as a natural held-out set. | [DER] from code; [MEAS] |
| Training is bit-reproducible on this machine (Win 11, CPython 3.14.5): 2 runs gave identical weights. Final training loss 2.65 for the step-1000 document. | [MEAS] |
| The NumPy re-implementation matches upstream logits to 1.8×10⁻¹⁵ (FP64) | [MEAS] |

### 2.2 Confirmed facts: hardware and tools

| Fact | Evidence |
|---|---|
| XC7Z010: 17,600 LUTs, 35,200 FFs, 60 × 36 Kb BRAM (2.1 Mb), 80 DSP48E1. XC7Z020: 53,200 LUTs, 106,400 FFs, 140 BRAM (4.9 Mb), 220 DSP. XC7Z007S: 14,400 LUTs, 50 BRAM, 66 DSP. Cortex-A9 667 MHz at -1 speed grade. | [SRC] DS190 v1.11.1, Table 1 |
| 7-series DSP: 18×25 signed multiply, 48-bit accumulator, 25-bit pre-adder. BRAM: 36 Kb true dual-port, up to 72 bits wide, splittable into 2 × 18 Kb. | [SRC] DS190 p.2 |
| Kria K26 (XCK26): 117,120 LUTs, 234,240 FFs, 144 BRAM (5.1 Mb), 64 UltraRAM, 1,248 DSP (27×18); 4× Cortex-A53 at 1,333 MHz; 4 GB DDR4 | [SRC] DS987 v1.2 Table 5 and p.2, 5 |
| The free (no-license) Vivado ML Standard edition covers XC7Z007S/010/012S/014S/015/020/030, Artix-7 up to XC7A200T, and Zynq US+ including XCZU1–ZU7 parts. Kria SOMs are "included as part of the ML Standard Edition". | [SRC] [UG973 Supported Devices](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Devices), via search summary. **Re-check for the exact version you install.** |
| The standalone `vitis_hls` GUI is deprecated. The current flow is the Vitis Unified IDE or `v++ -c --mode hls` with a config file, plus `vitis-run` for csim/cosim. | [SRC] [UG1399 migration](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Migrating-from-Vitis-HLS-to-the-Vitis-Unified-IDE), [UG1702 v++ mode HLS](https://docs.amd.com/r/2025.1-English/ug1702-vitis-accelerated-reference/v-Mode-HLS) |
| `ap_fixed` defaults: quantization **AP_TRN** (truncate toward −∞), overflow **AP_WRAP** (wrap-around). Rounding and saturation must be requested explicitly. | [SRC] [UG1399 fixed-point types](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types) |
| `hls_math.h` has fixed-point versions of sqrt, rsqrt, recip and exp, with limits `ap_fixed<W,I>`: I ≤ 33, W−I ≤ 32. When `hls_math` is used, C-sim and C/RTL co-sim results are identical, but they differ from the C standard library. | [SRC] [UG1399 2022.2 fixed-point math](https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/Fixed-Point-Math-Functions), [Verification and Math Functions](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Verification-and-Math-Functions) |
| `#pragma HLS allocation function instances=f limit=1` forces all calls to a non-inlined function onto one RTL instance | [SRC] [UG1399 pragma allocation](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-allocation) |
| A loop-carried dependency through an operation of latency L limits a pipeline to II ≥ L (for example, FP add accumulation) | [SRC] [UG1448 pipeline scheduling](https://docs.amd.com/r/en-US/ug1448-hls-guidance/Pipeline-Scheduling) |
| PYNQ v3.1 (released 2025-10-06) has SD images for PYNQ-Z1, PYNQ-Z2, ZCU104 (v3.1.1), AUP-ZU3 and others. The KV260 uses Ubuntu 22.04 + Kria-PYNQ; its v3.1 image is listed as "coming soon". | [SRC] [pynq.io/boards](https://pynq.io/boards.html), [v3.1 release](https://discuss.pynq.io/t/pynq-v3-1-carlisle-release/8797) |

### 2.3 Assumptions (replace with facts as they become known)

| # | Assumption | If wrong… |
|---|---|---|
| A1 | The board will be a **Zynq-class SoC** (ARM + FPGA) supported by free Vivado | A pure FPGA (Artix-7) needs a MicroBlaze or UART host. That adds 2–4 weeks of integration and makes host I/O dominate latency. |
| A2 | Team of 2–3 students, ~8–12 h/week each, first HLS project | Effort ranges in §10 scale accordingly |
| A3 | Initial PL clock target 100 MHz | Higher clocks are possible, especially on UltraScale+, but are not needed for a baseline |
| A4 | Greedy (argmax) decoding is acceptable for the on-board demo; temperature sampling, if wanted, runs on the host from returned logits | If on-device sampling is required, add an RNG and an inverse-CDF unit, and reproducibility gets harder |
| A5 | The model stays at the verified 4,192-parameter configuration for the minor project | A larger model changes memory and parallelism conclusions (§7, §8) |
| A6 | The free toolchain is installable on a lab or personal machine with enough disk and RAM | See R2 / R15 in §12 |

### 2.4 Missing information (blocking for architecture freeze)

The exact board and part number, the Vivado/Vitis version, the host OS for the tools, the course's
demonstration and report requirements, deadlines, and the availability of a power-measurement instrument.
See §3.

---

## 3. Critical questions that must be answered before implementation

Only questions that change the architecture are listed.

| # | Question | Why it matters / what changes |
|---|---|---|
| **Q1** | **Which board, and exact FPGA part, can you get, and when?** | Determines the host interface, clock target, DSP type (DSP48E1 vs E2) and maximum parallelism for the major project. **Recommendation:** PYNQ-Z2 (XC7Z020) for the lowest integration risk, since it has an official PYNQ image. Kria KV260 if you want headroom for the major project (5.7× the DSPs and 2.2× the LUTs of XC7Z020, plus 64 UltraRAM blocks; see §8.5). A Zybo Z7-10/20 also works, but with less PYNQ support. Avoid a bare Artix-7 board unless that is all you can get. |
| **Q2** | **Is the on-chip ARM allowed to act as the host (Linux + PYNQ/Python, or bare-metal C)?** | With ARM: AXI4-Lite + AXI master, ~1 week of integration. Without: UART + soft-processor or RTL UART. That costs more, and transfer time dominates per-token latency. |
| **Q3** | **Which Vivado/Vitis version will you use, on which OS?** | PYNQ images expect overlays built with a matching tool version and board files. HLS pragma syntax and the GUI differ across versions (§2.2). Fix one version for the whole project and record it. |
| **Q4** | **What does the examiner require you to demonstrate and report?** (for example: live generation, accuracy, latency, resources, power) | Decides whether power instrumentation and an in-kernel generation loop are in scope |
| **Q5** | **Team size, weekly hours, and minor- and major-project deadlines?** | Calibrates §10. Decides whether both FP32 and INT16 reach the board in the minor project. |
| **Q6** | **Is greedy decoding acceptable for the hardware demo?** (A4) | Keeps the device deterministic and verification bit-exact |
| **Q7** | **Is the model size frozen for the minor project?** (A5) | Any change requires retraining, re-pinning and re-measuring |

---

## 4. Verified model specification and tensor-shape inventory

### 4.1 Pinned reference

```
gist      : https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
revision  : 14fb038816c7aae0bb9342c2dbf1a51dd134a5ff   (2026-02-16 08:30:37 -0800)
file      : microgpt.py   sha256 d47d88c2fd432c8ebdc1048beab7f7eb64ea7e0e664e11b812d72a6d95ebccee
dataset   : https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt
            sha256 0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d
checkpoint: reference/checkpoints/microgpt_14fb038_seed42.json
            state_dict sha256 (json.dumps of the dict) 1761e472e0c66d17274ded4fbd62a6512df9a7f86f8d09a1202375cc0f422e8a
```

### 4.2 Exact forward pass for one token at position `pos` (FP64 in the reference)

`rmsnorm(x) = x · (mean(x²) + 1e-5)^(-1/2)`, with no learnable gain.

```
x0     = wte[tok] + wpe[pos]                         (16)
x1     = rmsnorm(x0)                                 (16)  <- residual source of the attention block
a_in   = rmsnorm(x1)                                 (16)  <- the second, back-to-back rmsnorm is real (see 4.6)
q,k,v  = Wq·a_in, Wk·a_in, Wv·a_in                   (16 each)
K[pos] = k ; V[pos] = v                              (cache: 16 positions × 16)
for h in 0..3 (slice h*4 .. h*4+3):
    s[t] = 0.5 · (q_h · K[t]_h)        t = 0..pos    (pos+1)
    p    = softmax(s)    (max-subtracted)            (pos+1)
    o_h  = Σ_t p[t] · V[t]_h                         (4)
o      = concat(o_0..o_3)                            (16)
x2     = Wo·o + x1                                   (16)
m_in   = rmsnorm(x2)                                 (16)
hdn    = relu(W1·m_in)                               (64)
x3     = W2·hdn + x2                                 (16)
logits = Wlm·x3                                      (27)  <- no final norm
```

### 4.3 Parameter inventory

Layout: row-major `[out][in]`. Upstream `linear` computes `y[o] = Σ_i W[o][i]·x[i]`, the same as PyTorch `nn.Linear`.

| Tensor | Shape (out × in) | Count | Max \|w\| [MEAS] | Max row L2 [MEAS] |
|---|---|---:|---:|---:|
| `wte` (token embedding) | 27 × 16 | 432 | 0.690 | 1.072 |
| `wpe` (position embedding) | 16 × 16 | 256 | 0.726 | 0.874 |
| `layer0.attn_wq` | 16 × 16 | 256 | 0.479 | 0.898 |
| `layer0.attn_wk` | 16 × 16 | 256 | 0.551 | 0.791 |
| `layer0.attn_wv` | 16 × 16 | 256 | 0.338 | 0.654 |
| `layer0.attn_wo` | 16 × 16 | 256 | 0.328 | 0.621 |
| `layer0.mlp_fc1` | 64 × 16 | 1,024 | 0.642 | 0.864 |
| `layer0.mlp_fc2` | 16 × 64 | 1,024 | 0.450 | 1.283 |
| `lm_head` | 27 × 16 | 432 | 1.333 | 2.546 |
| **Total** | | **4,192** | | |

### 4.4 Intermediate-tensor inventory

T = number of cached positions (1…16). Ranges come from teacher-forced evaluation over 30,869 held-out
names (219,838 positions) [MEAS]. Static bounds are [DER]: rmsnorm output satisfies Σx² ≤ 16, so |xᵢ| ≤ 4,
and |W·x|ᵢ ≤ ‖Wᵢ‖₂·4.

| Tensor | Shape | Max \|value\| measured | Static bound | Proposed 16-bit format [PROP] |
|---|---|---:|---:|---|
| emb sum x0 | 16 | 1.39 | max\|wte\| + max\|wpe\| = 1.42 | `ap_fixed<16,2>` |
| rmsnorm outputs (x1, a_in, m_in) | 16 | 3.44 | 4.0 | `ap_fixed<16,3>` + **saturate** |
| q / k / v | 16 | 2.83 / 2.38 / 2.31 | 3.59 / 3.16 / 2.62 | `ap_fixed<16,3>` |
| attention scores s (scaled) | T per head | 6.89 | — | `ap_fixed<16,4>` |
| attention probs p | T per head | 1.0 (min nonzero 1.1e-6) | [0,1] | `ap_ufixed<16,1>` |
| head outputs o | 16 | 2.28 | max\|v\| | `ap_fixed<16,3>` |
| Wo·o | 16 | 1.10 | — | `ap_fixed<16,2>` |
| residual x2 | 16 | 3.44 | — | `ap_fixed<16,3>` (or `<16,4>` for margin) |
| fc1 pre-ReLU / post-ReLU | 64 | 2.75 / 2.44 | 3.45 | `ap_fixed<16,3>` |
| W2·hdn | 16 | 4.87 | — | `ap_fixed<16,4>` |
| residual x3 | 16 | 5.42 | — | `ap_fixed<16,4>` |
| logits | 27 | 14.4 | — | `ap_fixed<16,6>` |
| KV cache | 2 × 16 × 16 | (as k, v) | | same as k / v |

These formats are what the 99.976%-agreement emulation used. They are inputs to Stage 5, not final decisions.
The static bounds for the residual stream are loose, so x2 and x3 rely on measured range plus saturation.
Re-measure the ranges whenever the checkpoint changes.

### 4.5 Data layout for export [PROP]

Each tensor is a contiguous row-major array in the order listed in 4.3. A JSON manifest records name, shape,
offset, dtype, format (`W`, `I`) and SHA-256. Test vectors use the same convention:
`tokens.bin`, `pos.bin`, and one file per tap per position.

### 4.6 Discrepancies and clarifications vs. the provisional spec

| Provisional spec | Verified | Action |
|---|---|---|
| "Approximately 4,192 parameters" | **Exactly 4,192** at rev `14fb038`; revisions before `cf7331b` (2026-02-12) have 4,064 | Pin revision + hash |
| "RMSNorm" | RMSNorm **without gain**, 3 applications, ε = 1e-5, back-to-back at the start of the layer, no final norm | Implement all three. The back-to-back pair differs from a single norm by up to 4.7×10⁻⁴ [MEAS]. That exceeds 16-bit resolution, so it is *not* removable without a measured accuracy check. |
| Positional representation unspecified | Learned absolute table `wpe` [16×16] | Position ≥ 16 is invalid: the hardware must reject it |
| "Optional KV cache" | The reference **is** a KV-cache implementation | Baseline includes the KV cache. Full recompute is an ablation only. |
| "Causal self-attention" | Causality is implicit (no mask). Equivalent to a causal mask in the batched form. | The test suite checks causality explicitly (§9.8) |
| Autoregressive generation | Reference samples at T = 0.5 with Python's RNG | Device: greedy. Host: optional sampling from logits (Q6). |
| Vocabulary 27 | 26 letters + BOS (id 26). BOS also acts as end-of-sequence. | — |
| "High-precision reference" | Python floats are IEEE FP64 | FP64 is the golden reference. FP32 is a second reference. |

---

## 5. Proposed system architecture and block diagram

> **This is the PROPOSED architecture.** The *implemented* architecture is documented separately once
> Stage 6 exits; it may use fewer features (for example P = 1, no debug port). Q1–Q3 can change the host side.

```
┌─────────────────────────── Processing System (ARM; Linux+PYNQ, or bare-metal C) ───────────────────────────┐
│ host app: tokenizer · weight/manifest loader · decode loop (greedy / host sampling) · teacher-forced eval   │
│           · timing (cycle counter readout + wall clock) · logging                                           │
│ DDR: weight image (~8-17 KB) · logits buffer · debug-tap buffer (debug build)                                 │
└──────────────┬────────────────────────────────────────────────────────────────────┬────────────────────────┘
               │ AXI4-Lite (s_axilite): cmd, mode, token, pos, status, cycles       │ AXI4 master (m_axi, HP port)
┌──────────────▼──────────────────────── Programmable Logic: microgpt_core (one HLS top) ───▼────────────────┐
│  ┌─────────────────────────┐                                                                              │
│  │ Control / sequencer     │  fixed op schedule:  EMB → NORM → NORM → MV(Wq,Wk,Wv) → KV-WRITE →            │
│  │ (FSM, mode, pos, reset) │  ATTN(scores→softmax→PV) ×4 heads → MV(Wo) → ADD → NORM → MV(W1) → RELU →     │
│  └────────────┬────────────┘  MV(W2) → ADD → MV(Wlm) → ARGMAX                                               │
│               │                                                                                             │
│  ┌────────────▼───┐    ┌──────────────────────────┐    ┌───────────────────────┐   ┌──────────────────────┐ │
│  │ Weight RAMs    │───►│ Matrix-vector engine      │◄───│ Activation buffers    │◄─►│ Special-function unit│ │
│  │ one per tensor │    │ P MAC lanes, adder tree,  │───►│ x1,x2,x3,a_in,q,o,hdn │   │ sum-sq, rsqrt, exp,  │ │
│  │ (BRAM/LUTRAM)  │    │ wide accumulator,         │    │ scores/probs, logits  │   │ recip, max, ReLU,    │ │
│  └───────▲────────┘    │ requantize (round+sat)    │    └──────────┬────────────┘   │ residual add         │ │
│          │ one-time    └──────────────────────────┘    ┌───────────▼───────────┐   └──────────────────────┘ │
│          │ load (m_axi)                                │ KV cache K,V [16][16] │                            │
│  weight loader                                         └───────────────────────┘                            │
│  cycle counters (per token, per op class) ──► s_axilite     debug taps ──► m_axi (debug build only)         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data flow (per token, STEP mode)

1. The host writes `token` and `pos`, sets `start`. Weights are already resident from the one-time LOAD command.
2. The sequencer runs the fixed schedule from §4.2. Every matrix product goes through the **same** engine,
   with the weight base address and dimensions as arguments.
3. k and v are written into `K[pos]`, `V[pos]`. Attention reads `K[0..pos]`, `V[0..pos]` only.
4. Logits (27) are written to the output buffer. `argmax` and the cycle count go to AXI-Lite registers.
5. The host reads logits (STEP mode), or the kernel feeds argmax back as the next token (GENERATE mode).

### Modes [PROP]

| Mode | Purpose |
|---|---|
| `LOAD` | Copy the weight image from DDR into on-chip RAM (once per checkpoint) |
| `RESET` | Clear sequence state (pos counter, cache-valid length) |
| `STEP` | One token in, logits out. Used for teacher-forced evaluation, host sampling and per-token debugging. |
| `GENERATE` | Greedy loop on-chip from BOS or a prefix, up to 16 tokens. Gives kernel-only steady-state latency without host overhead. *Stretch goal for the minor project.* |

---

## 6. Component-level hardware/software responsibility matrix

### 6.1 Summary

| # | Component | HW / SW | Inputs → Outputs | Dims | Data type (FP32 build / fixed build) | On-chip memory |
|---|---|---|---|---|---|---|
| 1 | Host interface & control | SW + HW regs | cmd, token, pos → status, argmax, cycles | scalars | uint32 | ~10 regs |
| 2 | Input token handling | SW (tokenize) + HW (bounds check) | char → id; id, pos | 0–26, 0–15 | uint8 / uint8 | — |
| 3 | Weight storage | HW | weight image → per-tensor RAMs | 4,192 | f32 / `ap_fixed<16,2>` (per tensor) | 16.4 KiB / 8.2 KiB |
| 4 | Activation buffers | HW | — | ≈251 values | f32 / `ap_fixed<16,·>` | ≈1.0 KB / 0.5 KB |
| 5 | Matrix-vector engine | HW | W[r×c], x[c] → y[r] | r ≤ 64, c ≤ 64 | f32 / 16×16→wide acc | — |
| 6 | MAC datapath | HW | w, x, acc → acc | P lanes | f32 / acc `ap_fixed<40,12>` | — |
| 7 | Q/K/V projections | HW (uses 5) | a_in → q,k,v | 3 × 16×16 | as 5 | — |
| 8 | Causal self-attention | HW | q, K[0..pos], V[0..pos] → o | 4 heads × T × 4 | scores `<16,4>` | KV 2 KB / 1 KB |
| 9 | Softmax | HW (SFU) | s[T] → p[T] | T ≤ 16 | f32 / exp in, `ap_ufixed<16,1>` out | 16 values |
| 10 | RMSNorm + residual | HW (SFU + adders) | x → x·rsqrt(ms+ε); a + b | 16 | f32 / sum-sq wide, rsqrt `ap_ufixed` | — |
| 11 | FFN + ReLU | HW (uses 5) | m_in → hdn → W2·hdn | 16→64→16 | as 5 | 64 values |
| 12 | Output projection & next token | HW (logits, argmax) + SW (sampling, loss) | x3 → logits[27] → id | 27 | logits `<16,6>` | 27 values |
| 13 | Autoregressive state / KV cache | HW (cache, pos) + SW (loop in STEP mode) | k, v, pos | 16 × 16 × 2 | as k / v | 2 KB / 1 KB |
| 14 | HW-SW communication | both | AXI-Lite, AXI master | — | — | DDR buffers |
| 15 | Verification & instrumentation | both | taps, counters | per stage | as produced | debug build only |

### 6.2 Details per component

Fields per component: **Computation**, **Dependencies**, **Verification**, **Risks**.

1. **Host interface & control.**
   - *Computation:* the HLS block-level protocol (`ap_ctrl_hs` via `s_axilite`) plus scalar registers.
     The host polls `ap_done`.
   - *Dependencies:* board, PYNQ or bare-metal BSP.
   - *Verification:* hello-world IP first; register read-back test.
   - *Risks:* address-map mismatch; PYNQ overlay and tool-version mismatch (R2).
2. **Input token handling.**
   - *Computation:* the host tokenizes (a–z → 0–25, BOS = 26). Hardware checks `token < 27` and `pos < 16`, and
     sets an error flag otherwise.
   - *Verification:* invalid-input tests (§9.8).
   - *Risks:* off-by-one at pos 15, or the BOS id confused with index 0.
3. **Weight storage.**
   - *Computation:* the one-time `LOAD` copies a packed image over `m_axi` into one RAM per tensor. Baking weights
     into the bitstream as ROM is possible, but needs re-synthesis per checkpoint, so it is poor for quantization
     experiments.
   - *Verification:* read-back checksum of the RAM contents vs the manifest SHA.
   - *Risks:* layout transposition errors (row- vs column-major); per-tensor format mismatch.
4. **Activation buffers.**
   - *Computation:* small arrays, ping-pong where an op reads and writes the same logical vector.
   - *Verification:* debug taps.
   - *Risks:* HLS partitioning choices inflate LUT/FF use; read-after-write hazards in a shared buffer.
5. **Matrix-vector engine.**
   - *Computation:* `y[r] = Σ_c W[r][c]·x[c]` for runtime `(base, rows, cols)`. Called by every linear layer, so it
     must exist **once** in RTL. Enforce this with `#pragma HLS allocation function … limit=1`, or a single call site
     inside a loop over an op table (§2.2).
   - *Verification:* stand-alone csim against NumPy for each of the 7 shapes, plus random matrices.
   - *Risks:* duplicated instances (R4); variable trip counts hide latency in reports (use `LOOP_TRIPCOUNT`).
6. **MAC datapath.**
   - *Computation:* a 16×16-bit product (fits one DSP48E1 18×25 [SRC]) into a 40-bit accumulator.
     `ap_fixed<16,2>` × `ap_fixed<16,4>` is exactly `ap_fixed<32,6>`. Summing up to 64 terms adds 6 integer bits,
     giving `ap_fixed<38,12>`, so `ap_fixed<40,12>` cannot overflow [DER]. Requantize **once** per output with
     round-to-nearest + saturation.
   - *FP32 build:* fmul + fadd; break the accumulation dependency (§8.1).
   - *Verification:* the bit-accurate emulator (§9.1 L3).
   - *Risks:* default AP_TRN/AP_WRAP (R6, R7).
7. **Q/K/V projections.**
   - *Computation:* three 16×16 mat-vecs (768 MACs) on the same input.
   - *Verification:* taps q, k, v.
   - *Risks:* none beyond 5.
8. **Causal self-attention.**
   - *Computation:* for each head, T dot products of length 4, ×0.5 (an arithmetic shift: exact, no
     rounding [DER]), then softmax, then a T-term weighted sum per output dimension. 32·T MACs. Reads only cache
     rows `0..pos`.
   - *Verification:* taps for scores, probs and o; causality and stale-cache tests.
   - *Risks:* length-4 dot products underuse a wide engine. Map all 4 heads in parallel with a 16-lane multiply
     and a segmented (4 × 4) adder tree when P ≥ 4 [PROP].
9. **Softmax.**
   - *Computation:* max → exp(s − max) (inputs ≤ 0, outputs in (0,1]) → sum (≤ 16) → one reciprocal → T multiplies.
     Fixed-point options: `hls::exp` on `ap_fixed`, a LUT + linear interpolation, or base-2 (`2^(x·log₂e)`, split into
     integer shift + fractional LUT) as in Softermax [SRC: arXiv 2103.09301].
   - *Verification:* tap p. Check Σp = 1 ± tolerance and compare against FP64 softmax on the same quantized scores.
   - *Risks:* underflow of tiny probabilities (min observed 1.1e-6 < 2⁻¹⁵) is harmless here [MEAS: included in
     W16A16 result]; reciprocal precision.
10. **RMSNorm + residual.**
    - *Computation:* Σx² in a wide accumulator. For an `ap_fixed<16,4>` input, Σx² < 16·8² = 1,024, so 11 integer
      bits suffice [DER]. Then the mean as a shift (÷16 exact), add ε, `rsqrt`, and 16 multiplies. ε = 1e-5 is below 16-bit resolution of many formats, so guard rsqrt(0) explicitly.
      Residual add with saturation.
    - *Verification:* taps x1, a_in, x2, m_in, x3.
    - *Risks:* rsqrt accuracy dominates every downstream value. Test `hls::rsqrt` against a LUT + Newton-Raphson
      alternative.
11. **FFN + ReLU.**
    - *Computation:* 16→64 mat-vec, ReLU (sign test), 64→16 mat-vec (the only mat-vec with cols = 64).
    - *Verification:* taps hdn and W2·hdn.
    - *Risks:* fc2 output has the largest pre-residual range (4.87) [MEAS].
12. **Output projection & next token.**
    - *Computation:* 27×16 mat-vec, then argmax with ties broken by the **lowest index** (matches `numpy.argmax`).
      Softmax over logits is **not** needed on-device for greedy decoding. Loss and sampling run on the host in FP64.
    - *Verification:* logits tap; argmax agreement with margin analysis (§9.5).
    - *Risks:* tie-break mismatch.
13. **Autoregressive state / KV cache.**
    - *Computation:* `pos` register and cache-valid length. `RESET` clears length (contents need not be zeroed if
      reads are bounded by length).
    - *Verification:* stale-cache test (run A then B, compare with B alone).
    - *Risks:* off-by-one at pos = 15.
14. **HW-SW communication.**
    - *Computation:* AXI-Lite for scalars; `m_axi` for weight load, logits and debug.
    - *Verification:* loop-back tests, then the full pipeline.
    - *Risks:* cache coherency of DDR buffers. PYNQ `allocate` handles this. Bare-metal needs explicit cache
      flush/invalidate.
15. **Verification & instrumentation.**
    - *Computation:* a free-running cycle counter latched at start/done, plus per-op-class counters (matvec, attn,
      SFU). The debug build writes every tap for one selected position to DDR.
    - *Verification:* compare counter totals with the HLS schedule estimate and wall-clock.
    - *Risks:* instrumentation perturbs the design. Report resources for both release and debug builds.

---

## 7. Memory and arithmetic estimates

### 7.1 Parameter and state storage [DER]

| Item | Values | FP32 | INT16 / 16-bit fixed | INT8 | INT4 |
|---|---:|---:|---:|---:|---:|
| Weights | 4,192 | 16,768 B | 8,384 B | 4,192 B | 2,096 B |
| KV cache (2 × 16 × 16) | 512 | 2,048 B | 1,024 B | 512 B | 256 B |
| Activations (all live vectors, all heads) | ≈251 | ≈1.0 KB | ≈0.5 KB | ≈0.25 KB | — |
| Quantization scales (per tensor / per row) | 9 / 222 | — | (shift amounts) | 36 B / 888 B as FP32 | same |
| **Total state** | | **≈19.8 KB** | **≈9.9 KB** | **≈5 KB** | |

**What decides on-chip fit:** port count and partitioning, not capacity.

- *Capacity.* FP32 state is ≈ 158 kbit. That is ≈ 7% of the XC7Z010's 2.1 Mb [DER from SRC].
- *Granularity.* With one RAM per tensor (needed so the engine can switch tensors without address arithmetic
  across RAMs), FP32 needs about 11 BRAM18 halves: wte, wpe, lm_head, Wq, Wk, Wv, Wo at ≤ 512×32 each, and
  fc1, fc2 at 1024×32 each. 16-bit needs about 9 BRAM18 [DER, assuming 512×36 / 1K×18 aspect ratios; confirm in
  UG473 and the HLS report]. HLS may also put the 256-entry tensors in LUTRAM.
- *Parallel access.* A P-lane engine needs P weights per cycle. A 7-series BRAM36 is "up to 72 bits wide"
  [SRC: DS190]: one 72-bit read port in simple-dual-port mode, or two 36-bit ports. Either way that is
  ≈ 64 data bits, or 4 × 16-bit weights, per BRAM36 per cycle [DER].
  - At P = 16 and 16-bit (256 bits/cycle), each weight tensor read in parallel must be split across
    **≥ 4 BRAM36**, however small it is.
  - For the 7 projection matrices that is ≈ 28 BRAM36: 47% of an XC7Z010 or 20% of an XC7Z020, for only 7 KB
    of data [DER]. Less if small tensors move to LUTRAM or registers.

  **BRAM consumption grows with parallelism, not model size.** This is a central quantity for the major project.

### 7.2 Arithmetic per token [DER from §4.2]

| Operation | MACs / ops | Note |
|---|---:|---|
| Q, K, V projections | 768 | 3 × 16 × 16 |
| Attention scores | 16·T | 4 heads × T × 4 |
| Attention × V | 16·T | 4 heads × 4 dims × T |
| Output projection Wo | 256 | |
| FFN fc1 / fc2 | 1,024 / 1,024 | |
| lm_head | 432 | |
| **Weight MACs subtotal** | **3,504** | = 4,192 − 432 (wte) − 256 (wpe): every non-embedding weight is used exactly once per token |
| **Total MACs** | **3,504 + 32·T** | **3,536 (T = 1) … 4,016 (T = 16)** |
| RMSNorm | 3 × (16 sq-acc + 16 mul) + 3 rsqrt | |
| Softmax | 4·T exp, 4 reciprocals, 4·T mul, 4·(T−1) compares | No softmax over the 27 logits on-device (greedy) |
| Adds | 16 (embedding) + 32 (residuals) | |
| ReLU / argmax | 64 compares / 26 compares | |

- **Per 16-token sequence (KV cache):** Σ(3,504 + 32T) for T = 1…16 gives **60,416 MACs**.
- **Without KV cache:** a 16-step generation that recomputes the prefix but only the last logits costs
  3,072·136 + 432·16 + 16·Σn(n+1) = **450,816 MACs (7.5×)**. Computing all positions' logits gives 502,656 (8.3×).
- **FLOPs:** about 2 × MACs, so ≈ 7–8 kFLOP per token.

### 7.3 Memory traffic per token [DER]

All 3,504 non-embedding weights plus 2 embedding rows (32 values) are read once per token: 3,536 weight reads.
That is 14.1 KB per token at FP32, or 7.1 KB at 16-bit. Reads of KV cache rows `0..pos`: 2 × 16·T values.
Because arithmetic intensity is 1 MAC per weight read, a P-lane engine needs P weight reads per cycle (§7.1).

### 7.4 Latency model [EST: must be replaced by HLS schedule reports, then by on-board measurements]

```
cycles/token ≈ Σ_matvec rows·⌈cols/P⌉·II   +   attention (≈ 2·T·⌈16/P⌉)   +   SFU (3 rsqrt + 4 recip + 4T exp, partly pipelined)
             + per-operation pipeline fill/drain (~ten ops × pipeline depth)   +   control
```

| Config (fixed-point, II = 1) | Mat-vec cycles | Attention (T = 16) | Total incl. overhead [EST] | At 100 MHz [EST] |
|---|---:|---:|---:|---:|
| P = 1 | 3,504 | 512 | ≈ 4.2k–5k | ≈ 42–50 µs |
| P = 4 (lanes across input dim) | 876 | 128 | ≈ 1.3k–1.8k | ≈ 13–18 µs |
| P = 16 | 219 (rows × ⌈cols/16⌉) | 32 | ≈ 0.5k–0.9k (SFU/control dominate) | ≈ 5–9 µs |

- **FP32 build caveat.** A naive FP accumulation loop runs at II = L_fadd (several cycles at 100 MHz; read it
  from the HLS report). That multiplies the mat-vec term by L unless rows are interleaved or partial sums are
  used [SRC: UG1448].
- **Amdahl warning.** Beyond P ≈ 8 the non-MAC work dominates. Quantifying that crossover is a good
  major-project result.
- **Host overhead.** Measure a no-op kernel call to get the floor. It is likely comparable to or larger than the
  P = 1 kernel time [EST].

---

## 8. Comparison of candidate compute architectures

Resource entries are **structural** (counts of multipliers or memories) or **qualitative**. LUT/FF numbers must
come from post-implementation reports, never from estimates.

### 8.1 Compute organization

| | A1: single time-multiplexed MAC (P = 1) | A2: partially parallel MAC array (P = 4…16) | A3: highly parallel / dedicated engines / systolic |
|---|---|---|---|
| Latency/token | ≈ 4–5k cycles [EST] | ≈ 0.5–1.8k cycles [EST] | Bounded below by the sequential dependency chain (NORM → QKV → ATTN → Wo → NORM → FFN → lm_head) and SFU latency. Little gain over A2 at P = 16. |
| Throughput | 1 token per latency (batch 1, autoregressive: the next token depends on argmax) | same | Only higher if several *independent* sequences are interleaved (evaluation, not generation) |
| DSP | 1 (fixed 16×16) or ≈ 2–5 (FP32 fmul + fadd [EST, read from report]) | P (fixed) | 256 for a 16×16 array. **Exceeds XC7Z020's 220** [DER from SRC]. Per-layer engines: ≥ 7 engines, mostly idle. |
| BRAM | ≈ 9–11 BRAM18 (capacity-bound) | Grows with P (port-bound, §7.1) | Weights must sit in registers/LUTRAM: tens of kbit of FFs for INT8. Infeasible for FP32 on small parts. |
| LUT/FF | Lowest | Moderate (adder tree, partitioned memories, muxing) | Highest |
| Memory bandwidth | 1 weight/cycle | P weights/cycle | All weights/cycle |
| Complexity | Lowest. Easiest to verify. | Medium (partitioning, segmented tree for heads) | High |
| Numerics | Identical across A1–A3 if the summation order is fixed. Adder trees change order vs a sequential sum: bit-exact for fixed point, not for FP32. | | |
| Fit for 4,192 params | **Best for the minor project** | **Best first major-project axis** | Poor: a batch-1 mat-vec has no weight reuse, so systolic arrays (built for mat-mat reuse) idle |

**Single reusable engine vs multiple dedicated engines.** In batch-1 autoregressive inference, consecutive
operators are data-dependent, so dedicated per-layer engines would mostly wait for each other. A single
reusable engine plus SFU has the same critical path with a fraction of the area [DER]. Dedicated engines only pay
off when pipelining independent work:

- prefill of a long prompt (not relevant at ≤ 16 tokens)
- interleaving many independent sequences, for example to raise evaluation throughput.

**Recommendation:** A1 for the minor project, written so that P is a template parameter (A2) [PROP].

### 8.2 Arithmetic

| | FP32 (HLS float) | 16-bit fixed (`ap_fixed`, per-tensor formats) | INT8 / INT4 + scales |
|---|---|---|---|
| Accuracy vs FP64 | Indistinguishable: 100% argmax, ≤ 3.5e-6 logit error [MEAS] | 99.976% argmax, ≤ 2.5e-3 logit error, disagreements only at near-ties [MEAS, emulated] | INT8 weights: 99.1%. INT4 weights: 77%, +1.5% NLL [MEAS, weights only, per-tensor PTQ] |
| DSP per MAC | Several (fmul + fadd; exact count from report) | 1 (16×16 in an 18×25 DSP48E1) | ≤ 1. Packing 2 INT8 MACs per DSP is a known technique on wider DSPs [verify for DSP48E1]; or use LUT multipliers. |
| Accumulation | Loop-carried FP add: II = L unless restructured | II = 1, exact (wide accumulator) | II = 1 |
| Nonlinear ops | `hls::exp`/`hls::rsqrt` float: simple | Fixed-point `hls_math` (width limits §2.2) or LUTs: needs format design | Usually kept at ≥ 16-bit (mixed precision) |
| Verification | Tolerance-based vs NumPy FP32 (order and libm differ) | Bit-exact vs emulator possible | Bit-exact vs emulator possible |
| Role | Integration vehicle; "FP32-compatible baseline" | **Minor-project numerical baseline** | Major project |

### 8.3 Memory placement

| | On-chip (BRAM/LUTRAM) | External DDR streaming |
|---|---|---|
| Latency | Single-cycle reads | Burst setup and DDR latency per weight block. Adds latency comparable to or greater than the whole on-chip kernel [EST]. |
| Bandwidth need | P weights/cycle from local RAM | 3,536 weights per token from DDR |
| Resources | ≈ 9–11 BRAM18 | AXI master + FIFOs (LUT/FF, some BRAM) |
| Complexity | Low | Medium–high |
| **Verdict** | **Use.** Everything fits with > 90% BRAM headroom. | Only meaningful as a major-project scaling study (larger models) |

### 8.4 Attention: full recompute vs KV cache

| | Full recompute each step | Incremental + KV cache |
|---|---|---|
| MACs for 16-token generation | 450,816 | 60,416 (7.5× fewer) [DER] |
| Memory | No cache, but every past token must be re-embedded | 512 values (1–2 KB) |
| Faithfulness to reference | Mathematically equal; the reference does **not** do this | **Matches reference structure** |
| **Verdict** | Ablation only | **Baseline** |

### 8.5 Board choice impact

| Candidate (all free-Vivado-supported per §2.2) | Effect on the design |
|---|---|
| **PYNQ-Z2** (XC7Z020: 140 BRAM, 220 DSP, 53k LUT) | Recommended for the minor project: official PYNQ image, Python host, plenty of room for P ≤ 16 at 16-bit. 7-series timing: aim for 100 MHz. |
| Zybo Z7-10 / XC7Z010 (60 BRAM, 80 DSP) | Fits the minor project. Tight for the major project's parallel sweeps (BRAM-port bound, §7.1). Less PYNQ support. |
| **Kria KV260** (XCK26: 144 BRAM + 64 URAM, 1,248 DSP, 117k LUT, 4× A53) | Most headroom for the major project (larger models, wide INT8 arrays). Higher clocks are likely. Kria-PYNQ setup on Ubuntu (§2.2). DSP48E2 27×18. |
| Artix-7 boards (Basys 3 / Arty) | No hard CPU: MicroBlaze or UART host, slow I/O, more integration. Use only if nothing else is available. |

---

## 9. Numerical correctness and golden-reference test plan

### 9.1 Reference ladder

Every level is compared with the level above it.

| Level | Implementation | Compared against | Expected relation |
|---|---|---|---|
| L0 | Upstream `microgpt.py` (FP64 scalar autograd) | — | Definition |
| L1 | NumPy FP64 golden model (`analyze_checkpoint.py`) | L0 | ≤ 1e-12 (achieved: 1.8e-15) [MEAS] |
| L2 | NumPy FP32 model | L1 | Measured error: the FP32 noise floor |
| L3 | **Bit-accurate fixed-point emulator** (NumPy int64, explicit rounding/saturation per §4.4) | L1 | Measured error = *the numerical error being studied* |
| L4 | HLS C-simulation (the same C++ that is synthesized) | L3 (fixed) / L2 (FP32) | **Fixed: bit-exact.** FP32: tolerance (order, `hls_math` vs libm). |
| L5 | C/RTL co-simulation | L4 | **Bit-exact** [SRC: UG1399, with `hls_math`] |
| L6 | On-board execution | L5 | **Bit-exact** |

**Principle.** Any mismatch in L3 ↔ L4 ↔ L5 ↔ L6 is a **bug** (zero tolerance). The gap L1 ↔ L3 is
**numerical error** and is judged against acceptance criteria. Keeping these separate is what makes the
hardware results defensible.

### 9.2 Operator implementation guidance [PROP]

- **Rounding and overflow.** Never rely on `ap_fixed` defaults (AP_TRN, AP_WRAP) [SRC].
  - Use AP_RND_CONV or AP_RND plus AP_SAT on every *requantization* point: the end of each dot product, rmsnorm
    outputs, residual adds, probabilities.
  - Keep internal accumulators wide and wrap-free, sized by analysis: 40 bits for a 64-term 16×16 dot product.
  - Measured effect of truncation: mean logit bias −4.7×10⁻⁴, versus +1.7×10⁻⁵ with rounding (27× larger
    bias), with similar agreement [MEAS]. The bias matters more at 8 bits.
- **Attention scale.** ×0.5 is an exact right-shift by 1 (head_dim = 4) [DER]. No multiplier needed.
- **Softmax.** Max-subtract (as the reference does), then exp of non-positive inputs. Sum ≤ T ≤ 16 needs 5
  integer bits. Compute one reciprocal per head, then multiply.
  - Start with `hls::exp` / `hls::recip` on `ap_fixed` (bit-identical csim/cosim).
  - Compare against a LUT/base-2 design later (major project).
  - The online (single-pass) normalizer [SRC: arXiv 1805.02867] is unnecessary at T ≤ 16 but relevant if context grows.
- **RMSNorm.** Σx² of 16 values (exact in a wide accumulator), ÷16 = shift, +ε, `hls::rsqrt`, then
  multiply-and-requantize.
  - Guard ms + ε against underflow to zero in fixed point.
  - Keep both back-to-back norms (§4.6).
- **ReLU:** sign test. **Argmax:** first-maximum tie-break.
- **Embedding add:** exact in `ap_fixed<17,3>` before requantizing to x0's format.

### 9.3 Test vectors and taps

Exported by Stage 3, versioned with the checkpoint hash. Each test vector has:
- inputs: token sequence and positions
- per-position taps:
  - `x0, x1, a_in, q, k, v`
  - `s[h]` (scores per head), `p[h]` (softmax output per head)
  - `o, attn_proj, x2, m_in, hdn_pre, hdn, ffn_out, x3, logits`
  - `argmax`
- each tap in FP64 (L1), FP32 (L2) and fixed-point integers (L3)

On hardware, taps come from the debug build (one selected position per run, written to DDR).

### 9.4 Test catalogue

| Category | Test | Pass condition |
|---|---|---|
| Masking / causality | Teacher-force a sequence, then change token t+1 … 15: logits at ≤ t unchanged | Bit-identical |
| Masking | Kernel never reads `K[j]` for j > pos (assertion in csim; RTL: cache poisoned with sentinel values beyond pos) | No sentinel leakage |
| Sequence length | Lengths 1, 2, 7, 15, 16 (all names + max-length names) | L3 ↔ L6 bit-exact |
| Edge: positions | pos = 0 (T = 1, softmax of a single element = 1.0), pos = 15 | Exact p = 1.0 at T = 1 |
| Edge: tokens | All 27 tokens × 16 positions single-step (432 cases); BOS-only input | Bit-exact |
| Invalid inputs | token ≥ 27, pos ≥ 16, STEP without LOAD | Error flag set; no state corruption |
| Reset / stale state | Run seq A, RESET, run seq B; compare with B on a fresh kernel | Bit-identical |
| Overflow | (a) Saturation counters in the emulator/csim over the full held-out set (expect 0); (b) "stress checkpoint" with weights scaled ×2, ×4 to force saturation | (a) 0 saturations or documented; (b) emulator ↔ HW still bit-exact (proves saturation logic matches) |
| Reproducibility | Same input twice; re-run after reprogramming the bitstream; rebuild from a clean tree | Bit-identical outputs; identical resource report for the same tool version and seed |
| Weight integrity | Read-back of on-chip RAM vs manifest SHA after LOAD | Match |
| Model-level | Teacher-forced held-out evaluation (30,869 names) | Criteria in §9.6 |
| Demo | Greedy generation from BOS and from fixed prefixes (`a`, `b`, `ka`, `mar`, `zo`) | Equals L3 emulator output. FP64 reference gives `anan`, `bina`, `karian`, `marian`, `zona` [MEAS]. |

### 9.5 Metrics (definitions)

For a tap `h` (hardware/emulated) vs `r` (FP64 reference), over all positions and elements:

- **Max abs error** `max|h − r|`. Also report the 99.9th percentile, so a single outlier does not hide behind the max.
- **Normalized error** `‖h − r‖₂ / ‖r‖₂` per vector, then max and mean. For probabilities, use **KL(p_ref ‖ p_hw)**.
  Avoid element-wise relative error near zero.
- **Error in LSBs** of the tap's format (fixed point). Error growth in LSB units from layer to layer shows where
  precision is lost.
- **Next-token agreement** `mean[argmax h = argmax r]` over all teacher-forced positions. Also report **margin-stratified
  disagreements**: for each disagreement, the FP64 top-1/top-2 margin. A disagreement at a margin larger than the
  measured max logit error indicates a **bug**, not rounding.
- **Validation NLL** (below), as ΔNLL with a paired bootstrap 95% CI over documents.
- **Greedy-sequence match rate** from fixed prefixes. Demo metric; very sensitive to near-ties.
- **Latency per generated token**: see §11.

### 9.6 Loss methodology and acceptance criteria

**How to compute loss.**
- *Dataset:* the frozen held-out list. It is defined by the post-shuffle order in the checkpoint
  (`docs[1000:]` minus any names in `docs[:1000]`), 30,869 documents. Store it as a file with its SHA-256.
- *Encoding:* `[BOS] + chars + [BOS]`, truncated to 16 inputs, exactly as upstream.
- *Inference:* teacher forcing. Feed reference tokens (not the hardware's own predictions) so errors do not compound
  through different contexts.
- *Loss:* the **host** dequantizes hardware logits and computes `log_softmax` and NLL in **FP64**.
  - Primary: token-weighted mean NLL.
  - Secondary: document-mean NLL (upstream training uses the per-document mean).
  - Always state which one is reported.
- *Comparison:* report ΔNLL = NLL_hw − NLL_ref with a paired bootstrap CI. Exact equality is **not** expected.

**FP64 reference values** [MEAS]: token-mean NLL **2.36721**, doc-mean 2.36208, top-1 accuracy vs ground
truth 28.2%. The task is high-entropy, so use *agreement with the reference*, not accuracy, as the hardware metric.

**Acceptance criteria.** The framework is fixed now. The numbers are finalized at the Stage 5 exit, once the
formats are frozen.

1. **Integration (zero tolerance):** L3 ≡ L4 ≡ L5 ≡ L6 bit-exact on all test vectors (fixed-point build), and
   L4 ≡ L5 ≡ L6 for the FP32 build.
2. **Operator-level:** per-tap max error ≤ k LSBs of the tap format, where k is the emulator's measured value
   plus margin, recorded in PROJECT_CONTEXT.md.
3. **Model-level:**
   - **(a)** next-token agreement ≥ the emulator's value, with **zero** disagreements above the max-error margin;
   - **(b)** |ΔNLL| small compared with *training noise*. Train K ≥ 5 seeds in FP64, measure σ_seed of held-out
     NLL, and require |ΔNLL| ≤ 0.1·σ_seed. Report the FP32 ΔNLL (≈ 3×10⁻¹⁰ [MEAS]) as the floor.

   This makes "acceptable" an experimentally justified statement rather than an arbitrary threshold.
   σ_seed is **not yet measured** [OPEN]. About 25 minutes of CPU for 5 seeds with the upstream script.

### 9.7 Preliminary numerical results (checkpoint `14fb038_seed42`, 219,838 held-out positions) [MEAS]

| Configuration | Token-mean NLL | ΔNLL vs FP64 | Argmax agreement | # disagree | Max \|Δlogit\| | Max FP64 margin at a disagreement |
|---|---:|---:|---:|---:|---:|---:|
| FP64 (golden) | 2.367208 | — | — | — | — | — |
| FP32 | 2.367208 | ≈ −3e-10 | 100% | 0 | 3.5e-6 | — |
| W16 weights only | 2.367209 | +1e-6 | 100% | 0 | 6.7e-4 | — |
| **W16 A16, round + sat** | 2.367212 | +4e-6 | **99.976%** | 53 | 2.5e-3 | 8.8e-4 |
| W16 A16, truncate + sat | 2.367188 | −2e-5 | 99.972% | 61 | 3.3e-3 | 1.4e-3 |
| W16 A12, round + sat | 2.367303 | +9.5e-5 | 99.46% | 1,179 | 3.2e-2 | 1.5e-2 |
| W8 A16, round + sat | 2.366733 | −4.7e-4 | 99.08% | 2,028 | 0.153 | 3.4e-2 |
| W8 weights only | 2.366730 | −4.8e-4 | 99.05% | — | 0.152 | — |
| W6 weights only | 2.365954 | −1.3e-3 | 94.4% | — | 0.484 | — |
| W4 weights only | 2.403151 | **+3.6e-2** | 77.3% | — | 2.80 | — |

Caveats:
- This is per-tensor symmetric post-training quantization.
- Nonlinear functions run in FP64 and accumulators are exact. It is a *fake-quant emulation*, **not** a model of HLS.
- It uses one checkpoint and one seed.

The slight NLL *decrease* at W8 shows why ΔNLL alone is a weak criterion: agreement and margin analysis are needed too.

---

## 10. Phased implementation roadmap with exit criteria

**Effort assumptions:** 2–3 students at 8–12 h/week, first HLS project, board in hand by the end of Stage 1.
Ranges are calendar weeks. Effort falls toward the low end if someone has prior Vivado/AXI experience, and
rises if the board is late.

| Stage | Prerequisites | Deliverables | Tests / exit criteria | Likely failure modes | Effort | HW / people dependencies |
|---|---|---|---|---|---|---|
| **1. Confirm model, HW, requirements** | This document | Answers to Q1–Q7; board + part; pinned tool version installed; **hello-world HLS IP (vector add) running on the board** from host code | Hello-world returns correct results on the board; versions recorded in PROJECT_CONTEXT.md | Board delayed; tool/PYNQ version mismatch; install problems | 1–3 wk | **Board needed for exit**; everyone |
| **2. Python reference** (largely done) | Pinned source | FP64/FP32 NumPy models (done); **bit-accurate fixed-point emulator** (L3); seed-variance study (K ≥ 5) | L1 ≡ L0 (done); emulator's agreement/NLL consistent with the §9.7 W16A16 fake-quant result for the same formats (the emulator then becomes the definition HLS must match bit-exactly); σ_seed measured | Emulator semantics ≠ `ap_fixed` semantics (rounding/saturation corner cases) | 1–2 wk | None (parallel with Stage 1) |
| **3. Export weights, layouts, test vectors** | 2 | Weight images (FP32, 16-bit) + manifest; test-vector sets + taps (§9.3); frozen held-out list | Round-trip check: load export → NumPy → identical logits; SHA-256 recorded | Row/column-major mix-ups; endianness | 0.5–1 wk | None |
| **4. Mat-vec kernel** | Tools (Stage 1 install), 3 | Templated `matvec<T, P>` HLS kernel + testbench; csim + cosim for all 7 shapes; synthesis report | Fixed: bit-exact vs emulator. FP32: within tolerance vs NumPy FP32. II and latency recorded. One instance only (allocation check). | FP accumulation II; duplicated instances; testbench file-path issues | 2–3 wk | Tools only (no board) |
| **5. Supporting operators + formats** | 4 | rmsnorm, softmax/exp/recip, residual, ReLU, argmax kernels; **frozen numerical formats** and acceptance numbers in PROJECT_CONTEXT.md | Each operator bit-exact vs emulator over all held-out positions' taps; saturation counts reported | `hls_math` fixed-point width limits; rsqrt accuracy; emulator divergence | 2–4 wk | Tools only |
| **6. Full forward pass + attention** | 5 | `microgpt_core` top (STEP mode, KV cache, LOAD/RESET); csim + cosim; post-synthesis resources | All §9.4 tests in csim; a cosim subset bit-exact; resources < 50% of target device | Cache indexing bugs; cosim runtime (keep vectors short); interface pragmas | 2–3 wk | Tools only |
| **7. Host integration + on-board demo** | 6 + Stage 1 hello-world | Vivado block design + bitstream; host driver (PYNQ or C); greedy demo; teacher-forced eval on board | L6 ≡ L5 bit-exact on all test vectors; demo strings = emulator; held-out eval complete on board | Timing closure; AXI/cache coherency; overlay load errors | 2–4 wk | **Board** |
| **8. Measurement** | 7 | Latency (kernel/E2E/first/steady), resources (post-route), timing, power (estimate or measured), numerical report vs §9.6 | Procedure in §11 followed; raw data + scripts committed | Host-timer noise; unclear definitions; measuring a debug build by mistake | 1–2 wk | Board; power meter optional |
| **9. Document baseline + research questions** | 8 | Minor report: architecture *as implemented*, results, limitations; major-project proposal (§14) | Reviewer can reproduce the key numbers from the repo | Late writing; claims beyond evidence | 1–2 wk | Everyone |

**Critical path:** 1 → (4 → 5 → 6) → 7 → 8 → 9 ≈ 12–22 weeks.

**Parallel tracks:**

```
Track A (software) : Stage 2 ──► Stage 3 ──────────────────────────────┐
Track B (platform) : Stage 1: install tools ─► board hello-world ─► AXI template ─┤
Track C (HLS)      :            Stage 4 ──► Stage 5 ──► Stage 6 ───────────────────┴─► Stage 7 ─► Stage 8 ─► Stage 9
```

- **Must be sequential:** 4 → 5 → 6 → 7 → 8. Formats must be frozen (end of 5) before full-pipeline verification.
- **Parallelizable:** Stage 2/3 with Stage 1; FP32 and fixed-point variants of the same kernel (one source);
  host-driver development (Track B) with Stages 4–6.

---

## 11. Benchmark and reporting methodology

### 11.1 Metric definitions

| Metric | Definition | How measured |
|---|---|---|
| Kernel-only latency | `ap_start` → `ap_done` for one STEP at position T | PL cycle counter (in the IP), ÷ f_clk. Report per T = 1…16. |
| End-to-end per-token latency | Host writes token → host has logits/argmax | Host monotonic timer (`time.perf_counter_ns` / `clock_gettime(CLOCK_MONOTONIC)`), N ≥ 1,000 repeats |
| First-token latency | From RESET (weights resident) to the first logits, at pos 0 | As above. **Report weight LOAD time separately** (one-time). |
| Steady-state latency | Mean/median per-token latency for positions 1…15 | Both kernel-only and E2E |
| Tokens/s | 1 / steady-state E2E latency (batch 1). Separately: the GENERATE-mode figure (kernel loop, no host). | Derived. State which. |
| Host overhead floor | E2E latency of a no-op kernel call | Same harness |
| Resources | LUT, FF, DSP, BRAM, URAM: **post-place-and-route** `report_utilization` (not HLS estimates) | Vivado reports, archived |
| Timing | Achieved WNS at the target clock | `report_timing_summary` |
| Power | Vivado `report_power` = **estimate** (label it). Measured only with an instrument and an idle baseline subtracted. | Optional |
| Numerical | §9.5 / §9.6 metrics vs FP64 | Host scripts |

### 11.2 Baselines

Name them exactly and report all settings.

- **B1 (fairest):** the same algorithm in C/C++ (FP32), compiled with `-O2`/`-O3`, on the **board's own ARM core**,
  single-threaded. Use the HLS kernel source itself compiled with g++ where possible. Same SoC and power envelope.
- **B2:** the NumPy FP32 model on the host PC (CPU model, NumPy version, thread count).
- **B3:** upstream pure-Python `microgpt.py`. A *functional* reference only. Do **not** present a speedup over it:
  it uses scalar autograd objects and is not a meaningful performance baseline.

### 11.3 Procedure

- Warm up 100 calls; measure ≥ 1,000.
- Report median, p5, p95 and min. Pin CPU frequency/governor where possible.
- Release build only (no debug taps).
- Run the same sequences on every platform; record the git commit and checkpoint SHA.

### 11.4 Mandatory disclosure

Report all of the following:

- board, part and speed grade
- PL clock (target and achieved)
- data type and formats
- HLS/Vivado/Vitis versions, synthesis and implementation strategies, HLS directives
- PYNQ/OS version
- compiler and flags for baselines
- checkpoint SHA, sequence lengths, number of repeats, statistic reported
- whether the timing is kernel-only or end-to-end

### 11.5 If the FPGA is slower

That is a legitimate, informative baseline. Decompose E2E latency into kernel, host overhead and transfer.
Then show:

1. which component dominates
2. how kernel latency scales with P (from the A2 sweep)
3. what model size would make the kernel term dominate.

This turns "slower" into a design rule for the major project.

---

## 12. Risk register

L = likelihood, I = impact (H/M/L).

| ID | Risk | L | I | Early warning | Mitigation |
|---|---|---|---|---|---|
| R1 | Board not available or late | M | H | No board by week 2 | Decide Q1 now. Do Stages 2–6 in csim/cosim (board-free). Borrow a lab board for Stage 7. |
| R2 | Tool / PYNQ / board-file version mismatch | H | H | Overlay fails to load; PS config errors | Fix one Vivado version matching the PYNQ image docs; hello-world in Stage 1; record versions |
| R3 | First-time AXI/driver integration problems | H | M | Hello-world slips | Start with the PYNQ default flow and a template IP; keep the interface minimal (s_axilite + one m_axi) |
| R4 | HLS duplicates the mat-vec engine (area blow-up) or resists sharing | M | M | DSP/LUT count ≫ expected | `allocation … limit=1`; single call site over an op table; inspect the hierarchy in the report |
| R5 | FP32 accumulation II > 1 distorts the "FP32 baseline" | H | L | II = L_fadd in the report | Interleave rows / partial sums; report II explicitly |
| R6 | Silent fixed-point overflow (AP_WRAP default) | M | H | Large sudden errors | Explicit AP_SAT at requantization; saturation counters in csim; stress checkpoint |
| R7 | Truncation bias (AP_TRN default) | H | M | Biased mean logit error | Explicit rounding mode; the emulator mirrors it |
| R8 | rsqrt/exp/recip accuracy or `hls_math` width limits | M | M | Per-tap error grows at the norm/softmax taps | Compare `hls_math` vs LUT+NR; widen internal formats |
| R9 | KV-cache indexing / stale state bugs | M | H | Position-dependent mismatches | §9.4 causality, stale-cache and sentinel tests |
| R10 | Reference drift (different gist revision, retrained weights) | M | H | Param count ≠ 4,192; hash mismatch | Pin revision and SHA; all tools check the checkpoint SHA |
| R11 | Host overhead dominates; speedup misreported | H | M | E2E ≫ kernel | Report kernel-only and E2E separately; no-op call floor |
| R12 | Timing closure fails at the target clock | L–M | M | Negative WNS | Start at 100 MHz (7-series); pipeline the SFU |
| R13 | Scope creep (INT8, parallel arrays before baseline) | H | H | Work on Stage 5+ features before Stage 7 | Scope boundaries in §13; baseline first |
| R14 | Team availability / single point of knowledge | M | H | One person owns all Vivado knowledge | Pair on Stage 1; scripted builds; PROJECT_CONTEXT.md |
| R15 | Toolchain host OS/disk limits (Windows vs Linux flows) | M | M | Install fails; PYNQ docs assume Linux | Check UG973 OS support; a Linux VM/WSL2 for scripting if needed |
| R16 | No way to measure power | M | L | No instrument | Report the Vivado estimate as an estimate; defer measured energy to the major project |
| R17 | Co-simulation too slow for large vector sets | M | L | Hours-long cosim | Cosim a subset; full set in csim + on-board |

---

## 13. Recommended scope boundaries for the minor project

**Minimum viable implementation (must have):**

1. Pinned reference, frozen checkpoint, NumPy FP64/FP32 models and a bit-accurate 16-bit fixed-point emulator.
2. One HLS core (P = 1) implementing the full forward pass with on-chip weights and KV cache in **STEP** mode.
3. At least **one** datapath (FP32 *or* 16-bit fixed) verified on a physical board, bit-exact against co-sim.
   *Target:* both, via the templated source.
4. Host program: greedy generation demo and teacher-forced held-out evaluation on the board.
5. Measurements: kernel-only and E2E latency per position, post-route resources, timing, numerical report (§9.6).
6. Documentation of the architecture *as implemented* and reproducible scripts.

**Stretch (only after 1–6):** GENERATE mode (on-chip greedy loop); both datapaths on board; P = 4 variant;
Vivado power estimate.

**Deferred to the major project:** INT8/INT4, MAC-array sweeps beyond P = 4, softmax/norm approximation
studies, operator fusion, power measurement, larger/retrained models, DDR streaming, batching multiple sequences.

**Out of scope:** on-FPGA training/backprop; on-device temperature sampling; tokenizer in hardware; RTL
hand-coding (HLS only) unless HLS blocks a requirement.

---

## 14. Major-project research extensions

### 14.1 Proposed central question

> *For batch-1 autoregressive Transformer inference on a resource-constrained FPGA, how do arithmetic precision
> (16/8/4-bit) and MAC-array parallelism (P) jointly determine the latency–area–accuracy trade-off? At what point
> does on-chip memory partitioning or non-MAC work (normalization, softmax, control), rather than DSP count, limit
> the benefit of parallelism?*

**Why this is coherent.** The minor project already supplies everything it needs:
- a verified pipeline
- a bit-accurate emulator, to predict accuracy before synthesis
- a parameterized engine
- a measurement harness

Each experiment changes **one** factor while the others are held fixed.

**Experimental design:**
- Factors: precision ∈ {16, 8, 4} (with per-tensor and per-row scales), P ∈ {1, 2, 4, 8, 16}, and optionally model
  width ∈ {16, 32, 64} (retrained, pinned, to test whether the conclusions scale).
- Responses: kernel latency per token, post-route LUT/FF/DSP/BRAM, achieved f_max, argmax agreement, ΔNLL relative
  to σ_seed, and (if instrumented) energy per token.
- Hypotheses to test:
  - **H1.** BRAM usage scales with P, not with parameter count, at this model size (§7.1).
  - **H2.** Latency saturates beyond some P* because of SFU/control (§7.4).
  - **H3.** INT8 needs finer-grained scales (or quantization-aware training) to keep agreement within the 16-bit
    result; INT4 needs both.

### 14.2 Ranked extensions

| Rank | Extension | Research value | Difficulty | Fit for an ECE project | Notes |
|---:|---|---|---|---|---|
| 1 | **Precision × parallelism study** (central question) | High | Medium | High | Combines the next two items under one controlled design |
| 2 | **Quantization-induced loss and prediction changes** (INT8/INT4; per-tensor vs per-row; PTQ vs QAT) | High | Low–Medium | High | Preliminary data: W8 99.1%, W4 77% (§9.7). Emulator first, then hardware. |
| 3 | **MAC-array parallelism and memory partitioning** | High | Medium | High | Tests H1/H2 directly |
| 4 | **Softmax/RMSNorm approximation trade-offs** (`hls_math` vs LUT vs base-2 vs integer-only) | Medium–High | Medium | High | Grounded in Softermax [2103.09301] and I-BERT [2101.01321] |
| 5 | **Operator fusion** (for example, folding RMSNorm scaling into the next mat-vec; fused attention) | Medium | Medium | Medium | Must be checked against the emulator, since fusion changes rounding |
| 6 | **Latency/area/energy trade-offs** | High | Medium–High | Medium | Value depends on having a power instrument |
| 7 | **Buffering/memory reuse; DDR streaming for larger models** | Medium | Medium–High | Medium | Only meaningful with a larger model |
| 8 | **KV-cache benefits** | Low | Low | Low–Medium | The 7.5× MAC reduction is analytic (§8.4) and the cache is already in the baseline. Interesting only as memory-vs-recompute at longer contexts. |

---

## 15. Persistent project-context document

The persistent context lives in **[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md)** at the repository root, so
that it can be updated independently of this report. It contains:

- goal and non-goals
- verified model specification
- hardware and toolchain (pending Q1–Q3)
- architecture decisions with rationale
- open questions
- interfaces and tensor shapes
- numerical formats and acceptance criteria
- milestones and status
- reproducible commands
- references

[CLAUDE.md](../CLAUDE.md) points future assistant sessions to it.

---

## References

**Model and data**
- A. Karpathy, *microgpt.py* (gist), rev. 14fb038 — https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
- A. Karpathy, "microgpt" blog post, 2026-02-12 — https://karpathy.github.io/2026/02/12/microgpt/
- makemore `names.txt` @ 988aa59 — https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt

**AMD devices and tools**
- AMD/Xilinx DS190 v1.11.1, *Zynq-7000 SoC Data Sheet: Overview* (Table 1 PL resources).
- AMD/Xilinx DS987 v1.2, *Kria K26 SOM Data Sheet* (Table 5 PL resources).
- AMD UG973, *Vivado Release Notes, Installation and Licensing*: Supported Devices — https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Devices
- AMD UG1399, *Vitis High-Level Synthesis User Guide*:
  [fixed-point types](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Overview-of-Arbitrary-Precision-Fixed-Point-Data-Types),
  [HLS math library](https://docs.amd.com/r/en-US/ug1399-vitis-hls/HLS-Math-Library),
  [fixed-point math functions (2022.2)](https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/Fixed-Point-Math-Functions),
  [pragma allocation](https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-allocation),
  [migration to Unified IDE](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Migrating-from-Vitis-HLS-to-the-Vitis-Unified-IDE)
- AMD UG1448, *HLS guidance*: pipeline scheduling — https://docs.amd.com/r/en-US/ug1448-hls-guidance/Pipeline-Scheduling
- AMD UG1702, *v++ mode HLS* — https://docs.amd.com/r/2025.1-English/ug1702-vitis-accelerated-reference/v-Mode-HLS
- PYNQ boards and v3.1 release — https://pynq.io/boards.html, https://discuss.pynq.io/t/pynq-v3-1-carlisle-release/8797

**Prior work**
- Milakov & Gimelshein, *Online normalizer calculation for softmax*, arXiv:1805.02867 (2018).
- Stevens et al., *Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers*, DAC 2021, arXiv:2103.09301.
- Kim et al., *I-BERT: Integer-only BERT Quantization*, ICML 2021, arXiv:2101.01321.
- Li et al., *FTRANS: Energy-Efficient Acceleration of Transformers using FPGA*, ISLPED 2020, arXiv:2007.08563.
- Lu et al., *Hardware Accelerator for Multi-Head Attention and Position-Wise Feed-Forward in the Transformer*, SOCC 2020, arXiv:2009.08605.
