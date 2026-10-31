# microgpt-fpga

Read @PROJECT_CONTEXT.md at the start of every session; it holds the verified model spec, decisions, open
questions and status. The full rationale is in docs/01_feasibility_and_architecture_plan.md.

Working rules for this project:
- The golden reference is upstream gist rev `14fb038` + `reference/checkpoints/microgpt_14fb038_seed42.json`.
  Never retrain silently or switch revisions; any new checkpoint gets its own file, SHA-256 and context entry.
- Label claims [SRC]/[MEAS]/[DER]/[EST]/[PROP]/[OPEN]. Never report an estimate as a measurement, or a speedup
  that was not measured against a named baseline.
- Baseline before optimization: no INT8/INT4 or MAC-array work until Stage 7 (on-board baseline) is done.
- Fixed-point code must specify rounding and overflow modes explicitly (`ap_fixed` defaults are AP_TRN/AP_WRAP).
- Mismatches between emulator, HLS csim, co-sim and board are bugs (zero tolerance); differences from FP64 are
  numerical error judged by the acceptance criteria.
- Run Python with `.venv\Scripts\python -I`.
- At the end of a session, update PROJECT_CONTEXT.md (status, decisions, open questions, results).
