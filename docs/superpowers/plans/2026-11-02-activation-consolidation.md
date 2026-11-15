# Activation consolidation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** One activation_quantization folder owns both supported serial analyses.
**Architecture:** Shared CLI/storage/pairing rules; retain independent W6A4 and W/A
grid engines and immutable historical outputs.
**Tech Stack:** Python opencode_env, pandas, numpy, matplotlib, standard library.
**Spec:** ../specs/2026-11-02-activation-consolidation-design.md

## Global constraints

- Work in the main checkout, preserve existing uncommitted work.
- No rewriting historical inputs, outputs, reports, or manifests; receipt for relocation.
- Input/run/report IDs reject overwrites and path traversal.
- Use the explicit Windows opencode_env Python executable in the sandbox.
- Temporary functional checks live outside the repo and are removed at finish.

## Tasks

1. [x] Write temporary failing checks for manifest-driven serial selection,
   serial mismatch, W/A read-only verification, safe IDs, and unified acquisition.
2. [x] Consolidate CLI and snapshot import, pairing rules, and both analysis engines;
   expose --verify-run for both formats, preserve numerical calculations.
3. [x] Move historical reports with per-file hashes/receipt and remove duplicate
   ownership; update module README, root index and operational navigation.
4. [x] Run actual 20008 input through the unified command, compare all six tables
   with the pre-refactor completed run, verify hashes/formulas and refusal to overwrite.
   Check 20007 contracts outside the repo; do not claim an unavailable real-data rerun.
5. [x] Obtain an independent code review, resolve material findings, clean temporary
   files and record verification evidence below.

## Review focus

Manifest paths escaping input/run roots; unsupported serial or explicit mismatch;
read-only verification of both old and new receipt schemas; missing 20007 inputs;
historical figure/report bytes and relative links after relocation.

## Execution evidence

- Original six unified-interface checks failed before implementation and passed afterward.
- All 13 temporary functional checks passed in the sandbox using the explicit opencode_env Python.
- 20007: temporary 144-cell schema fixture completed snapshot import, execution,
  reference pairing, heatmap, read-only verification and overwrite rejection.
  This is not a real serial-20007 inference or original-data rerun.
- 20008: actual 3,024-cell input ran through the unified entry point;
  new run `20261102_140733_229106_wa_s20008`. All six table CSVs are byte-identical
  to completed pre-refactor run `20261102_135946_317954_s20008_brief`.
- Both actual 20008 runs pass read-only verification. Existing input/result/report
  files: 32 preserved; historical report/image files: six relocated with equal hashes.
- Review found receipt-chain loss on repeated imports and a destination boundary
  gap; both reproduced by regression checks before fixes, then passed with the full suite.
- Independent final review: no remaining material findings. No downloads or server jobs.
- Audit records: `activation_quantization/migrations/20261102_consolidation/`.
  Temporary test code/fixtures removed after verification; no commits or index changes.
