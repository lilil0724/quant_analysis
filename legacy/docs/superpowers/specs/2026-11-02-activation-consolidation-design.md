# Activation analysis consolidation

User request: combine `w6a4_analysis` and `activation_quantization` so one folder
owns activation quantization organization and analysis.

Use `activation_quantization` as the owning module. `run` selects serial 20007 or
20008 from the input manifest, optionally checked against `--serial`. Keep the
two analysis contracts distinct: 20007 requires its complete corrected baseline
and historical reference; 20008 requires its complete 3,024-cell grid and keeps
unavailable/ambiguous baselines as N/A. Both expose read-only run verification.
`acquire` imports each supported snapshot format without overwriting an input.
Keep all existing 20008 commands working.

Move 20007 reports byte-for-byte with a relocation receipt. Remove the old owning
folder after verifying copies; update current navigation and document old CLI
replacement. Do not modify historical report commands or manifests. The current
checkout has 20008 inputs/results but only 20007 reports: explicitly distinguish
20008 real-data regression from temporary 20007 contract checks.

Approaches: retaining two owning folders plus a wrapper would leave fragmented
data/report ownership; merging both numerical pipelines would blur different
protocol and baseline requirements. A single owner with separate analysis
functions gives the requested folder and retains the established checks.

Use the main checkout per AGENTS.md. No downloads, server jobs, dependency
installation, commits, or changes to original inference data are needed.
