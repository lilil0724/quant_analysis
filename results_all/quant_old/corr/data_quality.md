# Data quality

- Raw rows: 913
- Raw HQQ rows: 887
- Complete HQQ rows: 887
- Prepared factor cells: 864
- Duplicate prepared cells: 16
- Baseline contexts: 27
- Baseline-matched cells: 864
- Unmatched cells: 0
- Max duplicate time range (s): 100.41

## Baseline sources

- `raw_4999`: 26
- `manual_corrected_run`: 1

## Excluded fixed config

- hqq_quant_zero=True, hqq_quant_scale=True, hqq_offload_meta=False,
  hqq_compute_dtype=float16, hqq_exclude=head, image_size=224,
  batch_size=64, host=server-3090.

Time results are exploratory: repeated same-seed runs were collapsed by median.
