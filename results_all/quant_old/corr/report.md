# Quant factor report

- Prepared cells: 864
- Baseline-matched cells: 864
- Baseline contexts: 27

## Factor contribution ranking (delta R2 ablation)

Bars in `factor_contribution_<metric>.png` are delta R2 from the 
levels (categorical) model: each nbits and group_size level is a 
separate dummy column. This avoids forcing a linear relationship and 
captures non-linear patterns like "low bit collapses, high bit 
plateaus".  Delta R2 = how much the full-model R2 drops when that 
term group is removed.  This is a ranking, not an additive allocation.

### accuracy_ratio
      term_group  full_r2  reduced_r2  delta_r2
           nbits 0.955692    0.842905  0.112788
nbits:group_size 0.955692    0.874885  0.080807
     nbits:model 0.955692    0.943353  0.012339
 nbits:ckpt_kind 0.955692    0.952595  0.003097
   nbits:dataset 0.955692    0.954664  0.001029
         dataset 0.955692    0.955680  0.000012

### params_reduction_pct
     term_group  full_r2  reduced_r2     delta_r2
          nbits      1.0    0.900106 9.989406e-02
    nbits:model      1.0    0.999990 1.013646e-05
          model      1.0    0.999992 7.645941e-06
nbits:ckpt_kind      1.0    0.999999 6.391521e-07
      ckpt_kind      1.0    0.999999 3.914900e-07
  nbits:dataset      1.0    1.000000 7.394052e-08

### time_delta_pct
      term_group  full_r2  reduced_r2  delta_r2
 nbits:ckpt_kind 0.571514    0.519485  0.052029
         dataset 0.571514    0.543242  0.028272
           nbits 0.571514    0.547137  0.024377
nbits:group_size 0.571514    0.551449  0.020065
   nbits:dataset 0.571514    0.557900  0.013614
           model 0.571514    0.560706  0.010808

### vram_reduction_pct
     term_group  full_r2  reduced_r2  delta_r2
    nbits:model  0.92737    0.779876  0.147494
          model  0.92737    0.827511  0.099859
          nbits  0.92737    0.856284  0.071087
nbits:ckpt_kind  0.92737    0.906114  0.021256
      ckpt_kind  0.92737    0.922891  0.004480
     group_size  0.92737    0.925365  0.002005

## Spearman rank correlation supplement

Pearson r measures linear association; Spearman rho measures monotonic 
association (is the rank order preserved?). When rho substantially exceeds 
r, the relationship exists but is non-linear — a common pattern with 
nbits-driven accuracy drop. Spearman is then the more reliable first-pass 
summary of association strength.

Below are median within-context Spearman correlations: each value fixes 
dataset, model, checkpoint kind, and (for nbits) group size, or 
(for group size) nbits.  This controls confounding from task difficulty 
and isolates the monotonic association between the factor and the outcome.

All values are dimensionless (-1 to 1).

### accuracy_ratio
                              scope          factor  correlation
within_summary:group_within_context log2_group_size     -0.80000
within_summary:nbits_within_context       hqq_nbits      0.97619

### params_reduction_pct
                              scope          factor  correlation
within_summary:group_within_context log2_group_size    -1.000000
within_summary:nbits_within_context       hqq_nbits    -0.822473

### time_delta_pct
                              scope          factor  correlation
within_summary:group_within_context log2_group_size    -0.200000
within_summary:nbits_within_context       hqq_nbits    -0.809524

### vram_reduction_pct
                              scope          factor  correlation
within_summary:group_within_context log2_group_size      1.00000
within_summary:nbits_within_context       hqq_nbits     -0.94523

## Limitations

- This is a single-seed complete factor grid. Results are descriptive effect decompositions, not significance tests.
- Time is exploratory because repeated same-seed rows have variable elapsed time.
- Metrics are host-specific (`server-3090`) and fixed to the observed HQQ configuration.
