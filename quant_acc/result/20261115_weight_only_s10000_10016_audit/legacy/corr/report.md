# Quant accuracy factor report

- Prepared cells: 4490

## Factor contribution ranking (delta R2 ablation)

      term_group  full_r2  reduced_r2  delta_r2
nbits:group_size 0.931815    0.872141  0.059674
           nbits 0.931815    0.885484  0.046331
   nbits:dataset 0.931815    0.918475  0.013339
     nbits:model 0.931815    0.924514  0.007301
 nbits:ckpt_kind 0.931815    0.929887  0.001928
         dataset 0.931815    0.931470  0.000345
           model 0.931815    0.931776  0.000039
       ckpt_kind 0.931815    0.931778  0.000036
      group_size 0.931815    0.931806  0.000009

## Spearman rank correlation supplement

                              scope          factor   method context   n  correlation
within_summary:group_within_context log2_group_size spearman summary NaN     -0.80000
within_summary:nbits_within_context       hqq_nbits spearman summary NaN      0.97619
