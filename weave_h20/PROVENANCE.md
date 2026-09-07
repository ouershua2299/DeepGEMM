# Provenance

## Sources
- Weave sources: CAKE tree tag `h20-megamoe-weave-20260906` (head `fd019738`, branch `local/final-cake`), files
  `loom/examples/weave/sm90_nvfp4_megamoe_h20_small_rs.py` and `loom/examples/weave/sm90_nvfp4_megamoe_h20_split_l2.py`.
  The measured default schedule is the generated CUDA of commit `fde540bf`; every later commit only adds default-off
  knobs (default CUDA verified byte-identical per schedule).
- Integration base: the published `all-m-opt` branch @5ce3ab0 (H200 original), plus the H20 admission commit
  `4fafb10` (78-SM bucket table; cherry-picked onto this branch). The kernels reproduced are the arms that table
  selects. The environment-gated phase mask used only during split validation is kept as a patch in `validation/`.
- Generated CUDA: `generated/*.cu`, sha256 in `generated/MANIFEST.json`, produced headlessly with the Weave compiler
  at the tag above for `sm_90a`.

## Kernel-side commits in the CAKE tree (chronological)
| commit | change |
|---|---|
| f9616654 | rebase drift fixes: WGMMA descriptor hoisting soundness, `SmemLoadVec` count=2 |
| 7980dc0f | `lane_index_source` knob, timestamp buffer before the distributed ABI |
| b756e2c7 | half-overlap of Mode2 decode and RS WGMMA inside a stage |
| 739882d8 | rotated K loop, 4-stage swap-AB rings |
| d79ba0ff | 6-stage swap-AB rings |
| 4cf473c2 | K-half overlap in the non-swap SS stage |
| b10d6dcd / 2030ae93 | split L2 kernel port + `lm.timestamp` sites |
| 320b46f3 | compiler: WGMMA operand tiles with 64B / 32B swizzles |
| cdfeac5d | decode-team knob (default off) |
| 8a439e7e | no per-K-block warpgroup barrier before the empty arrive (default) |
| fde540bf | no proxy fence after the full-stage wait (default; measured CUDA) |
| 6e83cf86 / 2543ace3 / 3d1c4c6a | stage-rotation, wide-math and compact-LUT knobs (default off) |

## Receipts (8x H20-3e, eight ranks)
- Small-M correctness: eight-rank gang evaluator, objective 1.0, ten rows bit-exact, all gates true
  (`evaluator_result_cake_local_latest_final_20260906`).
- Small-M timing: paired ABBA vs the frozen arms, runs `weave_nofence_vs_{dynrs,424,devm}_job4151138_20260906i`
  (five blocks x 32 calls, rank-MAX, profile off, GPU launch gate).
- Large M: hybrid driver `step3_{flash,pro}_m{2048,4096,8192}.log` (production L1 -> Weave L2 bit-exact; cleanup
  check bit-exact; paired L2 timing 0.986-0.995 of the production L2).
- Attribution and PM sampling: `attrib_kb_h{4096,7168}_m8.log`, `pm_h4096_m8.log`, `pm_bank_h4096_m8.log`,
  `pm_ldst_h4096_m8.log`.
