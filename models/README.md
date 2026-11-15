# UCR-MFG UAVDT weights (one file per curriculum stage)

| File | Stage | Epoch | Test mAP / D-ECE (single view) | Came from |
|---|---|---|---|---|
| stage1_ep020.pth | 1 (FCOS + C1) | 20 | 18.4 / 0.088 | checkpoints/uavdt_stage1_rerun/stage1_end.pth (re-run with final recipe) |
| stage2_ep035.pth | 2 (+ fuzzy graph) | 35 | 18.8 / 0.094 | saved_models/uavdt_ep35_mAP18.9.pth (best stage-2 epoch on monitoring set) |
| stage3_ep060.pth | 3 (+ C2, C3) | 60 | 18.0 / 0.065 | checkpoints/uavdt_stage4_residual/stage3_end.pth |
| stage4_ep100_final.pth | 4 (+ C4), **final model** | 100 | 17.5 / 0.066 (18.9 mAP with TTA) | checkpoints/uavdt_stage4_residual/stage4_end.pth |

Test results for each file are in `results/stages/stageN/` and `results/final_model/`.
The `meta.weights` field inside those JSON files still shows the old path above.
Training logs (`train_log.jsonl`, `stdout.log`, `spikes.jsonl`) stay with each run in `checkpoints/<run>/`.
