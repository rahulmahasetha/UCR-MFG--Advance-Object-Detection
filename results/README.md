# Results (UAVDT test: 3,318 frames, every 5th frame from offset 2)

- `final_model/` - final model (models/stage4_ep100_final.pth)
  - `test_single_view.json/.log` - single view, after the box-std fix (mAP 17.5, D-ECE 0.066, AUROCs)
  - `test_tta.json/.log` - test-time augmentation (scales 1, 1.25 + flip; mAP 18.9)
  - `stage4_end_choice.json` - chosen TTA setting; `single_view_choice.json` - plain setting
  - `stage4_end_tune*.json/.log`, `*_cache_s30.pkl` - TTA search on the 554 tuning frames
- `stages/stage1..3/` - `test.json` + `test.log` for models/stage1_ep020, stage2_ep035, stage3_ep060
- `demo/` - single-image demo pictures
