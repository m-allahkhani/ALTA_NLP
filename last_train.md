============================================================
DEVICE CONFIGURATION
============================================================
Device: cuda
GPU count: 1
GPU 0: Tesla T4
============================================================

============================================================
TRAINING CONFIGURATION
============================================================
Epochs: 8
Folds:  3
============================================================
Training samples: 5531

============================================================
PREPROCESSING SUMMARY
============================================================
Total samples:     5531
Changed samples:   1085
Unchanged samples: 4446
Changed percentage: 19.62%
Training samples: 5531


============================================================
FOLD 1/3
============================================================

[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base
Key                                     | Status     |  | 
----------------------------------------+------------+--+-
mask_predictions.classifier.weight      | UNEXPECTED |  | 
lm_predictions.lm_head.dense.bias       | UNEXPECTED |  | 
mask_predictions.classifier.bias        | UNEXPECTED |  | 
mask_predictions.dense.weight           | UNEXPECTED |  | 
lm_predictions.lm_head.bias             | UNEXPECTED |  | 
mask_predictions.dense.bias             | UNEXPECTED |  | 
lm_predictions.lm_head.LayerNorm.weight | UNEXPECTED |  | 
mask_predictions.LayerNorm.bias         | UNEXPECTED |  | 
lm_predictions.lm_head.LayerNorm.bias   | UNEXPECTED |  | 
lm_predictions.lm_head.dense.weight     | UNEXPECTED |  | 
mask_predictions.LayerNorm.weight       | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Using single device: cuda

Epoch 1 / 8

model.safetensors: downloading bytes:  97% 359M/371M [00:02<00:00, 336MB/s, 28.2MB/s  ]
model.safetensors: reconstructing file:  36% 134M/371M [00:02<00:04, 51.3MB/s]
model.safetensors: downloading bytes: 100% 371M/371M [00:03<00:00, 116MB/s, 34.0MB/s  ]
model.safetensors: reconstructing file: 100% 371M/371M [00:03<00:00, 116MB/s, 33.9MB/s  ]
#########################################
Sentiment score:  0.7638
Sentiment en-AU: 0.7638
Sentiment en-UK:0.8209
--------
Sarcasm score:  0.6206
Sarcasm en-AU:   0.6396
Sarcasm en-UK:   0.6206
--------
Train Loss: 0.3517
Validation Loss: 0.2706
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.6922
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 2 / 8
#########################################
Sentiment score:  0.8390
Sentiment en-AU: 0.8390
Sentiment en-UK:0.9294
--------
Sarcasm score:  0.6731
Sarcasm en-AU:   0.7198
Sarcasm en-UK:   0.6731
--------
Train Loss: 0.2268
Validation Loss: 0.1851
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7561
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 3 / 8
#########################################
Sentiment score:  0.8563
Sentiment en-AU: 0.8563
Sentiment en-UK:0.9424
--------
Sarcasm score:  0.6972
Sarcasm en-AU:   0.7308
Sarcasm en-UK:   0.6972
--------
Train Loss: 0.1701
Validation Loss: 0.1742
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7767
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 4 / 8
#########################################
Sentiment score:  0.8581
Sentiment en-AU: 0.8581
Sentiment en-UK:0.9423
--------
Sarcasm score:  0.6911
Sarcasm en-AU:   0.7425
Sarcasm en-UK:   0.6911
--------
Train Loss: 0.1542
Validation Loss: 0.1629
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7746
#########################################
No improvement for 1 epoch(s).

Epoch 5 / 8
#########################################
Sentiment score:  0.8654
Sentiment en-AU: 0.8654
Sentiment en-UK:0.9488
--------
Sarcasm score:  0.7085
Sarcasm en-AU:   0.7460
Sarcasm en-UK:   0.7085
--------
Train Loss: 0.1476
Validation Loss: 0.1565
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7870
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 6 / 8
#########################################
Sentiment score:  0.8712
Sentiment en-AU: 0.8712
Sentiment en-UK:0.9488
--------
Sarcasm score:  0.7223
Sarcasm en-AU:   0.7483
Sarcasm en-UK:   0.7223
--------
Train Loss: 0.1429
Validation Loss: 0.1526
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7967
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 7 / 8
#########################################
Sentiment score:  0.8734
Sentiment en-AU: 0.8734
Sentiment en-UK:0.9488
--------
Sarcasm score:  0.7307
Sarcasm en-AU:   0.7465
Sarcasm en-UK:   0.7307
--------
Train Loss: 0.1368
Validation Loss: 0.1536
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8020
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 8 / 8
#########################################
Sentiment score:  0.8701
Sentiment en-AU: 0.8701
Sentiment en-UK:0.9488
--------
Sarcasm score:  0.7300
Sarcasm en-AU:   0.7519
Sarcasm en-UK:   0.7300
--------
Train Loss: 0.1380
Validation Loss: 0.1521
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8001
#########################################
No improvement for 1 epoch(s).

Best Fold Score: 0.8020
Best Epoch: 7

============================================================
FOLD 2/3
============================================================
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base
Key                                     | Status     |  | 


Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Using single device: cuda

Epoch 1 / 8
#########################################
Sentiment score:  0.7724
Sentiment en-AU: 0.7724
Sentiment en-UK:0.8408
--------
Sarcasm score:  0.6068
Sarcasm en-AU:   0.6778
Sarcasm en-UK:   0.6068
--------
Train Loss: 0.3499
Validation Loss: 0.2741
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.6896
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 2 / 8
#########################################
Sentiment score:  0.8349
Sentiment en-AU: 0.8349
Sentiment en-UK:0.9297
--------
Sarcasm score:  0.6432
Sarcasm en-AU:   0.7356
Sarcasm en-UK:   0.6432
--------
Train Loss: 0.2250
Validation Loss: 0.1827
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7391
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 3 / 8
#########################################
Sentiment score:  0.8487
Sentiment en-AU: 0.8487
Sentiment en-UK:0.9393
--------
Sarcasm score:  0.7219
Sarcasm en-AU:   0.7504
Sarcasm en-UK:   0.7219
--------
Train Loss: 0.1695
Validation Loss: 0.1711
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7853
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 4 / 8
#########################################
Sentiment score:  0.8554
Sentiment en-AU: 0.8554
Sentiment en-UK:0.9435
--------
Sarcasm score:  0.7015
Sarcasm en-AU:   0.7231
Sarcasm en-UK:   0.7015
--------
Train Loss: 0.1510
Validation Loss: 0.1558
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7784
#########################################
No improvement for 1 epoch(s).

Epoch 5 / 8
#########################################
Sentiment score:  0.8669
Sentiment en-AU: 0.8669
Sentiment en-UK:0.9414
--------
Sarcasm score:  0.7313
Sarcasm en-AU:   0.7615
Sarcasm en-UK:   0.7313
--------
Train Loss: 0.1483
Validation Loss: 0.1597
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7991
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 6 / 8
#########################################
Sentiment score:  0.8679
Sentiment en-AU: 0.8679
Sentiment en-UK:0.9446
--------
Sarcasm score:  0.7313
Sarcasm en-AU:   0.7608
Sarcasm en-UK:   0.7313
--------
Train Loss: 0.1404
Validation Loss: 0.1545
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7996
#########################################
No improvement for 1 epoch(s).

Epoch 7 / 8
#########################################
Sentiment score:  0.8714
Sentiment en-AU: 0.8714
Sentiment en-UK:0.9446
--------
Sarcasm score:  0.7386
Sarcasm en-AU:   0.7663
Sarcasm en-UK:   0.7386
--------
Train Loss: 0.1363
Validation Loss: 0.1562
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8050
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 8 / 8
#########################################
Sentiment score:  0.8714
Sentiment en-AU: 0.8714
Sentiment en-UK:0.9435
--------
Sarcasm score:  0.7386
Sarcasm en-AU:   0.7632
Sarcasm en-UK:   0.7386
--------
Train Loss: 0.1353
Validation Loss: 0.1550
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8050
#########################################
No improvement for 1 epoch(s).

Best Fold Score: 0.8050
Best Epoch: 7

============================================================
FOLD 3/3
============================================================
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base
Key                                     | Status     |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Using single device: cuda

Epoch 1 / 8
#########################################
Sentiment score:  0.7619
Sentiment en-AU: 0.7619
Sentiment en-UK:0.7644
--------
Sarcasm score:  0.6751
Sarcasm en-AU:   0.7113
Sarcasm en-UK:   0.6751
--------
Train Loss: 0.3485
Validation Loss: 0.2758
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7185
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 2 / 8
#########################################
Sentiment score:  0.8058
Sentiment en-AU: 0.8058
Sentiment en-UK:0.9240
--------
Sarcasm score:  0.6939
Sarcasm en-AU:   0.7141
Sarcasm en-UK:   0.6939
--------
Train Loss: 0.2279
Validation Loss: 0.1872
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7498
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 3 / 8
#########################################
Sentiment score:  0.8271
Sentiment en-AU: 0.8271
Sentiment en-UK:0.9444
--------
Sarcasm score:  0.7404
Sarcasm en-AU:   0.7622
Sarcasm en-UK:   0.7404
--------
Train Loss: 0.1690
Validation Loss: 0.1721
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7838
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 4 / 8
#########################################
Sentiment score:  0.8520
Sentiment en-AU: 0.8520
Sentiment en-UK:0.9477
--------
Sarcasm score:  0.7414
Sarcasm en-AU:   0.7705
Sarcasm en-UK:   0.7414
--------
Train Loss: 0.1523
Validation Loss: 0.1594
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.7967
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 5 / 8
#########################################
Sentiment score:  0.8591
Sentiment en-AU: 0.8591
Sentiment en-UK:0.9531
--------
Sarcasm score:  0.7540
Sarcasm en-AU:   0.7540
Sarcasm en-UK:   0.7542
--------
Train Loss: 0.1443
Validation Loss: 0.1521
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8065
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 6 / 8
#########################################
Sentiment score:  0.8603
Sentiment en-AU: 0.8603
Sentiment en-UK:0.9499
--------
Sarcasm score:  0.7549
Sarcasm en-AU:   0.7653
Sarcasm en-UK:   0.7549
--------
Train Loss: 0.1396
Validation Loss: 0.1529
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8076
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 7 / 8
#########################################
Sentiment score:  0.8636
Sentiment en-AU: 0.8636
Sentiment en-UK:0.9509
--------
Sarcasm score:  0.7542
Sarcasm en-AU:   0.7677
Sarcasm en-UK:   0.7542
--------
Train Loss: 0.1370
Validation Loss: 0.1522
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8089
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 8 / 8
#########################################
Sentiment score:  0.8658
Sentiment en-AU: 0.8658
Sentiment en-UK:0.9488
--------
Sarcasm score:  0.7542
Sarcasm en-AU:   0.7719
Sarcasm en-UK:   0.7542
--------
Train Loss: 0.1347
Validation Loss: 0.1529
Sentiment Macro F1: {sentiment_f1:.4f}
Sarcasm Macro F1: {sarcasm_f1:.4f}
official_score: 0.8100
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Best Fold Score: 0.8100
Best Epoch: 8

============================================================
CROSS-VALIDATION RESULTS
============================================================
Fold scores: [0.8020372156242173, 0.804987111656779, 0.8100301716857302]
Mean: 0.8057
Std: 0.0033

============================================================
FINAL ENSEMBLE EVALUATION
============================================================

Validation/Test preprocessing:
Total samples:     757
Changed samples:   156
Changed percentage: 20.61%
Evaluation samples: 757

Dialect distribution:
variety
en-UK    386
en-AU    371
Name: count, dtype: int64
Loading Fold 1: checkpoints/best_fold_1.pt
Loading weights: 100% 198/198 [00:00<00:00, 19296.70it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading Fold 2: checkpoints/best_fold_2.pt
Loading weights: 100% 198/198 [00:00<00:00, 12525.22it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base
Key                                     | Status     |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading Fold 3: checkpoints/best_fold_3.pt
Loading weights: 100% 198/198 [00:00<00:00, 27063.55it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.

Loaded 3 models for ensemble.

============================================================
FINAL ALTA ENSEMBLE RESULTS
============================================================

f1-sentiment-en-AU: 0.8813
f1-sentiment-en-UK: 0.9244
f1-sarcasm-en-AU:   0.7664
f1-sarcasm-en-UK:   0.7228

FINAL SCORE: 0.8021

============================================================
Saved submission file:
/content/answer.csv


---------------------------------------------------------------------------------

Area:	Ensemble weighting              
Action:
        Your fold scores are [0.8020, 0.8050, 0.8100]. Use a weighted average (e.g., normalised by score validation  instead of a simple mean. This will push the ensemble closer to Fold 3’s performance.
----------------
Increase max_epochs to 12 and set patience to 3. Even if the best is saved, the model might still improve beyond epoch 8 with a slower decay.
-----------------------
Area:   Focus on Sarcasm‑EN‑UK
Action:
        This is your weakest class (0.7228). Try:
        • Class‑weighted loss (upweight sarcasm samples, especially en‑UK).
        • Dialect‑specific fine‑tuning: freeze the base and train a small adapter head just on en‑UK sarcasm.
        • Oversample en‑UK sarcasm samples in your training splits (synthetic or duplication).
--------------------------
Test‑Time Augmentation (TTA): Generate multiple paraphrases of each test sample, average predictions – often worth +0.005–0.01 in this task.

Add a second model: DeBERTa‑v3‑base is great, but an ensemble with a different architecture (e.g., RoBERTa‑large, XLMR‑large, or even a DeBERTa‑v3‑large) will reduce correlated errors. Even a simple 2‑model ensemble usually beats a 3‑fold ensemble of the same model.

Pseudo‑labelling: If you have unlabelled data, generate pseudo‑labels with your ensemble and re‑train – but be careful with noise.
---------------------------

