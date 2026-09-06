
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
Epochs: 12
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

model.safetensors: downloading bytes:   0% 0.00/371M [00:00<?, ?B/s]Using single device: cuda

Sarcasm class weights:
en-AU: negative=0.7116, positive=1.6816
en-UK: negative=0.5421, positive=6.4418

Epoch 1 / 12
#########################################
Sentiment score:  0.7753
Sentiment en-AU: 0.7753
Sentiment en-UK:0.8294
--------
Sarcasm score:  0.6319
Sarcasm en-AU:   0.6734
Sarcasm en-UK:   0.6319
--------
Train Loss: 0.3676
Validation Loss: 0.3077
Sentiment Macro F1: 0.8023
Sarcasm Macro F1: 0.6526
official_score: 0.7036
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 2 / 12
#########################################
Sentiment score:  0.8289
Sentiment en-AU: 0.8289
Sentiment en-UK:0.9207
--------
Sarcasm score:  0.6564
Sarcasm en-AU:   0.6982
Sarcasm en-UK:   0.6564
--------
Train Loss: 0.2454
Validation Loss: 0.1954
Sentiment Macro F1: 0.8748
Sarcasm Macro F1: 0.6773
official_score: 0.7426
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 3 / 12
#########################################
Sentiment score:  0.8553
Sentiment en-AU: 0.8553
Sentiment en-UK:0.9413
--------
Sarcasm score:  0.6944
Sarcasm en-AU:   0.7124
Sarcasm en-UK:   0.6944
--------
Train Loss: 0.1730
Validation Loss: 0.1832
Sentiment Macro F1: 0.8983
Sarcasm Macro F1: 0.7034
official_score: 0.7748
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 4 / 12
#########################################
Sentiment score:  0.8637
Sentiment en-AU: 0.8637
Sentiment en-UK:0.9412
--------
Sarcasm score:  0.6893
Sarcasm en-AU:   0.7441
Sarcasm en-UK:   0.6893
--------
Train Loss: 0.1562
Validation Loss: 0.1690
Sentiment Macro F1: 0.9024
Sarcasm Macro F1: 0.7167
official_score: 0.7765
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 5 / 12
#########################################
Sentiment score:  0.8754
Sentiment en-AU: 0.8754
Sentiment en-UK:0.9488
--------
Sarcasm score:  0.7328
Sarcasm en-AU:   0.7384
Sarcasm en-UK:   0.7328
--------
Train Loss: 0.1478
Validation Loss: 0.1647
Sentiment Macro F1: 0.9121
Sarcasm Macro F1: 0.7356
official_score: 0.8041
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 6 / 12
#########################################
Sentiment score:  0.8791
Sentiment en-AU: 0.8791
Sentiment en-UK:0.9509
--------
Sarcasm score:  0.7350
Sarcasm en-AU:   0.7484
Sarcasm en-UK:   0.7350
--------
Train Loss: 0.1403
Validation Loss: 0.1568
Sentiment Macro F1: 0.9150
Sarcasm Macro F1: 0.7417
official_score: 0.8070
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 7 / 12
#########################################
Sentiment score:  0.8868
Sentiment en-AU: 0.8868
Sentiment en-UK:0.9509
--------
Sarcasm score:  0.7445
Sarcasm en-AU:   0.7485
Sarcasm en-UK:   0.7445
--------
Train Loss: 0.1329
Validation Loss: 0.1546
Sentiment Macro F1: 0.9189
Sarcasm Macro F1: 0.7465
official_score: 0.8157
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 8 / 12
#########################################
Sentiment score:  0.8854
Sentiment en-AU: 0.8854
Sentiment en-UK:0.9519
--------
Sarcasm score:  0.7417
Sarcasm en-AU:   0.7523
Sarcasm en-UK:   0.7417
--------
Train Loss: 0.1305
Validation Loss: 0.1515
Sentiment Macro F1: 0.9187
Sarcasm Macro F1: 0.7470
official_score: 0.8135
#########################################
No improvement for 1 epoch(s).

Epoch 9 / 12
#########################################
Sentiment score:  0.8934
Sentiment en-AU: 0.8934
Sentiment en-UK:0.9509
--------
Sarcasm score:  0.7462
Sarcasm en-AU:   0.7533
Sarcasm en-UK:   0.7462
--------
Train Loss: 0.1286
Validation Loss: 0.1574
Sentiment Macro F1: 0.9222
Sarcasm Macro F1: 0.7497
official_score: 0.8198
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 10 / 12
#########################################
Sentiment score:  0.8946
Sentiment en-AU: 0.8946
Sentiment en-UK:0.9498
--------
Sarcasm score:  0.7549
Sarcasm en-AU:   0.7549
Sarcasm en-UK:   0.7613
--------
Train Loss: 0.1252
Validation Loss: 0.1549
Sentiment Macro F1: 0.9222
Sarcasm Macro F1: 0.7581
official_score: 0.8247
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Epoch 11 / 12
#########################################
Sentiment score:  0.8968
Sentiment en-AU: 0.8968
Sentiment en-UK:0.9531
--------
Sarcasm score:  0.7530
Sarcasm en-AU:   0.7642
Sarcasm en-UK:   0.7530
--------
Train Loss: 0.1248
Validation Loss: 0.1502
Sentiment Macro F1: 0.9249
Sarcasm Macro F1: 0.7586
official_score: 0.8249
#########################################
No improvement for 1 epoch(s).

Epoch 12 / 12
#########################################
Sentiment score:  0.8979
Sentiment en-AU: 0.8979
Sentiment en-UK:0.9520
--------
Sarcasm score:  0.7628
Sarcasm en-AU:   0.7628
Sarcasm en-UK:   0.7657
--------
Train Loss: 0.1240
Validation Loss: 0.1528
Sentiment Macro F1: 0.9249
Sarcasm Macro F1: 0.7642
official_score: 0.8303
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_1.pt

Best Fold Score: 0.8303
Best Epoch: 12

============================================================
FOLD 2/3
============================================================
Loading weights: 100% 198/198 [00:00<00:00, 14103.77it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base



Sarcasm class weights:
en-AU: negative=0.7116, positive=1.6816
en-UK: negative=0.5421, positive=6.4418

Epoch 1 / 12
#########################################
Sentiment score:  0.7746
Sentiment en-AU: 0.7746
Sentiment en-UK:0.8541
--------
Sarcasm score:  0.6529
Sarcasm en-AU:   0.7166
Sarcasm en-UK:   0.6529
--------
Train Loss: 0.3700
Validation Loss: 0.3103
Sentiment Macro F1: 0.8143
Sarcasm Macro F1: 0.6847
official_score: 0.7138
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 2 / 12
#########################################
Sentiment score:  0.8342
Sentiment en-AU: 0.8342
Sentiment en-UK:0.9190
--------
Sarcasm score:  0.6449
Sarcasm en-AU:   0.7350
Sarcasm en-UK:   0.6449
--------
Train Loss: 0.2434
Validation Loss: 0.1892
Sentiment Macro F1: 0.8766
Sarcasm Macro F1: 0.6899
official_score: 0.7395
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 3 / 12
#########################################
Sentiment score:  0.8394
Sentiment en-AU: 0.8394
Sentiment en-UK:0.9371
--------
Sarcasm score:  0.6840
Sarcasm en-AU:   0.7576
Sarcasm en-UK:   0.6840
--------
Train Loss: 0.1721
Validation Loss: 0.1703
Sentiment Macro F1: 0.8883
Sarcasm Macro F1: 0.7208
official_score: 0.7617
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 4 / 12
#########################################
Sentiment score:  0.8666
Sentiment en-AU: 0.8666
Sentiment en-UK:0.9425
--------
Sarcasm score:  0.6994
Sarcasm en-AU:   0.7620
Sarcasm en-UK:   0.6994
--------
Train Loss: 0.1546
Validation Loss: 0.1593
Sentiment Macro F1: 0.9045
Sarcasm Macro F1: 0.7307
official_score: 0.7830
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 5 / 12
#########################################
Sentiment score:  0.8671
Sentiment en-AU: 0.8671
Sentiment en-UK:0.9424
--------
Sarcasm score:  0.7385
Sarcasm en-AU:   0.7426
Sarcasm en-UK:   0.7385
--------
Train Loss: 0.1425
Validation Loss: 0.1775
Sentiment Macro F1: 0.9048
Sarcasm Macro F1: 0.7405
official_score: 0.8028
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 6 / 12
#########################################
Sentiment score:  0.8682
Sentiment en-AU: 0.8682
Sentiment en-UK:0.9424
--------
Sarcasm score:  0.7524
Sarcasm en-AU:   0.7716
Sarcasm en-UK:   0.7524
--------
Train Loss: 0.1360
Validation Loss: 0.1652
Sentiment Macro F1: 0.9053
Sarcasm Macro F1: 0.7620
official_score: 0.8103
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 7 / 12
#########################################
Sentiment score:  0.8813
Sentiment en-AU: 0.8813
Sentiment en-UK:0.9456
--------
Sarcasm score:  0.7661
Sarcasm en-AU:   0.7770
Sarcasm en-UK:   0.7661
--------
Train Loss: 0.1307
Validation Loss: 0.1575
Sentiment Macro F1: 0.9135
Sarcasm Macro F1: 0.7716
official_score: 0.8237
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 8 / 12
#########################################
Sentiment score:  0.8792
Sentiment en-AU: 0.8792
Sentiment en-UK:0.9489
--------
Sarcasm score:  0.7704
Sarcasm en-AU:   0.7803
Sarcasm en-UK:   0.7704
--------
Train Loss: 0.1285
Validation Loss: 0.1577
Sentiment Macro F1: 0.9140
Sarcasm Macro F1: 0.7753
official_score: 0.8248
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 9 / 12
#########################################
Sentiment score:  0.8901
Sentiment en-AU: 0.8901
Sentiment en-UK:0.9499
--------
Sarcasm score:  0.7600
Sarcasm en-AU:   0.7819
Sarcasm en-UK:   0.7600
--------
Train Loss: 0.1273
Validation Loss: 0.1516
Sentiment Macro F1: 0.9200
Sarcasm Macro F1: 0.7710
official_score: 0.8251
#########################################
No improvement for 1 epoch(s).

Epoch 10 / 12
#########################################
Sentiment score:  0.8912
Sentiment en-AU: 0.8912
Sentiment en-UK:0.9510
--------
Sarcasm score:  0.7692
Sarcasm en-AU:   0.7834
Sarcasm en-UK:   0.7692
--------
Train Loss: 0.1230
Validation Loss: 0.1525
Sentiment Macro F1: 0.9211
Sarcasm Macro F1: 0.7763
official_score: 0.8302
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_2.pt

Epoch 11 / 12
#########################################
Sentiment score:  0.8901
Sentiment en-AU: 0.8901
Sentiment en-UK:0.9499
--------
Sarcasm score:  0.7582
Sarcasm en-AU:   0.7806
Sarcasm en-UK:   0.7582
--------
Train Loss: 0.1231
Validation Loss: 0.1515
Sentiment Macro F1: 0.9200
Sarcasm Macro F1: 0.7694
official_score: 0.8242
#########################################
No improvement for 1 epoch(s).

Epoch 12 / 12
#########################################
Sentiment score:  0.8901
Sentiment en-AU: 0.8901
Sentiment en-UK:0.9499
--------
Sarcasm score:  0.7711
Sarcasm en-AU:   0.7852
Sarcasm en-UK:   0.7711
--------
Train Loss: 0.1242
Validation Loss: 0.1534
Sentiment Macro F1: 0.9200
Sarcasm Macro F1: 0.7781
official_score: 0.8306
#########################################
No improvement for 2 epoch(s).

Best Fold Score: 0.8302
Best Epoch: 10

============================================================
FOLD 3/3
============================================================
Loading weights: 100% 198/198 [00:00<00:00, 21202.82it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base


Sarcasm class weights:
en-AU: negative=0.7110, positive=1.6847
en-UK: negative=0.5421, positive=6.4452

Epoch 1 / 12
#########################################
Sentiment score:  0.7570
Sentiment en-AU: 0.7570
Sentiment en-UK:0.8494
--------
Sarcasm score:  0.5994
Sarcasm en-AU:   0.5994
Sarcasm en-UK:   0.6807
--------
Train Loss: 0.3716
Validation Loss: 0.3121
Sentiment Macro F1: 0.8032
Sarcasm Macro F1: 0.6400
official_score: 0.6782
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 2 / 12
#########################################
Sentiment score:  0.8161
Sentiment en-AU: 0.8161
Sentiment en-UK:0.9252
--------
Sarcasm score:  0.6498
Sarcasm en-AU:   0.7300
Sarcasm en-UK:   0.6498
--------
Train Loss: 0.2402
Validation Loss: 0.1829
Sentiment Macro F1: 0.8707
Sarcasm Macro F1: 0.6899
official_score: 0.7330
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 3 / 12
#########################################
Sentiment score:  0.8456
Sentiment en-AU: 0.8456
Sentiment en-UK:0.9380
--------
Sarcasm score:  0.6922
Sarcasm en-AU:   0.7396
Sarcasm en-UK:   0.6922
--------
Train Loss: 0.1698
Validation Loss: 0.1650
Sentiment Macro F1: 0.8918
Sarcasm Macro F1: 0.7159
official_score: 0.7689
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 4 / 12
#########################################
Sentiment score:  0.8669
Sentiment en-AU: 0.8669
Sentiment en-UK:0.9434
--------
Sarcasm score:  0.7299
Sarcasm en-AU:   0.7634
Sarcasm en-UK:   0.7299
--------
Train Loss: 0.1514
Validation Loss: 0.1587
Sentiment Macro F1: 0.9051
Sarcasm Macro F1: 0.7466
official_score: 0.7984
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 5 / 12
#########################################
Sentiment score:  0.8812
Sentiment en-AU: 0.8812
Sentiment en-UK:0.9541
--------
Sarcasm score:  0.7497
Sarcasm en-AU:   0.7703
Sarcasm en-UK:   0.7497
--------
Train Loss: 0.1423
Validation Loss: 0.1517
Sentiment Macro F1: 0.9177
Sarcasm Macro F1: 0.7600
official_score: 0.8154
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 6 / 12
#########################################
Sentiment score:  0.8878
Sentiment en-AU: 0.8878
Sentiment en-UK:0.9552
--------
Sarcasm score:  0.7533
Sarcasm en-AU:   0.7617
Sarcasm en-UK:   0.7533
--------
Train Loss: 0.1346
Validation Loss: 0.1512
Sentiment Macro F1: 0.9215
Sarcasm Macro F1: 0.7575
official_score: 0.8206
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 7 / 12
#########################################
Sentiment score:  0.8781
Sentiment en-AU: 0.8781
Sentiment en-UK:0.9509
--------
Sarcasm score:  0.7500
Sarcasm en-AU:   0.7580
Sarcasm en-UK:   0.7500
--------
Train Loss: 0.1313
Validation Loss: 0.1570
Sentiment Macro F1: 0.9145
Sarcasm Macro F1: 0.7540
official_score: 0.8140
#########################################
No improvement for 1 epoch(s).

Epoch 8 / 12
#########################################
Sentiment score:  0.8869
Sentiment en-AU: 0.8869
Sentiment en-UK:0.9530
--------
Sarcasm score:  0.7564
Sarcasm en-AU:   0.7564
Sarcasm en-UK:   0.7580
--------
Train Loss: 0.1279
Validation Loss: 0.1564
Sentiment Macro F1: 0.9200
Sarcasm Macro F1: 0.7572
official_score: 0.8216
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 9 / 12
#########################################
Sentiment score:  0.8934
Sentiment en-AU: 0.8934
Sentiment en-UK:0.9552
--------
Sarcasm score:  0.7559
Sarcasm en-AU:   0.7709
Sarcasm en-UK:   0.7559
--------
Train Loss: 0.1243
Validation Loss: 0.1504
Sentiment Macro F1: 0.9243
Sarcasm Macro F1: 0.7634
official_score: 0.8246
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 10 / 12
#########################################
Sentiment score:  0.8923
Sentiment en-AU: 0.8923
Sentiment en-UK:0.9552
--------
Sarcasm score:  0.7564
Sarcasm en-AU:   0.7689
Sarcasm en-UK:   0.7564
--------
Train Loss: 0.1245
Validation Loss: 0.1499
Sentiment Macro F1: 0.9238
Sarcasm Macro F1: 0.7626
official_score: 0.8244
#########################################
No improvement for 1 epoch(s).

Epoch 11 / 12
#########################################
Sentiment score:  0.8956
Sentiment en-AU: 0.8956
Sentiment en-UK:0.9552
--------
Sarcasm score:  0.7567
Sarcasm en-AU:   0.7623
Sarcasm en-UK:   0.7567
--------
Train Loss: 0.1214
Validation Loss: 0.1522
Sentiment Macro F1: 0.9254
Sarcasm Macro F1: 0.7595
official_score: 0.8261
#########################################
✓ New best model
✓ Saved best model → checkpoints/best_fold_3.pt

Epoch 12 / 12
#########################################
Sentiment score:  0.8945
Sentiment en-AU: 0.8945
Sentiment en-UK:0.9573
--------
Sarcasm score:  0.7548
Sarcasm en-AU:   0.7628
Sarcasm en-UK:   0.7548
--------
Train Loss: 0.1197
Validation Loss: 0.1520
Sentiment Macro F1: 0.9259
Sarcasm Macro F1: 0.7588
official_score: 0.8247
#########################################
No improvement for 1 epoch(s).

Best Fold Score: 0.8261
Best Epoch: 11

============================================================
CROSS-VALIDATION RESULTS
============================================================
Fold scores: [0.8303455968225883, 0.830226734896882, 0.8261412139612774]
Mean: 0.8289
Std: 0.0020

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
Loading weights: 100% 198/198 [00:00<00:00, 29515.31it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base

Loading Fold 2: checkpoints/best_fold_2.pt
Loading weights: 100% 198/198 [00:00<00:00, 28458.37it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base

Loading Fold 3: checkpoints/best_fold_3.pt
Loading weights: 100% 198/198 [00:00<00:00, 24523.75it/s]
[transformers] DebertaV2Model LOAD REPORT from: microsoft/deberta-v3-base

Loaded 3 models for ensemble.

============================================================
FINAL ALTA ENSEMBLE RESULTS
============================================================

f1-sentiment-en-AU: 0.8893
f1-sentiment-en-UK: 0.9295
f1-sarcasm-en-AU:   0.7609
f1-sarcasm-en-UK:   0.7416

FINAL SCORE: 0.8154

============================================================
Saved submission file:
/content/answer.csv

-----------------
Test:

probability ensemble
vs
logit ensemble
-----------------
Test mild UK sarcasm oversampling.

-------------

A second encoder such as another pretrained architecture.
---------------------
I would do continued supervised sarcasm-domain adaptation of our existing DeBERTa encoder before ALTA fine-tuning.

The two datasets you found are genuinely relevant. iSarcasmEval was specifically built for intended sarcasm detection and includes English sarcastic/non-sarcastic texts; importantly, its authors collected labels from the authors of the texts rather than relying only on distant supervision
The ETS FigLang 2020 Reddit data is also directly a sarcasm-detection dataset, with about 4,400 Reddit training examples and explicit SARCASM/NOT_SARCASM labels; it also contains context, which is useful for sarcasm research even though your ALTA examples are not necessarily structured that way.
Option A — train only the general adapter

This is my preferred first experiment.

Your architecture currently has:

                    DeBERTa
                       │
              ┌────────┴────────┐
              │                 │
          General             dialect
                          ┌──────┴──────┐
                          AU            UK

For external sarcasm data, there is no reliable AU/UK label corresponding to ALTA's dialect definition.

So I would not train the AU/UK adapters on those datasets.

Instead:

external sarcasm data
        ↓
General LoRA
        ↓
sarcasm head

Then discard/reset the temporary sarcasm head and continue with the normal ALTA model:

general LoRA  ← transferred
AU LoRA       ← fresh
UK LoRA       ← fresh

sentiment head ← fresh
sarcasm head   ← fresh

This gives us a nice separation:

External datasets teach the general encoder how sarcasm behaves, while ALTA teaches it Australian/UK-specific sarcasm and the joint sentiment task.

I like this much more than pretraining the whole model indiscriminately.
iSarcasmEval specifically focuses on intended sarcasm, which is actually attractive for us.

The Reddit shared-task dataset, however, is built around sarcastic responses and conversation context.

Your ALTA data may differ in:

platform
writing style
labeling policy
sarcasm prevalence
context availability
dialect
topic

So external-data training can produce negative transfer.

The question isn't:

"Can external sarcasm data improve sarcasm?"

The question is:

"Can it improve ALTA sarcasm without changing the representation in a way that hurts ALTA?"

That's why we need a controlled experiment.

I would NOT combine all external data immediately

I'd actually run three stages.

Experiment 1 — iSarcasmEval only

Use the English portion of iSarcasmEval.

Why first?

Because it is particularly attractive for our purpose: its labels are based on author-reported intended sarcasm rather than indirect labels.

Then ALTA fine-tuning.

Experiment 2 — ETS Reddit only

Then:

iSarcasmEval
        vs
ETS Reddit

This tells us which domain transfers better.

Experiment 3 — both combined

Only if the individual experiments are helpful:

iSarcasmEval
      +
ETS Reddit

This gives us an actual scientific comparison rather than throwing everything into one model.
Should we pretrain the whole DeBERTa instead?

I would say no, not as the first experiment.

Full encoder adaptation on a small external dataset can cause catastrophic forgetting or make the resulting representation overly specialized to the external domain.

Your current LoRA design gives us a much safer experiment:

Base DeBERTa
   ↓
frozen pretrained knowledge
   +
general sarcasm LoRA

Then ALTA can continue adapting those LoRA parameters.

That's much lower-risk.

And I would NOT use the external test sets

This is important for competition methodology.

Use only the external datasets' training portions for adaptation.

Don't use their test labels to train or select anything.

Similarly, don't tune the ALTA system against the hidden competition test set.
There is one particularly powerful idea here

iSarcasmEval has something unusual: for sarcastic examples, it also contains a non-sarcastic rephrase conveying the same intended message.

That gives us a potential future training signal:

same semantic message
        │
   ┌────┴────┐
sarcastic   non-sarcastic

This is extremely interesting because it teaches the model:

"What changes when the same message becomes sarcastic?"

That could be more valuable than ordinary binary classification alone.

But I would not implement this in the first version. First see whether ordinary binary sarcasm adaptation helps ALTA.

Another important consideration: context

The ETS Reddit data provides:

response
context

and explicitly expects context to be used for sarcasm detection.

Your ALTA model currently receives only:

text

So don't simply concatenate Reddit context to the ALTA model and expect that to transfer cleanly.

For the initial adaptation stage, I'd use the response text itself as the input.

Later, we could investigate context-aware pretraining separately.