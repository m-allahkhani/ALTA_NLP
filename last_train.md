
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









0000000000000000000000000000000000000000
Streaming output truncated to the last 5000 lines.
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.582836e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.856810e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.105532e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.997010e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.008911
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.406907e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.166324e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.648789e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.296180e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.237797
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.274233e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.557953e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.413232e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.031046e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.819383
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.032244e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.901334e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.634581e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.126821e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005423
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.781635e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.579819e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.066698e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.075325e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000153
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.210000e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.889936e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.339146e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.231554e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.254874
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.374660e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.648466e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.384910e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.078748e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.731236
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.223007e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.346326e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.321591e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.165661e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.019597
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.465909e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.726301e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.856740e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.036453e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006202
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.212568e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.896814e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.317052e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.535387e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.469410
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.949040e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.103348e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.109774e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.393225e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000600
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.791096e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.484085e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.200814e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.295353e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.037670
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.923971e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.914954e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.109776e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.855702e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004249
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.295034e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.994022e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.552781e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.175269e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.091112
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.668735e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.745567e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.524268e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.028043e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.003717
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.105034e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.850478e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.981447e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.477129e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001188
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.292680e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.804570e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.199430e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.575095e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.078055
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.520597e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.404030e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.366303e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.997187e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.059734
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.997132e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.734124e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.435814e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.715090e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.076587
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.221469e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.441794e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.021559e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.782106e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000959
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.095595e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.496673e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.719853e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.911878e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.033963
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.216148e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.266171e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.302950e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.059556e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.028529
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.881131e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.722623e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.076837e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.614343e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.023320
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.219488e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.980753e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.708448e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.962458e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.141282
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.691938e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.517027e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.800582e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.348480e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002958
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.328125e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.211952e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.216136e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.978730e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.008701
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.281459e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.943655e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.100574e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.594619e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.153158
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.983354e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.271652e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.765140e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.931815e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.092092
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.597800e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.420292e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.577956e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.261389e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005698
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.944528e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.602010e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.932580e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.715817e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001375
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.416432e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.897686e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.323662e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.683762e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.032897
After optimizer step: bad_parameters=0
#########################################
Sentiment score:  0.9076
Sentiment en-AU: 0.9076
Sentiment en-UK:0.9520
--------
Sarcasm score:  0.7609
Sarcasm en-AU:   0.7609
Sarcasm en-UK:   0.7708
--------
Train Loss: 0.1166
Validation Loss: 0.1417
Sentiment Macro F1: 0.9298
Sarcasm Macro F1: 0.7659
official_score: 0.8343
#########################################
✓ New best model
✓ Saved best model → checkpoints/task_specific/best_fold_1.pt

======================================================================
FINAL LoRA DTYPE CHECK
======================================================================
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.0.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.1.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.2.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.3.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.4.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.5.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.6.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.7.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.8.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.9.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.10.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.query_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.key_proj.lora_B.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_A.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_A.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_A.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_A.sarcasm_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_B.sentiment_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_B.sentiment_uk.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_B.sarcasm_au.weight: torch.float32
encoder.base_model.model.encoder.layer.11.attention.self.value_proj.lora_B.sarcasm_uk.weight: torch.float32
======================================================================
✓ All LoRA parameters are float32.

Epoch 8 / 12

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.402590e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.806186e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.937576e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.714193e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.257437
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.627648e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.762524e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.742922e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.638515e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.148489
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.381370e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.559136e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.039057e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.071455e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.027617
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.519193e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.981148e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.547714e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.170747e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.053623
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.495904e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.851707e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.681139e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.210474e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.026702
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.898760e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.521555e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.664357e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.116684e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006387
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sentiment_uk    | norm=3.266289e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sarcasm_uk      | norm=2.102409e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.034129
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.909004e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.980596e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.210765e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.057048e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.037304
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.872468e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.022247e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.577978e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.671266e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.089850
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.235395e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.575519e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.422573e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.186030e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.056165
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.219676e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.097133e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.034949e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.273850e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.012192
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.067895e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.322913e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.707344e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.301719e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.066357
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.355310e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.660373e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.676394e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.085213e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.061900
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.650515e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.945368e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.615338e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.566508e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002021
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.135531e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.972048e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.588358e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.361289e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002270
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.521452e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.997613e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.701359e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.428085e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.022680
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.636199e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.452853e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.429782e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.900407e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.198788
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.973106e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.387875e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.532871e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.333306e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.012441
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.538540e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.030163e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.890817e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.633415e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006486
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.654369e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.825318e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.121395e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.469717e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001084
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sentiment_uk    | norm=7.793489e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sarcasm_uk      | norm=2.251363e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.043044
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.299724e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.009007e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.085930e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.865806e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.046939
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.837774e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.095722e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.499545e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.108242e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.015271
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.587149e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.435991e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.431866e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.890662e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.057654
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.532790e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.716610e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.652184e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.137292e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.074206
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.231956e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.711562e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.495274e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.928961e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001345
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.448282e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.100730e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.168605e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.456371e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.085196
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.411718e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.890487e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.339917e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.154290e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.033572
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.177561e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.142841e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.693278e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.086101e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.172071
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.073767e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.810847e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.353172e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.148869e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.015390
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sentiment_uk    | norm=7.241745e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.995921e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.587458e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000750
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.728167e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.007947e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.966207e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.936324e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.044005
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.799608e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.910255e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.946792e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.956024e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.300162
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.865100e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.255964e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.041846e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.790133e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.149125
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.560856e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.420448e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.975277e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.724815e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.258179
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.207754e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.654614e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.404931e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.007059e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.017635
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.605396e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.131086e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.132075e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.912560e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.095003
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.176212e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.152367e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.368867e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.556854e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.712612
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.426855e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.119480e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.643663e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.782321e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.053223
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.260286e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.564204e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.378700e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.320047e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.092248
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.510025e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.319740e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.503633e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.563431e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.020113
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.644406e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.950617e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.858845e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.984539e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.066437
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.061482e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.456299e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.621015e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.522946e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000311
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.792473e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.499770e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.804500e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.082219e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.110559
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.348534e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.145056e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.504091e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.155801e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005173
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.736179e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.187111e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.736754e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.405216e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.244593
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.612764e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.543144e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.472086e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.609342e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.008464
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.164443e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.555899e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.710187e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.254578e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004110
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.147619e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.066775e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.842133e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.394583e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.009031
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.069043e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.502484e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.048547e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.175045e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001689
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.619521e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.269486e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.631024e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.084406e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.027287
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.911614e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.317069e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.166753e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.484889e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001130
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.112503e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.254081e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.248607e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.407801e-05 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.075230
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.976787e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.329584e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.301478e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.434351e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013852
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.473831e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.434147e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.290332e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.857599e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001254
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.262113e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.525893e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.207372e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.745522e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.012979
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.139407e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.195408e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.288984e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.968033e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.233660
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.009770e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.057985e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.532527e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.917393e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.091364
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.864708e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.518739e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.984892e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.421664e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.034991
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.757028e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sarcasm_au      | norm=4.319777e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.741555e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.567166
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.692930e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.934773e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.473760e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.177456e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.041469
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.278318e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.346273e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.471361e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.291821e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.025762
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.659653e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.560093e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.730968e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.800240e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000288
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.452635e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.091535e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.380234e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.540707e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.202460
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.037915e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.321712e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.350131e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.376235e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013205
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.793555e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.420346e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.627738e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.740035e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.031199
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.623858e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.018363e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.761845e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.668095e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001311
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.135593e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.635261e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.900002e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.087068e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001559
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.129506e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sarcasm_au      | norm=8.276341e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.619118e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.033614
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.100202e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.666207e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.337503e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.220463e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.083637
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.557760e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.165342e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.744377e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.132478e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.241936
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.270304e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.927280e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.497866e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.769565e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.394101
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.862682e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.830078e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.918388e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.905210e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000183
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.997059e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.155374e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.304099e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.557565e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000260
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.021089e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.509000e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.280543e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.413710e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000356
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.086115e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.390983e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.816157e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.130913e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.070901
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.752253e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.870672e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.664141e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.382633e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.044267
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.106146e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.340854e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.131673e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.719068e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013518
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.545885e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.341081e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.834273e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.109636e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.092889
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.487807e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.260368e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.940331e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.273016e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.743176
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.538782e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.378807e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.897687e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.715548e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.012061
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.984464e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.700262e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.851368e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.288969e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001152
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.855877e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.036497e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.546326e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.783763e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.318532
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.093703e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.987843e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.598048e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.180143e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000153
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.451718e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sarcasm_au      | norm=1.233178e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.312543e-07 | grad=72 | missing=0 | nonzero=14
======================================================================
Before optimizer step: loss=0.104005
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.020809e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.339670e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.091720e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.040770e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.003698
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.953436e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.909035e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.758409e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.150557e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.055383
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.936229e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.869655e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.658878e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.904262e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.030382
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.310676e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.882863e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.561971e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.951466e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.062718
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.691978e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.705667e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.435464e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.550540e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.082070
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.588632e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.697812e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.220866e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.134628e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.029549
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.178783e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.285037e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.344212e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.920044e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.792527
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.772808e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.327820e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.418546e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.724648e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.068677
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.108601e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.536580e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.240042e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.176815e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000236
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.689031e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.181772e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.934213e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.596938e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.028505
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.074834e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.370691e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.152243e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.588158e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.851468
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.832726e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.607639e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.699831e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.001910e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001492
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.483218e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.311311e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.765359e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.135518e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.211547
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.360052e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.776303e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.495246e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.038529e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.078605
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.714498e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.732998e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.132660e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.342173e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000721
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.317055e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.929644e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.413464e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.009567e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.003381
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.509871e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.867864e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.925668e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.482951e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000356
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.354068e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.255339e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.083287e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.205655e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.089470
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.974268e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.072296e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.034494e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.321542e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.019764
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.617539e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.991896e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.002714e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.632848e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.261867
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.624733e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.870652e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.058172e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.664764e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.109329
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.324278e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.827544e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.665783e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.602326e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.050262
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.808473e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.053225e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.944712e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.045658e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.067424
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.571471e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.764752e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.862364e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.106487e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.055155
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.170768e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.700457e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.695939e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.624570e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.020574
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.304139e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.401326e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.248893e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.350303e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004098
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.526452e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.301228e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.702289e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.880087e-06 | grad=72 | missing=0 | nonzero=24
======================================================================
Before optimizer step: loss=0.023919
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.397802e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.808216e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.100051e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.793629e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000768
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.274990e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.517285e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.783547e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.538510e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.109531
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.523872e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.560467e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.701866e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.892738e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.009130
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.069433e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.098810e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.223321e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.416743e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.038779
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.317538e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.906300e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.463628e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.072810e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.024388
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.008975e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.122427e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.995214e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.924833e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.063818
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.235362e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.686954e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.719387e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.045702e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.559935
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.850721e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.184971e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.935156e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.392905e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.545366
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.434199e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.746224e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.749464e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.583286e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000188
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.569402e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.429354e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.694956e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.750771e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.035714
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.720815e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.011156e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.174139e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.638662e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.007023
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.078155e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.773355e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.211719e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.134159e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000235
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.638035e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.191131e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.482239e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.954744e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000601
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.497311e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.574361e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.072206e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.873442e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.035406
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.329935e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.952536e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.955307e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.164125e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.018966
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.808776e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.008532e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.507284e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.835776e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.084988
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.092119e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.728219e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.637301e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.638353e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.177603
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.560575e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.977274e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.623320e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.036476e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=1.266057
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.395561e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.988568e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.128451e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.936922e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.056153
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.291362e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.768690e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.299687e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.148963e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.044172
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.955296e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.596227e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.884174e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.885001e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010702
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.318102e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.578291e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.997666e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.227135e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.062968
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.967611e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.318423e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.564734e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.191486e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013391
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.196482e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.207032e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.495632e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.730226e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.023295
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.501835e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.272897e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.531219e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.374606e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.027884
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.381283e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.978437e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.850618e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.382208e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.073093
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.169764e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.763082e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.096450e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.657653e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013805
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.905278e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.413226e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.856211e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.332873e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.161270
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.505098e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.123947e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.335540e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.123543e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.030696
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.359063e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.816267e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.038751e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.476022e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001084
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.154102e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.002009e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.761632e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.479760e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.172880
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.018844e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.850527e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.241431e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.094046e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.007735
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.575785e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.308568e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.864820e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.728709e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.011370
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.052033e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.108129e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.585656e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.964446e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005233
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.938051e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.166612e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.405748e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.462743e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.014345
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.732307e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.911495e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.247353e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.127910e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.284071
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.982271e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.041481e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.219262e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.823117e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.396274
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.119415e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.034754e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.638826e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.719322e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.080832
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.041750e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.160248e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.552257e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.245513e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.050668
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.284515e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.300795e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.205673e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.138012e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.209994
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.318385e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.580403e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.254956e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.748915e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.167149
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.229167e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.970882e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.661876e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.002531e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.027822
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.011813e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.901136e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.171272e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.754520e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.062812
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.692699e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.575162e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.769586e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.734742e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005459
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.694818e-04 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.841954e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.610245e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.455178e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.167309
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.716775e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.707558e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.705665e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.218846e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000798
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.280567e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sarcasm_au      | norm=2.919586e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.411604e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002654
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.468899e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.075875e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.119492e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.393568e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.035479
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sentiment_uk    | norm=4.314708e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.758112e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.553402e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.028777
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.887223e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.487302e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.409952e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.993502e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002960
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.818449e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.429785e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.663110e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.851298e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000891
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.626223e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.368096e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.205040e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.239141e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.061402
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.236512e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.479460e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.432292e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.750039e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.028892
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.425590e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.903355e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.552756e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.691402e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.045987
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.434830e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.795535e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.896669e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.556874e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.018981
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.311527e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.239866e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.647707e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.589065e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.026478
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.426072e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.928482e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.530658e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.814795e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005621
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.574216e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.398293e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.427565e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.225517e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.035168
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.927832e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.331581e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.140477e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.286460e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.554476
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.276116e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.329820e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.159052e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.435471e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.065605
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.257588e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.777160e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.763363e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.328325e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.352243
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.836780e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.494182e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.615976e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.401046e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.122072
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.109590e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.206475e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.803035e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.105652e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.057198
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.105183e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.698440e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.960938e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.081044e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.173715
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.967167e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.949008e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.160034e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.044676e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.007314
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.222929e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.361276e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.174462e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.957103e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001918
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.286246e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.287576e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.165245e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.164193e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.023052
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.283623e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.126522e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.217756e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.147434e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.124804
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.361002e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.964306e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.927658e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.772082e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.059698
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.664390e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.007628e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.234481e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.235730e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002851
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.714827e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.990219e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.472257e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.462888e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.164005
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.749632e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.038005e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.972266e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.205926e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.571418
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.622657e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.503551e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.395363e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.686812e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002692
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.101525e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.383358e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.846110e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.199636e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.019717
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.319929e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.025660e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.576592e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.838696e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010211
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.910730e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.751329e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.807725e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.085855e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013960
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.076398e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.961575e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.789673e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.174134e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.568062
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.439615e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.700861e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.282343e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.448832e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.317149
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.572485e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.881183e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.109258e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.093540e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.082037
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.570422e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.208128e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.454076e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.995991e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.039869
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.333308e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.723208e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.916824e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.377893e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.032691
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.616674e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.566614e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.219356e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.022705e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006137
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.997208e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
sarcasm_au      | norm=3.262976e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=0.000000e+00 | grad=72 | missing=0 | nonzero=0
======================================================================
Before optimizer step: loss=0.291738
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.825412e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.466217e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.479564e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.886296e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.679683
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.343560e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.230711e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.244353e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.242096e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001978
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.063621e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.619200e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.981177e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.730378e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.045403
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.020742e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.158152e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.053095e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.728003e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.038507
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.830687e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.490619e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.835038e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.184806e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.039759
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.365248e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.735745e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.903093e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.908463e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.017890
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.833924e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.856296e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.066116e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.346124e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.042654
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.275760e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.741104e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.023436e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.440221e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.382748
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.973465e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.863744e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.511376e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.473288e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.057881
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.515656e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.764095e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.152453e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.065427e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.007665
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.493901e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.274273e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.892932e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.923221e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.032951
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.780308e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.851591e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.417065e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.453295e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002060
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sentiment_uk    | norm=9.092063e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sarcasm_uk      | norm=1.013162e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010808
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.581072e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.388568e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.673355e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.601094e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002711
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.898311e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.520866e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.074978e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.865282e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006801
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.562454e-04 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.105013e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.595450e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.021801e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000808
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.914444e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.736999e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.568417e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.627042e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004646
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.773005e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.570906e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.657762e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.290500e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.119667
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.142500e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.032068e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.759624e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.315062e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.109919
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.691557e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.506199e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.975505e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.327367e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000705
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.762895e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.532509e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.122540e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.413501e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.015300
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.837617e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.660669e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.466985e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.963383e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.031329
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.243630e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.660450e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.385613e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.687832e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.143153
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.543211e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.206101e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.132079e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.367663e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.053578
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.933996e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.601272e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.744752e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.760800e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.153277
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sentiment_uk    | norm=7.410004e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=0.000000e+00 | grad=0 | missing=72 | nonzero=0
sarcasm_uk      | norm=1.957629e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000199
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.915970e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.695295e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.387063e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.006970e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.037545
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.573657e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.703941e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.785215e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.854998e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001177
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.941870e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.618243e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.256581e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.347054e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.853733
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.134024e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.120304e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.198314e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.896834e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.033302
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.504243e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.997644e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.160010e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.228695e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.026300
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.823557e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.864049e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.459498e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.289905e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.071182
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.682936e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.314821e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.814508e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.582980e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001947
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.581043e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.660700e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.194665e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.316958e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.066936
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.778808e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.300458e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.560939e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.600438e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.050025
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.627773e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.858938e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.662327e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.216316e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.009206
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.565789e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.584897e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.069271e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.831604e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006381
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.382879e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.608314e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.764982e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.505237e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.022958
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.505906e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.155798e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.281594e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.262154e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.048169
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.833562e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.399159e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.629432e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.882083e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.007063
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.623611e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.746108e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.901327e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.559277e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.079163
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.830495e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.058853e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.385903e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.971809e-05 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.382373
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.439312e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.020769e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.296160e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.742320e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.015009
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.436762e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.994354e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.567763e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.602885e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.024750
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.826840e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.668710e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.937136e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.316830e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.090971
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.475176e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.485532e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.473171e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.721883e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.701636
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.747833e-04 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.071685e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.811799e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.859886e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001406
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.645261e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.819370e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.026369e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.086970e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.036358
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.935656e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.854224e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.735614e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.205137e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.511134
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.609831e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.843659e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.038818e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.496114e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001788
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.454857e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.420080e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.554425e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.056349e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000420
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.595660e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.809596e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.367353e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.895277e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.755557
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.514347e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.979434e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.788002e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.511660e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.003159
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.833845e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.222548e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.548387e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.216672e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002629
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.641113e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.787116e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.561598e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.265440e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010170
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.613151e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.403512e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.158415e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.376223e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.599722
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.174731e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.123051e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.296453e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.603349e-07 | grad=72 | missing=0 | nonzero=14
======================================================================
Before optimizer step: loss=0.224053
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.126658e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.658278e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.835779e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.002248e-05 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010076
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.437492e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.348097e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.180687e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.138519e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.037461
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.095040e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.616188e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.877515e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.828255e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004693
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.504065e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.674821e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.085275e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.243921e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.113328
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.450555e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.509145e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.529149e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.694133e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.310324
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.623701e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.345248e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.551907e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.868326e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.352375
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.165922e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.954110e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.445282e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.345811e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.117994
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.142772e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.325025e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.140172e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.642347e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.061332
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.676832e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.359097e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.862468e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.873555e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.063749
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.332837e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.166024e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.170824e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.325017e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010460
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.491304e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.565484e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.757152e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.918302e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000927
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.602933e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.429474e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.253293e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.122422e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.104916
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.365436e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.400968e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.975928e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.897983e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.047060
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.274773e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.356796e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.213501e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.729316e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000259
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.571471e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.825258e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.946714e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.664502e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.121814
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.085866e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.111541e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.236768e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.243016e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.054989
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.338610e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.958757e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.001424e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.835452e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.014433
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.241631e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.824151e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.358557e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.052967e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.038846
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.643391e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.669857e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.459017e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.790658e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.042986
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.207072e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.375223e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.492073e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.137153e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.058843
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.513489e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.992998e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.331428e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.385265e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.030023
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.545157e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.652709e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.084912e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.353590e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001683
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.366995e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.604775e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.685246e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.206163e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.359737
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.406724e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.419335e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.145678e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.835065e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.637651
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.882100e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.124308e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.917327e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.661449e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.057535
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.502190e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.427527e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.419836e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.342741e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004860
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.374997e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.916442e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.826412e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.304188e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.806443
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.944057e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.121749e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.731948e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.455876e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.052456
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.609503e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.433819e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.668973e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.624795e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004674
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.683472e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.112802e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.593464e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.341832e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.021733
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.408088e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.130000e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.471158e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.457366e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.303226
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.356361e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.332225e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.885556e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.568231e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000325
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.994730e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.328817e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.469680e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.736663e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.108116
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.535985e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.215304e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.705499e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.038759e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.016519
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.272784e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.796788e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.452596e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.086150e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002206
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.095109e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.606531e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.283444e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.868468e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.019856
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.497179e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.063532e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.389591e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.179038e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005083
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.651236e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.286644e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.098191e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.247609e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.054906
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.058726e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.609459e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.738475e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.608809e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004550
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.614611e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.937481e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.447053e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.570166e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.023774
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.651210e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.231733e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.105372e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.615293e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.076007
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.429512e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.420275e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.672744e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.057253e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.050973
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.670667e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.764939e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.285221e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.210103e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.047631
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.997350e-04 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.933196e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.322563e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.895333e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.027923
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.430752e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.687405e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.452545e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.600061e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.132501
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.677144e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.123392e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.972518e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.178840e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.003183
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.287334e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.529189e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.786789e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.517733e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.007568
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.909816e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.098638e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.934972e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.978355e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.679933
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.203579e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.495151e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.956771e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.405197e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.219609
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.478179e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.333325e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.014714e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.751200e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006189
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.064195e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.945108e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.111373e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.758078e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.132853
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.788285e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.270611e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.145591e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.708207e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.504114
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.697530e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.561618e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.338842e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.580224e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.160178
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.726092e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.626625e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.014874e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.237863e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002729
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.550390e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.658027e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.756051e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.264320e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.629100
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.027736e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.611494e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.566617e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.949150e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001408
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.594757e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.055279e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.059740e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.085697e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.049155
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.381016e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.025568e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.295390e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.585793e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006269
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.706663e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.819032e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.045847e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.575909e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.011169
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.600444e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.647124e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.449300e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.901421e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.068980
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.182101e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.400675e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.764179e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.905265e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000802
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.412951e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.851642e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.748117e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.475666e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.020052
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.707760e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.361657e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.235710e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.778294e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001142
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.350984e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.567544e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.818325e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.045995e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.028184
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.515357e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.071618e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.924432e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.353131e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.286709
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.685060e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.277428e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.368775e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.597942e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.153925
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.668318e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.582204e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.261809e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.117043e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.034587
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.963849e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.809708e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.694968e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.864722e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.425040
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.233241e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.616483e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.875698e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.662735e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001046
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.194182e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.377366e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.197025e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.136441e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.284144
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.715013e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.074335e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.181409e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.455026e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.130258
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.723451e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.607624e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.678083e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.320387e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004387
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.716952e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.053629e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.297755e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.267773e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.287208
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.340389e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.095691e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.751083e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.293413e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.178781
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.484246e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.431409e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.319479e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.920208e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.008628
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.264233e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.684261e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.588282e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.677606e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.064689
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.079464e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.470818e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.706806e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.338235e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.012698
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.875148e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.189379e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.273868e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.313887e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.507544
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.560803e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.238936e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.002681e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.256743e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.018306
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.957422e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.533841e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.602521e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.351452e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.421913
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.465203e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.710845e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.872609e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.721718e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013060
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.307163e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.935430e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.119156e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.461274e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.040739
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.087934e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.095554e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.562321e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.674295e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.062129
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.259996e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.491601e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.172630e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.340980e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002920
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.043417e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.351596e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.923730e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.670835e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.136539
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.850292e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.358122e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.570032e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.859343e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.038260
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.200172e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.655970e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.273120e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.937213e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000095
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.673243e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.141888e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.098516e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.740893e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.154145
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.487670e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.686213e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.524915e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.149891e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.015523
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.015439e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.891659e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.374566e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.110389e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=2.370206
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.156534e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.581773e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.089306e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.040977e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.018580
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.015052e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.023359e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.262359e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.898571e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.631670
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.586830e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.869334e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.675809e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.251525e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000718
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.738991e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.798337e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.798472e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.370707e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.191061
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.139574e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.898918e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.619967e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.784541e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.221102
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.965149e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.997388e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.062798e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.716203e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.018251
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.714105e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.614775e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.483319e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.133282e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.509302
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.665707e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.670256e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.551214e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.060407e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.005605
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.199327e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.919921e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.033884e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.535551e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.014997
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.844195e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.637310e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.346658e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.510982e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.404678
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.689786e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.979015e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.511365e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.241326e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.013031
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.485182e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.204348e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.641989e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.372685e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.084256
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.098345e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.175338e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.170592e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.200648e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.072681
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.866375e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.008167e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.453260e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.996822e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.177627
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.118331e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.184679e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.865209e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.366192e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.135261
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.681029e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.020102e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.767179e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.465050e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.009129
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.972619e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.191380e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.456474e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.923658e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.248002
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.383313e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.814552e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=8.135601e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.152192e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.182713
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.836970e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.206266e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.488351e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.735983e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.014000
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.481319e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.634147e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.516158e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.398825e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.219498
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.222611e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.648465e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.100864e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.041280e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010662
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.536197e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.439759e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.083196e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.508937e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.021300
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.095821e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.394349e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.421737e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.587846e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.000827
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.036037e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.498091e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.092768e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=9.176123e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.086742
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.245364e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.484510e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.151984e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.146658e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.006033
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.366412e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.935946e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.249326e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.237125e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.059476
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.167806e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.276521e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.934030e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.149330e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.011241
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.015721e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.795753e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.171246e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.843848e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.010060
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.767379e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.724437e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.389174e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.936634e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.511611
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.120301e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.178870e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.134308e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.482099e-04 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.027169
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.437510e-03 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.845925e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.347018e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.727778e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.015638
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.479157e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.568344e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=9.112398e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.968627e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.041888
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=4.527709e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.012843e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=4.813921e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.267127e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.004128
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.499914e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.251924e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.479659e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.578211e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.002799
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.530967e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.244040e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=5.925266e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.441864e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.011693
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.540030e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.668655e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.344483e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.492693e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.101196
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.299078e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.868854e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.862471e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.766587e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.086741
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=5.423178e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.278604e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.585523e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=6.215809e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.098274
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.971759e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=5.321088e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.052000e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.391567e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.061995
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.627545e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.102610e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.279978e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.822239e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.025589
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.806031e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.677051e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.420900e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.111834e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.017631
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.109905e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.646020e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=7.919911e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.569839e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.036677
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=9.660293e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.239018e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.674598e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.960374e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.285013
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=1.459511e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.135627e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.593072e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.251453e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.040279
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=2.043965e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=9.021537e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=2.140772e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.822090e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.155955
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.862695e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=4.806954e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.161824e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=7.568285e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.031386
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=8.496757e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=1.538482e-04 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.177761e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=4.476297e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.578285
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=6.325498e-02 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=6.519176e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.189801e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=3.461018e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.001713
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.319167e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=7.282971e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=3.481833e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=5.219051e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.025574
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=3.251259e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=8.856109e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.937657e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=1.919675e-03 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.210730
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.247834e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.838531e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.558468e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.499749e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.437713
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.112920e-01 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=2.149576e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=1.659175e-01 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=8.325176e-02 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.030779
After optimizer step: bad_parameters=0

======================================================================
ADAPTER GRADIENT STATISTICS
======================================================================
sentiment_au    | norm=7.021659e-04 | grad=72 | missing=0 | nonzero=72
sentiment_uk    | norm=3.179076e-03 | grad=72 | missing=0 | nonzero=72
sarcasm_au      | norm=6.470100e-02 | grad=72 | missing=0 | nonzero=72
sarcasm_uk      | norm=2.245713e-01 | grad=72 | missing=0 | nonzero=72
======================================================================
Before optimizer step: loss=0.036846
After optimizer step: bad_parameters=0
