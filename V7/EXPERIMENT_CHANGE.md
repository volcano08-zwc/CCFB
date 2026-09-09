# V7: V5 + AdamW, Warmup, and Cosine Decay

V7 is an independent branch copied from V5. It preserves V5 data,
source-subject perturbation, model, loss, batch size, 100 epochs, seeds, and
LOSO split. Only optimization changes:

```text
optimizer      = AdamW
base_lr        = 0.001
weight_decay   = 0.0001
warmup_epochs  = 5
decay          = cosine to zero at the final update
```

Verification:

```text
python smoke_test_optimizer_schedule.py
python smoke_test_source_subject_perturbation.py
python DBConformer_LOSO.py 0
```
