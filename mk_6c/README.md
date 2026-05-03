# mk_6c — Post-hoc Ensemble Combinations

Reads mk_6b's saved test probabilities, produces two new ensemble candidates
without any new fitting.

## Run

In the docker container, from repo root:

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6c info539-mk1 bash
```

Inside:

```bash
python post_hoc.py 2>&1 | tee run.log
```

~10 seconds. No fitting.

## Submissions produced

```
mk_6c/submissions/
├── mk_6c_ensemble_4way_uniform.csv     # (mk_2 + mk_6 + mk_7 + mk_9_53) / 4
└── mk_6c_ensemble_4way_mk6_dom.csv     # 0.15/0.40/0.15/0.30 weighted
```

## Prerequisites

Requires these files from mk_6b/models/:
- mk_6b_full_data_test_proba.npy        (Push 1)
- mk_6b_mk2_full_test_proba.npy         (Push 3)
- mk_6b_mk7_full_test_proba.npy         (Push 3)
- mk_6b_mk9_53_full_test_proba.npy      (Push 4)
