# Membership Inference Attack — TML26 Task 1

This repository contains our implementation of a Membership Inference Attack (MIA)
against a pretrained ResNet-18 image classifier, submitted for the
Trustworthy Machine Learning 2026 course at Saarland University / CISPA.

---

## How to Recreate the Best Leaderboard Result

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Download the data

```bash
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
```

### 3. Set up your API key

Create a file named `API_KEY.txt` in the same folder as the script and paste
your personal API key inside it (no quotes, no newlines):

```
your_api_key_here
```

### 4. Place all files in the same directory

```
your_folder/
├── task_template.py      ← main attack script
├── pub.pt
├── priv.pt
├── model.pt
└── API_KEY.txt
```

### 5. Run the attack

```bash
python task_template.py
```

The script will automatically:
- Load the target ResNet-18 model and both datasets
- Train 32 shadow models on the combined pub+priv pool
- Extract LiRA features (phi scores, losses, z-scores) for every sample
- Train attack classifiers (Logistic Regression, MLP) with 5-fold CV
- Select the best attack variant based on pub OOF TPR@5%FPR
- Save `submission.csv` and submit it to the leaderboard

---

## Running on HPC (HTCondor)

Create a file `mia.sub`:

```condor
executable   = /usr/bin/python3
arguments    = task_template.py

transfer_input_files  = task_template.py, pub.pt, priv.pt, model.pt, API_KEY.txt
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = submission.csv

log    = job_id.log
output = job_id.out
error  = job_id.err

queue
```

Submit with:

```bash
condor_submit mia.sub
```

Monitor with:

```bash
condor_q
tail -f job_id.out
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `N_SHADOW` | 32 | Number of shadow models |
| `N_IN_PER_POINT` | 6 | IN shadows per sample |
| `SHADOW_EPOCHS` | 60 | Training epochs per shadow |
| `N_AUG` | 4 | Test-time augmentation views |
| Attack selection | auto | Best OOF TPR@5%FPR on pub |

---

## Expected Output

```
ASSIGNMENT METRIC  |  Score = TPR@5%FPR
Selected attack: online_S_dual
Score (TPR@5%FPR), pub OOF: ~0.069
```

The script prints a full diagnostic including per-candidate TPR@5%FPR before
submitting the best one automatically.
