# Membership inference Attach - Trustworthy Machine Learning Course 

**Repository:** https://github.com/SaraZahediMovahed/MIA  

## Requriements

- **Python 3.10+** (3.11 works well).
- **Task data** next to the script (not in this repo): `pub.pt`, `priv.pt`, and `model.pt`. Obtain them from the course materials or download location for the assignment.
- **`API_KEY.txt`** in the folder you run from (same directory as `task_template.py`): one line, your course API key. The script reads this at startup and uses it again when uploading `submission.csv`.

## Setup

```bash
git clone https://github.com/SaraZahediMovahed/MIA.git
cd MIA
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Put pub.pt, priv.pt, model.pt here and add API_KEY.txt
```

## Run (full pipeline → best submission)

From the directory that contains `task_template.py` and the three `.pt` files:

```bash
python task_template.py
```

The script:

1. Loads the target ResNet-18 and applies the fixed normalization from training.  
2. Trains **32 shadow models** on the combined public+private pool with the documented schedule (SGD, cosine schedule, warmup, label smoothing, flip + reflect-padded crop).  
3. Builds **LiRA-style features** (shadow IN/OUT statistics, z-scores, online likelihood-style terms, class-conditional cues) and trains several **light attack models** with **5-fold stratified CV** on the public labels.  
4. **Selects** the candidate with the best **out-of-fold TPR@5%FPR**, applying the same small “robust ensemble” guards coded in the file when appropriate.  
5. Writes **`submission.csv`** (rank-normalized scores for private IDs) and **POSTs it** to the grading server.

Expect a long run on CPU; a GPU speeds up shadow training and feature extraction.

**Our report’s leaderboard number** (TPR@FPR=0.05 ≈ **0.06356**) comes from this exact script and data; your rerun should match up to environment noise unless the platform or data revision changes.
