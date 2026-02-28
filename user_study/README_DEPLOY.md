# Deploying to Hugging Face Spaces

## 1 — One-time: create the feedback dataset repo

Go to https://huggingface.co/new-dataset and create a **private** dataset called
`sae-study-feedback` (or any name you like).

## 2 — Create the Space

1. Go to https://huggingface.co/new-space
2. Pick **Streamlit** as the SDK
3. Set visibility to **Private** (or Public if you want)
4. Clone the empty space repo:

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cd <your-space-name>
```

## 3 — Copy the app files into the space repo

From this repo, copy:

```
user_study/app.py               → app.py
user_study/requirements.txt     → requirements.txt
user_study/SAE.pdf              → SAE.pdf
user_study/data/curated_study_data.jsonl → data/curated_study_data.jsonl
```

> `hypotheses.json` (14 MB) is optional — the sandbox works without it by
> falling back to features already embedded in `curated_study_data.jsonl`.
> If you want the full hypothesis pool, copy it too and remove it from .gitignore.

## 4 — Add Space secrets

In your Space settings → **Secrets**, add:

| Name            | Value                                        |
|-----------------|----------------------------------------------|
| `HF_TOKEN`      | A HF token with **write** access to your dataset repo |
| `FEEDBACK_REPO` | `your-username/sae-study-feedback`            |

## 5 — Push and deploy

```bash
git add .
git commit -m "deploy SAE user study"
git push
```

The Space will build automatically. Every time a participant submits feedback,
`feedback.jsonl` is pushed to your private dataset repo and survives restarts.

## Downloading feedback

From your HF dataset repo page, click **Files** → download `feedback.jsonl`.
Or from your machine:

```bash
huggingface-cli download your-username/sae-study-feedback feedback.jsonl \
  --repo-type dataset --local-dir ./results
```
