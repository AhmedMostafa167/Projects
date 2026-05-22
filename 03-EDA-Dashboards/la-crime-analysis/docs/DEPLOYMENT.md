# Deployment

## Hugging Face Spaces (recommended)

```bash
pip install -U huggingface_hub
huggingface-cli login
bash scripts/deploy_hf_space.sh <your-hf-username> la-crime-dashboard
```

The script creates a Docker-SDK Space, swaps in the Space-formatted README, and uploads the project. The Space will be live at `https://huggingface.co/spaces/<your-hf-username>/la-crime-dashboard` within ~3 minutes.

Free tier is sufficient — the only memory cost is the 27 MB CSV.

---

## Streamlit Community Cloud (alternative)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and connect your GitHub.
3. Pick the repo, set the main file to `03-EDA-Dashboards/la-crime-analysis/app.py`, and Python version to 3.11.
4. Deploy.

---

## Local

```bash
pip install -e ".[dev]"
streamlit run app.py
# → http://localhost:8501
```

---

## Docker

```bash
docker build -t la-crime-dashboard .
docker run -p 8501:8501 la-crime-dashboard
```

---

## Notes

- The CSV is committed to the repo (~27 MB). If you fork and your platform has size limits, swap to Parquet (smaller, faster to load) and update `DATA_PATH` accordingly.
- The Dockerfile sets `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` to disable Streamlit's anonymous telemetry — appropriate for a public demo.
