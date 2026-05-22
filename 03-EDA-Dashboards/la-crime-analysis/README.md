# LA Crime Dashboard — interactive analytics from raw LAPD data

> Streamlit + Plotly dashboard built on top of an earlier exploratory notebook. The notebook answered three fixed questions; the dashboard turns the same data into a tool that lets a user explore any slice interactively.

![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![Plotly](https://img.shields.io/badge/Plotly-5.24-blue)
![Python](https://img.shields.io/badge/python-3.10+-yellow)

---

## Why this exists (and why it's not just a notebook)

The original `notebook.ipynb` answered three pre-specified questions about LA crime data (peak hour, top night-crime area, victim age distribution). That's fine as a learning exercise, but it has two issues for a portfolio:

1. **A reader has to run it to interact** — no deployed artifact.
2. **It answers fixed questions** — a stakeholder asking "what about *just* the Central area in 2023?" has to rewrite code.

This dashboard fixes both:
- Deployed as a public Streamlit app (HF Spaces).
- Interactive filters for date range, area, and crime type. All charts respond live.
- Multi-page layout with separate views for time, geography, demographics, and trends.

---

## Quickstart

### Run locally

```bash
git clone https://github.com/AhmedMostafa167/Projects.git
cd Projects/03-EDA-Dashboards/la-crime-analysis
pip install -e ".[dev]"
streamlit run app.py
# → http://localhost:8501
```

### Run with Docker

```bash
docker build -t la-crime-dashboard .
docker run -p 8501:8501 la-crime-dashboard
```

### Deploy to HF Spaces (one command)

```bash
bash scripts/deploy_hf_space.sh <your-hf-username> la-crime-dashboard
```

---

## What's in the dashboard

| Page | What it shows | Why it matters |
|---|---|---|
| **Overview** | KPI cards (total crimes, areas, date range, top type) + monthly time series | The first thing anyone wants — "is the trend going up or down?" |
| **Geography** | Crimes-per-area bar + crime-type-by-area heatmap | Where to allocate patrols |
| **Time patterns** | Hour-of-day + day-of-week + night vs. day | When to staff up |
| **Demographics** | Victim age groups, sex, descent | Helps identify under-served populations (with caveats — see "Limitations") |

All pages share a sidebar that filters the data globally: date range, area selector (multi-select), crime-type search.

---

## Project layout

```
la-crime-analysis/
├── app.py                # Streamlit entry point (Overview page)
├── pages/                # Streamlit multipage
│   ├── 1_Geography.py
│   ├── 2_Time_Patterns.py
│   └── 3_Demographics.py
├── src/
│   ├── data.py           # Load, clean, cache the raw CSV
│   ├── filters.py        # Sidebar filter widgets
│   └── viz.py            # Reusable Plotly chart functions
├── crimes.csv            # ~27 MB, 185k rows
├── notebook.ipynb        # Original exploratory notebook (kept for reference)
├── la_skyline.jpg
├── tests/
│   └── test_data.py      # Data-loading + filter tests
├── docs/
│   ├── DEPLOYMENT.md
│   └── INTERVIEW_NOTES.md
├── scripts/
│   └── deploy_hf_space.sh
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Tech stack

- **Streamlit 1.40** — interactive multipage UI
- **Plotly 5.24** — interactive charts (hover, zoom, export)
- **pandas 2.2** — data wrangling
- **st.cache_data** — load + parse the 27 MB CSV once per session

---

## What I'd add next

- Geocode the `LOCATION` column → real choropleth map (the raw data has street addresses, not lat/lon).
- Forecast next month's volume (Prophet or simple SARIMA).
- Cluster areas by crime composition (k-means on per-area type proportions).
- A "compare two periods" view.

---

## Limitations / caveats

- **Demographic data is reported, not predicted.** This dashboard does *not* predict anything about who commits crime — it visualizes who is reported as a *victim*. Confusing the two would be harmful.
- **Reporting bias.** Some crime categories are under-reported (e.g., domestic violence). A spike in reports of a category may reflect a policy change in reporting, not a real change in incidence.
- **Date `2020-01-27 reported 2023-02-27`-type rows exist.** The data has crimes that were reported years after they occurred. The dashboard filters on `DATE OCC` (occurrence date) by default; the raw data is preserved if you ever want to filter on `Date Rptd`.

See [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) for discussion points.
