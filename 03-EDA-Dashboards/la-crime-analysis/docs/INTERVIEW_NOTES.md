# Interview talking points — LA Crime Dashboard

Likely questions and 30-second answers. This project sits in the "data analysis + product thinking" lane — useful for showing you can take raw data and turn it into something a non-technical stakeholder can use.

---

## 1. "Walk me through the project."

> I started with a Jupyter notebook that answered three specific questions about LA crime data. The issue with that as a portfolio piece is that a reader has to run the code to get any value, and the questions are fixed. So I rebuilt it as an interactive Streamlit dashboard. It's a multi-page app — Overview, Geography, Time, Demographics — with shared sidebar filters for date range, area, and crime type. Every chart is Plotly so it's hoverable and zoomable. The data layer caches the 27 MB CSV so filters feel instant. It's deployed to a Hugging Face Space.

---

## 2. "Why Streamlit and not Dash / a custom React frontend?"

> Streamlit is the right tool when the question is 'how do I get a working interactive UI on top of a Python data pipeline as fast as possible'. The cost is that you give up fine-grained UI control. For a portfolio piece or an internal data app, that trade-off is correct. If this were a customer-facing product or needed custom auth/embedding, I'd reach for Dash or a React + FastAPI split.

---

## 3. "Why `st.cache_data`?"

> Without caching, Streamlit re-runs the entire script top-to-bottom on every interaction — every filter change re-parses the 27 MB CSV, which makes filtering feel sluggish. `st.cache_data` memoizes the result of `load_and_prepare()` keyed on the function arguments, so the parse happens once per session and subsequent filter changes just slice an already-loaded DataFrame. There's also `st.cache_resource` for things like ML models — same idea but for objects you want to share across sessions on the same server.

---

## 4. "How is the multipage structure organized?"

> Streamlit picks up any `.py` file in `pages/` and renders it in the left nav, sorted by filename. I prefixed them with numbers (`1_Geography.py`, `2_Time_Patterns.py`, `3_Demographics.py`) to control order. Each page is self-contained: it loads the data, applies the sidebar filters, and renders its charts. The shared bits — data loading, filter widgets, chart functions — live in `src/` so the page files stay short.

---

## 5. "What's the data layer doing?"

> Three responsibilities. `load_raw()` parses the CSV. `add_features()` derives the columns the dashboard needs — hour from `TIME OCC`, day-of-week from `DATE OCC`, age groups, friendly descent labels. `apply_filters()` is the pure function the sidebar uses to slice based on user selections. I deliberately split the parsing and the feature engineering — feature engineering is the part I'd want to test, and a pure function is easy to test.

---

## 6. "How would you scale this to a much bigger dataset?"

> Three changes. One, move the storage from a CSV in the repo to Parquet on object storage with PyArrow — much faster to load and supports column pruning. Two, push aggregations down to a database (DuckDB in-process, or Postgres remote) so the app only fetches pre-aggregated data. Three, replace the eager `load_and_prepare` with a query-on-demand layer — for, say, 100 M rows you don't want everything in memory; you want each page to issue a SQL query for its specific slice.

---

## 7. "Why no map?"

> The raw data has street addresses (`LOCATION`) but not lat/lon coordinates. A real choropleth or pin-map needs geocoding, which would be a project in itself — either batch through a paid geocoding API for ~190k addresses, or use the publicly available LAPD area shapefiles to draw the 21 patrol divisions. I called this out in the README as the next step rather than fake it.

---

## 8. "Tell me about the demographic page — anything sensitive there?"

> Yes, and I'm explicit about it on the page itself. The data is **reported victim** demographics, not predicted offender demographics — those are very different things, and confusing them would be harmful. I added a warning banner at the top of the demographics page that calls this out, and the page README repeats it. There's also well-documented reporting bias in this kind of data — some categories are systematically under-reported — and I'd want to discuss that with a stakeholder before drawing conclusions from any single chart.

---

## 9. "What's the deployment story?"

> One command. `bash scripts/deploy_hf_space.sh <username> la-crime-dashboard` creates a Docker-SDK Hugging Face Space, swaps in the Space-formatted README header, and uploads the project. The Dockerfile is a thin wrapper around `streamlit run app.py`. HF Spaces' free tier handles it because the only resource cost is loading the 27 MB CSV into memory — no models, no GPU.

---

## 10. "What's the original notebook still doing in there?"

> Honest answer — it's preserved for context. It's the starting point of this project, and keeping it shows the *evolution* from "I learned pandas and answered three questions" to "I built a stakeholder-usable tool over the same data". If I were submitting to a strict code review I'd archive it under `notebooks/` or remove it; for a portfolio it tells the story.

---

## Quick facts to memorize

- **Framework**: Streamlit 1.40 (multi-page via `pages/` convention)
- **Charts**: Plotly Express + Plotly Graph Objects
- **Caching**: `@st.cache_data` (per-session memoization)
- **Data size**: ~185k rows, 27 MB CSV
- **Source**: LAPD via LA Open Data
- **Deployment**: HF Spaces, Docker SDK, free tier
- **Pages**: Overview, Geography, Time Patterns, Demographics

---

## If they ask "show me one specific design decision"

Open `src/data.py:add_features`. Point at the pure function that takes the raw DataFrame and returns one with derived columns. "Separating this from the loading function means I can unit-test feature engineering without a CSV on disk, and I can swap the storage layer (CSV → Parquet → SQL) without touching feature logic."
