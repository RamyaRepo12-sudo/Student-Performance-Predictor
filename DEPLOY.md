# Free deployment

## Streamlit Community Cloud (easiest)
1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io -> New app -> pick the repo, branch `main`, main file `app.py`.
3. Deploy. If `models/` is missing it trains itself on first load (about a minute).

## Render (free web service)
New -> Blueprint -> connect the repo. `render.yaml` is already configured.
Free instances sleep when idle, so the first request after a pause is slow.

## Hugging Face Spaces
Create a Space with the Streamlit SDK and upload the files. Put this block at the very top of README.md:

    ---
    title: Student Performance Predictor
    sdk: streamlit
    app_file: app.py
    ---

Tip: commit `models/` so the app starts instantly instead of retraining.
