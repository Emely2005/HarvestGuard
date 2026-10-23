# Deploy HarvestGuard to Streamlit Community Cloud

The application runs directly from aggregate exports. It does not load the training model into memory or require raw farmer CSVs. Its model file is retained for provenance checks. The root `requirements.txt` includes the tested dashboard dependencies from `requirements-web.txt`; model-development dependencies remain separate.

## 1. Publish the reviewed GitHub files

Run `python scripts/build_public_release.py`. The builder prints the release directory and ZIP path and records them in `.runtime/public_release/latest.json`. Publish the **contents of that release directory** to the root of the new public `HarvestGuard` repository on branch `main`. Do not upload the ZIP itself as the application or upload the entire working directory.

The release contains the application modules, `.streamlit/config.toml`, knowledge base, aggregate exports, model provenance, methodology, preprocessing code, survey documentation and tests. It excludes private settings, credentials, runtime caches, individual profile predictions and raw survey/climate data. Notebook outputs are cleared in the release copy; local originals are retained.

If publishing through GitHub's website, create the repository, choose **Add file > Upload files**, upload the extracted release contents while preserving directories, and commit to `main`. Include the hidden `.streamlit` folder, containing the public theme and placeholder example only. A connected GitHub plugin can publish the same reviewed files directly.

## 2. Create the Streamlit app

Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) using the GitHub account that owns the repository. Choose **Create app**, then **Yup, I have an app**, and use these settings:

| Field | Value |
| --- | --- |
| Repository | Your GitHub account's `HarvestGuard` repository |
| Branch | `main` |
| Main file path | `harvestguard_app.py` |
| Python version, under Advanced settings | `3.14`, matching the locally verified web runtime |
| App URL | Choose an available subdomain, such as `harvestguard-rwanda` |

The root requirements file is detected automatically. No custom start command, model fitting or system-package installation is needed. See the official [deployment workflow](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## 3. Configure AI privately

In **Advanced settings > Secrets**, paste the contents of your private local `.streamlit/secrets.toml`. Its format is:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_MODEL = "gemini-3.1-flash-lite"
```

Replace the placeholder only inside Streamlit's private Secrets field. Never commit the actual key or paste it into an issue or chat. Save the settings and click **Deploy**. You can later update these values in the app's settings. See [Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

Without a key, the map, factual findings, evidence-backed actions and downloads remain usable. New Gemini explanations require valid project access and available quota. Online generation is enabled by default; `HARVESTGUARD_AI_ENABLED=0` in the server environment disables it for isolated local review. A new cloud instance starts without the local AI cache and uses the same bounded request handling as the local application.

## 4. Verify and share

After the build succeeds, open the actual `streamlit.app` address returned by Streamlit. Check Map and Insights, change the crop, season and district, open Metadata and Methodology, and download a survey PDF and district ranking. Check AI generation after configuring the secret. Open the URL in a private browser window to confirm other people can view it. Share that confirmed URL in the competition submission and repository README.

The public URL exists only after Streamlit confirms deployment. A proposed subdomain and a localhost address are not deployed links. If a build fails, use the app's **Manage app** logs to find the failing dependency or missing file; keep the displayed versions and directory structure from the release intact.

## Local start

Install `requirements.txt`, then run `streamlit run harvestguard_app.py`. Open the URL printed by Streamlit. The health endpoint is `/_stcore/health`. Model reproduction uses the separate pins in `model_v3_candidate/requirements_model_v3.txt`; `requirements-dev.txt` adds test and analysis tools to the web runtime.

## Optional Docker or Render hosting

The included Dockerfile and `render.yaml` remain available for other hosting choices. Build with `docker build -t harvestguard .` and run with `docker run --rm -p 8501:8501 -e GEMINI_API_KEY -e GEMINI_MODEL harvestguard`. Export the key privately in the invoking environment. The image listens on `PORT` when provided, otherwise 8501, and runs as an unprivileged user. For Render, create a Blueprint from `render.yaml` and set the key privately in its environment settings.
