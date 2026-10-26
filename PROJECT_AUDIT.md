# Project cleanup and completeness audit

Reviewed on 26 October 2026. The review read and inventoried 130 substantive files in the main HarvestGuard project and all 87 files in its public-upload copy. Generated runtime environments, browser profiles, caches and private credential contents were excluded from source review. The original inventory and checksums are preserved privately with the cleanup backup.

## Current submission

Use the reading order and workflow table in [README](README.md). The selected model is `harvestguard-v3-candidate-1`, trial 6. The current map data are `model_v3_quantity_results/harvestguard_2025_v3_quantity_priorities.csv`. There is one current methodology, submission brief and model record.

The application also includes the project question assistant in `harvestguard_chat.py` and `harvestguard_chat_view.py`, with the current Groq service adapter in `harvestguard_groq.py`. These modules retrieve checked aggregate evidence and public project documents, validate answers and provide the question interface. The fitted loss model, private inputs and recorded quantity/insight exports are unchanged. See `AI_SETUP.md` for private configuration and the assistant's evidence boundaries.

`PUBLIC_RELEASE_MANIFEST.json` lists every approved public file with its size and SHA256. It is regenerated with each release. The application modules, knowledge base, configuration, survey PDFs, cleaning code and audits, both notebooks, selected model, original validation evidence, aggregate quantity and insight outputs, tests and release tooling are retained because they run, reproduce, document or verify this project.

## Missing steps and inputs

| Item | Finding | Required action |
| --- | --- | --- |
| Cleaning and unification code | Present: `source_preprocessing/build_postharvest_model_ready_corrected.py` | Keep it and its audit/requirements files |
| Model-building code | Present: development notebook and root model module; original saved-model module is also retained | Keep both module roles documented |
| Initial 2025 survey merge | Missing from this project: the cleaning script consumes an already-merged `all_seasons.csv` from `all_seasons.zip` | Recover the original code that joined 2025 production/practice modules and combined seasons, with join and duplicate checks |
| Production-bearing input construction | Missing: exact code used to create `postharvest_regression_model_ready_2024_2025_with_production.csv` | Recover that export implementation and identify why ten additional rows were retained; preserve a stable record key and production-unit provenance |
| Private model-ready inputs | Both are present in the main local folder and deliberately excluded from the public upload | Keep privately; provide controlled access only where permitted |
| Three archives expected by the cleaner | None is in the project/submission folder | Supply them privately with `--input-dir`; do not publish farmer records |
| External source material | Sibling `NISR` contains `Dataset.zip`, `Climate.zip`, `Microdata.zip` and extracted 2025 `Microdata/Microdata/all_seasons.csv` | These were left in place. `Microdata.zip` is not the literal `all_seasons.zip` expected by the preserved cleaner; verify packaging rather than assume it is interchangeable |
| New uninspected validation and survey weights | Not supplied; already documented limitations | Keep the existing limits; do not describe sample totals as population estimates |

The two missing source steps are historical provenance gaps. No replacement implementation was invented and no selected model or production estimate was changed.

### Dataset reconciliation needed

| Private input | Rows | 2024 rows | 2025 rows | Columns |
| --- | ---: | ---: | ---: | ---: |
| Original model-ready CSV | 125,412 | 66,104 | 59,308 | 16 |
| Production-bearing CSV used for quantity inference | 125,422 | 66,108 | 59,314 | 17 |

The recorded inference-input SHA256 is `95644d43018bf37d02d8a464f3964088e8c7ee87b79c9ae2347490dc709671c0`; the reviewed private file matches it. It supplies `total_crop_production` in user-confirmed kg. The preserved cleaner exports the 16-column input and omits production. The extra four development rows and six 2025 rows therefore need an original construction record; attaching production by row position would be unjustified. The completed inference run scored 50,475 eligible 2025 profiles, matching 944 saved groups. This does not resolve the raw-data provenance gap.

## Cleanup decisions

- The obsolete root `model_v2/` is removed from the submission. Its two aggregate files that are still used by regression tests now live in `tests/fixtures/legacy_v2/`, with an explanatory README and updated test paths.
- `model_v3_candidate/ranking_detail_candidate.csv` was identical to `ranking_cohorts_2025.csv`. The latter is retained. The original saved-model module remains unchanged, including its historical export implementation; explicit packaging excludes the redundant filename if it is generated again.
- The earlier mixed-version `MODEL_REVIEW.md` is replaced with the selected model record. The original document remains in the private cleanup backup.
- The main-folder cleanup archives older root model binaries/metrics, earlier quantity copies and ZIPs, the duplicate `model_v3_quantity_results (1)/`, the earlier competition plan and obsolete V2 development outputs. These files remain recoverable privately and are excluded from publication.
- The two private unified inputs, private profile predictions, current aggregation verification, survey documentation, current validation baselines, both distinct model modules and both distinct notebooks are retained. Older work that is still used by tests is labelled rather than discarded.
- `real_data_aggregation_checks.json` is now included as aggregate verification evidence. It does not contain individual records.

## Automatic checks

```powershell
python scripts/audit_project.py --verify-manifest
python scripts/audit_project.py --data-dir "PATH_TO_PRIVATE_HARVESTGUARD" --raw-dir "PATH_TO_SOURCE_ARCHIVES" --require-reproducibility --json .runtime/project_audit.json
```

The first command checks all approved submission files, required workflow artifacts, Python/JSON validity, model/export hashes and the public manifest. It also prints the unresolved source steps. The second inspects private CSV row counts, verifies the recorded scoring-input hash and reports missing source archives. It returns exit code 2 while full reproduction remains incomplete, even when deployment artifacts pass. The two suggested missing-source filenames are placeholders for recovered original code, not claims that those were its historical filenames.

A successful file check is evidence of artifact consistency, not scientific validation or proof that a newly added source file recreates a historical result. Recovering source steps must include review against the recorded inputs and outputs.

## Private history and GitHub

Cleanup backups, original inventories and removed files are stored under ignored `.runtime/cleanup/`. They do not enter the release. Publish the contents of the newly built release directory, which has its own manifest. The release allow-list is explicit: newly added files are not automatically uploaded because they happen to sit in a model, preprocessing or test folder.

This report describes local file decisions. A GitHub update must be separately verified against the repository after publication; a local cleanup alone does not change GitHub or its deployed application.
