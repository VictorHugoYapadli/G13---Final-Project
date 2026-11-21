# Inspec data for the TextRank reproduction

## Default experiment: all 500 test abstracts

The default command uses `inspec_paper_test.jsonl`, containing all 500 documents from the MIDAS Inspec test split and their original uncontrolled keywords. `inspec_paper_metadata.json` records source hashes, IDs, label validation, and the output hash. The CLI defaults select `configs/experiment_paper_aligned.json` and this complete dataset.

Build or rebuild this default input with:

```powershell
python data/prepare_inspec_paper.py
```

The script requires the two source files below, checks all 500 document IDs and normalized labels, and stops if they do not match.

## Optional smaller experiment: fixed sample of 150

`fixed_subset.jsonl` is a deterministic sample of 150 documents from those same 500 abstracts, selected without replacement with seed 2026. To regenerate it, run:

```powershell
python data/prepare_inspec.py --count 150 --seed 2026
```

This sample is used only when explicitly selecting `configs/experiment_150.json` and `data/fixed_subset.jsonl`.

Sources:

- Abstracts: `midas/inspec`, `generation` test split at revision `9617780e99705df88a4cb239174c86b9d8d8300f`: <https://huggingface.co/datasets/midas/inspec>
- Original uncontrolled reference file: `boudinfl/ake-datasets`, `datasets/Inspec/references/test.uncontr.json`: <https://github.com/boudinfl/ake-datasets>

## License and redistribution

The [MIDAS dataset card](https://huggingface.co/datasets/midas/inspec) identifies the original Inspec source but does not state a dataset license. The [AKE-Datasets repository](https://github.com/boudinfl/ake-datasets) is marked Apache-2.0, but that repository license does not establish redistribution rights for the underlying Inspec abstracts or annotations. Treat the Inspec content as **not cleared for redistribution** unless the original rights holder or course staff confirms otherwise. The generated JSONL is a copy/derivative of that content and has the same restriction. For a submission archive, include this README, the preparation script, source URLs/revisions and hashes, but omit `source/`, `inspec_paper_test.jsonl`, and other copies of Inspec text/labels unless permission is confirmed. Instructors can regenerate locally from the documented sources where access is permitted.

The MIDAS file labels keyphrases as present/extractive and absent/abstractive rather than exposing the controlled/uncontrolled field. `prepare_inspec_paper.py` therefore checks the two sources before use: all 500 IDs match, and for every ID the normalized MIDAS present+absent phrase set exactly matches the original uncontrolled reference set. The generated file uses the original uncontrolled annotations. The script stops if either validation fails.

The MIDAS file is pinned to the Hugging Face revision above. The AKE reference URL currently follows the repository's `master` branch; the exact retrieved bytes are fixed by the SHA-256 stored in `inspec_paper_metadata.json`, so later source changes are detected by comparing hashes.

To obtain the source files in PowerShell:

```powershell
New-Item -ItemType Directory -Force data\source | Out-Null
Invoke-WebRequest -Uri "https://huggingface.co/datasets/midas/inspec/resolve/9617780e99705df88a4cb239174c86b9d8d8300f/test.jsonl" -OutFile data\source\inspec_test.jsonl
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/boudinfl/ake-datasets/master/datasets/Inspec/references/test.uncontr.json" -OutFile data\source\inspec_test_uncontrolled.json
```

Then build and validate the actual experiment input:

```powershell
python data/prepare_inspec_paper.py
```

This writes `data/inspec_paper_test.jsonl` (500 rows, original source order) and `data/inspec_paper_metadata.json` with both source hashes, all document IDs, label validation, and output hash. No sampling seed applies to this run.

## Distribution

The MIDAS loader and AKE-Datasets repository provide provenance, but the original Inspec abstracts and annotations may be subject to separate terms. Do not include `source/`, generated JSONL data, or reference files in the submission archive unless redistribution is confirmed as allowed. Keep source URLs, revisions, hashes, sample IDs, and preparation code so the experiment can be reconstructed.
