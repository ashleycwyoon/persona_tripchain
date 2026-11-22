# Vendored upstream code: tripchain_1015

These files are byte-identical copies from the upstream `tripchain_1015` project
(commit `637b7b01ee5fd57defd1b99f9198f8488f0f0ed4`), kept in their original
relative locations. PersonaTC builds on them for the KTDB preprocessing, the
Teacher, the Student SFT example builders and the QLoRA trainers.

**Do not edit these files.** Keeping them identical to their source is what lets
the data and the Teacher labels be reproduced exactly. They are excluded from
linting and formatting (`pyproject.toml`).

| File | Role | Used by |
|---|---|---|
| `scripts/prepare_ktdb_training_data.py` | KTDB 2021 preprocessing: Seoul residents, person-level 90 / 10 split, demographics, activity sequences, spatial targets, location codebook | run directly (tutorial step 1); its helper functions are imported by `scripts/prepare_residence_data.py`, `scripts/rebuild_ground_truth_v2.py` and `scripts/build_teacher_relabel_inputs.py` |
| `scripts/ktdb_admin_registry.py` | Reads the administrative-code workbook; builds the location codebook | the scripts above |
| `scripts/korean_admin_english.py` | Deterministic English rendering of Korean administrative names | the two files above |
| `testing/paths.py` | Data and output locations of the upstream project (`$DATA_HOME/tripchain_1015`) | the upstream scripts |
| `persona-rationale-multi-country-qwen35/prompts/rationalizer_prompt.md` | The Teacher prompt | `src/data/ktdb_two_stage_teacher.py` |
| `persona-rationale-multi-country-qwen35/src/data/ktdb_two_stage_teacher.py` | Teacher inputs, prompt construction, answer parsing and validation | the Teacher scripts, the example builders |
| `persona-rationale-multi-country-qwen35/scripts/09d_run_ktdb_teacher_2gpu.py` | Launches one Teacher worker per GPU and shows progress | run directly (tutorial steps 2 and 5) |
| `persona-rationale-multi-country-qwen35/scripts/09c_abduce_ktdb_two_stage.py` | The Teacher worker (vLLM) | `09d_run_ktdb_teacher_2gpu.py` |
| `persona-rationale-multi-country-qwen35/src/data/ktdb_student1.py`, `ktdb_student2.py`, `ktdb_student_common.py` | Build the Student SFT chat examples from the Teacher labels | `scripts/build_sft_v2.py` |
| `persona-rationale-multi-country-qwen35/scripts/20a_train_ktdb_student1.py` | Student1 QLoRA trainer | `scripts/train_student1.py` |
| `persona-rationale-multi-country-qwen35/scripts/20b_train_ktdb_student2.py` | Student2 QLoRA trainer | `scripts/train_student2_tripmode.py` |
| `persona-rationale-multi-country-qwen35/templates/qwen35_text_assistant_only.jinja` | Chat template that marks the assistant turn for the loss mask | the two trainers |
| `persona-rationale-multi-country-qwen35/src/paths.py` | Data and output locations of the Teacher stage (`$OUTPUT_HOME/tripchain_1015/persona-rationale-multi-country-qwen35`) | the files above |

Notes:

- The upstream code keeps its own data and output folders, `tripchain_1015/` under
  `$DATA_HOME` and `$OUTPUT_HOME`; the PersonaTC configs read from them.
- The Teacher scripts default `--download-dir` to a path on the original author's
  machine. Always pass `--download-dir` (for example `"$HF_HUB_CACHE"`).
- Korean text in these files is data: raw survey column values, file names and
  keyword rules for free-text survey answers.
- To use another checkout of the upstream project, set `TRIPCHAIN_1015_CODE`
  to its path (see `tripchain/paths.py`).
