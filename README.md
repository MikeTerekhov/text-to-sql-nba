# SQL Generation for NBA Stats

Natural-language-to-SQL generation over an NBA statistics database, built by fine-tuning [DeepSeek-Coder-1.3B-Instruct](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct) with LoRA and augmenting it with schema-aware retrieval (RAG).

Final project for CSCI-544 (Applied Natural Language Processing) at USC.

> Ask a question like *"What is the most points the Los Angeles Lakers have ever scored at home?"* and get back a runnable SQLite query:
> ```sql
> SELECT MAX(pts_home) FROM game WHERE team_name_home = 'Los Angeles Lakers';
> ```

## Overview

The task is text-to-SQL over three tables of NBA game data (`team`, `game`, `other_stats`, sourced from Kaggle's NBA database dump). We compare three approaches:

1. **Zero-shot prompting** — the base instruct model with the full schema and a few example queries stuffed into the prompt.
2. **LoRA fine-tuning** — parameter-efficient fine-tuning of the base model on ~1,000 hand-curated and augmented natural-language/SQL pairs.
3. **Retrieval-augmented prompting (RAG)** — instead of injecting the entire schema into every prompt, relevant schema/table descriptions are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`) and retrieved by cosine similarity against the user's question, then injected into a shorter, more focused prompt.

Each generated query is checked for (a) SQL validity, (b) exact match against a reference query, and (c) whether executing it against `nba-data/nba.sqlite` returns the same result as the reference — since many correct queries aren't textually identical.

## Repository structure

```
.
├── demo.py                    # Zero-shot prompting demo against the base model
├── demo_rag.py                 # RAG-augmented prompting demo
├── rag_metadata.py             # SQLMetadataRetriever: embeds & retrieves schema docs
├── test_rag.py                 # Batch evaluation of the RAG pipeline over train-data
│
├── database_setup.ipynb        # Sanity checks against nba.sqlite
├── augment_data.ipynb          # Expands the seed dataset (team name / season substitution)
├── finetune_model.ipynb        # LoRA fine-tuning of deepseek-coder-1.3b-instruct
├── test_pretrained.ipynb       # Evaluation of the zero-shot baseline
├── test_finetuned.ipynb        # Evaluation of the fine-tuned model
│
├── train-data/                 # NL question -> SQL query -> expected result pairs (TSV)
├── nba-data/nba.sqlite         # NBA stats database (team, game, other_stats, ...)
├── deepseek-coder-1.3b-instruct/  # Base model weights
└── fine-tuned-model/           # LoRA checkpoints
```

> **Note:** the base model weights, LoRA checkpoints, and `nba.sqlite` (several GB total) aren't stored in this GitHub repo. Grab them from the [Hugging Face Hub](https://huggingface.co/USC-Applied-NLP-Group/SQL-Generation) to run the notebooks/scripts locally.

## Data

`train-data/sql_train.tsv` holds `natural_query`, `sql_query`, and `result` columns. It's split into overlapping subsets used for targeted evaluation (see [train-data/ReadMe.md](train-data/ReadMe.md)):

| File | Contents |
|---|---|
| `queries_from_game.tsv` / `queries_from_team.tsv` / `queries_from_other_stats.tsv` | Queries grouped by which table they start `FROM` |
| `with_join.tsv` / `without_join.tsv` | Queries that do/don't require a `JOIN` |
| `less_than_90.tsv` | Shorter queries (< 90 characters) |

`augment_data.ipynb` expands the hand-written seed set by substituting team names and seasons across queries to multiply coverage without hand-writing every combination.

## Fine-tuning

`finetune_model.ipynb` wraps `deepseek-coder-1.3b-instruct` with a LoRA adapter targeting the attention projections:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

and trains on the natural-language/SQL pairs with a standard causal-LM cross-entropy objective. Checkpoints for a few LoRA ranks are saved under `fine-tuned-model/`.

## RAG schema retrieval

Rather than always injecting the full multi-table schema (which eats context and adds noise), `rag_metadata.py`'s `SQLMetadataRetriever` embeds per-table schema descriptions and the team-name/abbreviation lookup as documents, embeds the user's question with the same encoder, and returns the top-k most relevant documents by cosine similarity to build a targeted prompt. See `demo_rag.py` for the end-to-end flow and `test_rag.py` for batch evaluation.

## Evaluation

Each notebook/script reports three metrics per query, aggregated over a dataset (or subset):

- **Percent valid** — the generated text parses as a syntactically valid SQL statement
- **Percent SQLite matched** — the generated query is textually identical to the reference
- **Percent result matched** — executing the generated query against `nba.sqlite` returns the same result as the reference (the more meaningful metric, since equivalent queries can be phrased differently)

Run `test_pretrained.ipynb`, `test_finetuned.ipynb`, or `test_rag.py` to reproduce numbers for each approach on any of the `train-data/` subsets.

## Usage

```bash
pip install -r requirements.txt
```

Zero-shot baseline:

```bash
python demo.py
```

RAG-augmented prompting:

```bash
python demo_rag.py
```

Both scripts load `deepseek-coder-1.3b-instruct` from the local `deepseek-coder-1.3b-instruct/` directory — edit the `user_question` / `input_text` variable in each script to try your own question. To use the fine-tuned model instead, point `AutoModelForCausalLM.from_pretrained(...)` at `fine-tuned-model/` and load the LoRA adapter with `peft`.

## Team

Built by the USC Applied NLP Group. Model and data also hosted on the [Hugging Face Hub](https://huggingface.co/USC-Applied-NLP-Group/SQL-Generation).
