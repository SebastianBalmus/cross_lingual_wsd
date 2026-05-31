# Cross-Lingual Word Sense Disambiguation with LLMs

This repository accompanies the paper **Cross-Lingual Word Sense Disambiguation Remains Challenging for Large Language Models**. It contains the English-Romanian sense-aligned dataset and evaluation scripts used to test whether instruction-tuned multilingual LLMs can select fine-grained WordNet senses from natural-language definitions.

The goal is not to introduce a new WSD model. The repository provides a controlled diagnostic benchmark for comparing monolingual and cross-lingual lexical semantic alignment in LLMs.

## Paper Idea

Word Sense Disambiguation (WSD) is framed as a multiple-choice definition-selection task. Given a target lemma, part of speech, sentence, and candidate sense definitions, a model must output the label of the sense that best matches the target word in context.

The paper studies two settings:

- **Monolingual WSD**: the input sentence and candidate definitions are in the same language, evaluated separately for English and Romanian.
- **Cross-lingual WSD**: the input sentence and candidate definitions are in different languages, evaluated in both EN->RO and RO->EN directions.

The benchmark is built around shared WordNet/RoWordNet synset identifiers, so English and Romanian examples are aligned by sense rather than only by sentence-level translation equivalence.

## Dataset

The main dataset is:

```text
data/parallel_sentences.csv
```

Each row contains an English-Romanian sentence pair grounded in a shared lexical sense, with lemma, part-of-speech, synset, gloss, and sentence fields for both languages. The paper reports 2,771 sentence pairs spanning 1,197 English lemmas, 1,203 Romanian lemmas, and 2,771 distinct senses.

The `models/` directory contains UDPipe models used during dataset construction and validation, especially for checking Romanian target lemma realization under inflection.

## Repository Layout

```text
.
|-- wsd.py                  # Monolingual English/Romanian WSD evaluation
|-- cross_lingual_wsd.py    # EN->RO and RO->EN cross-lingual WSD evaluation
|-- run_benchmark.sh        # Convenience runner for baselines and configured HF models
|-- data/
|   `-- parallel_sentences.csv
|-- models/
|   |-- english-ewt.udpipe
|   `-- romanian-rrt.udpipe
`-- requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Hugging Face models that require gated access, authenticate with Hugging Face before running the benchmark.

## Baselines and Models

The scripts support three non-generative baselines:

- `random`: uniform random choice over candidate senses.
- `lesk`: lexical-overlap baseline between the sentence and candidate glosses.
- `gloss_similarity`: multilingual sentence-embedding similarity between the sentence and candidate glosses.

Any Hugging Face text-generation model id or local model path can also be passed through `--model` / `-m`. The paper evaluates instruction-tuned multilingual models including LLaMA 3.1, Mistral, Qwen 2.5, Gemma 3, and Salamandra.

## Usage

Run monolingual WSD with a lightweight baseline:

```bash
python wsd.py \
  -m lesk \
  -r results/wsd_lesk \
  -d data/parallel_sentences.csv
```

Run cross-lingual WSD:

```bash
python cross_lingual_wsd.py \
  -m lesk \
  -r results/cross_lingual_wsd_lesk \
  -d data/parallel_sentences.csv
```

Run with shuffled candidate definitions, used in the paper to test sensitivity to definition order:

```bash
python wsd.py \
  -m gloss_similarity \
  -r results/wsd_gloss_similarity_shuffled \
  -d data/parallel_sentences.csv \
  --shuffle_senses
```

Run the configured benchmark suite:

```bash
bash run_benchmark.sh
```

Edit `MODELS` in `run_benchmark.sh` to choose the baselines or Hugging Face models to evaluate.

## Outputs

Each run writes prediction CSV files and a `summary.json` file under the requested results directory.

Monolingual WSD writes:

```text
wsd_en_predictions.csv
wsd_ro_predictions.csv
summary.json
```

Cross-lingual WSD writes:

```text
cross_lingual_wsd_en2ro_predictions.csv
cross_lingual_wsd_ro2en_predictions.csv
summary.json
```

## Citation

If you use this dataset or code, please cite the accompanying paper:

```bibtex
@inproceedings{balmus2026crosslingualwsd,
  title = {Cross-Lingual Word Sense Disambiguation Remains Challenging for Large Language Models},
  author = {Balmus, Sebastian and Dura, Bogdan and Uban, Ana-Sabina and Nisioi, Sergiu},
  booktitle = {Proceedings of the 30th International Conference on Knowledge-Based and Intelligent Information and Engineering Systems (KES 2026)},
  year = {2026}
}
```
