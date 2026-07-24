# Presentations

Slide decks presenting Artefactual to internal and external audiences.

## Decks

|Deck|   Description  | Date  |
|----|----------------|-------|
| [Artefactual: Integrations](integrations_sklearn_langfuse/integrations_sklearn_langfuse.ipynb) | Introduces LLM hallucination detection, the EPR/WEPR scores, and integrations with scikit-learn and Langfuse. | 2026-07-22 |

## Tracked sources vs. rendered output

Notebook sources (`.ipynb`) and images are committed to git. HTML and accompanying file directories are excluded via `.gitignore` and generated at build time.

## Adding a new deck

1. Create a folder under `docs/presentations/` and place the `.ipynb` file and any images.
2. Add a row to the table above with a brief description and the presentation date.
3. To run locally:
   ```bash
   uvx --from quarto-cli quarto preview docs/presentations/your_sub_folder/your_deck.ipynb
   ````
