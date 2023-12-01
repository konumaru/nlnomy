# nlnomy

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlnomy.streamlit.app/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**nlnomy** is a demonstration application for text content moderation. The name "nlnomy" is derived from the concepts of "natural language" and "autonomy".

## Dataset Used

[inspection-ai/japanese-toxic-dataset](https://github.com/inspection-ai/japanese-toxic-dataset/tree/main)

## Pre-trained Models Utilized

- [nlp-waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese)
- [rinna/japanese-roberta-base](https://huggingface.co/rinna/japanese-roberta-base)
- [izumi-lab/deberta-v2-base-japanese](https://huggingface.co/izumi-lab/deberta-v2-base-japanese)

## Workflow

```mermaid
graph TD;
    Moderator --> id1(ModerationTool);
    id1 -- MissDetectionData --> Annotator; 
    Annotator --> id2(AnnotationTool);
    id2 -- AnnotatedData --> id3(MLSystem);
    id3 -- NewTrainedModel --> id1;
```
