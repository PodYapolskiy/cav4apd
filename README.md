# Concept Direction Ablation for Large Language Models

## Description

This project is based on Refusal in Language Models Is Mediated by a Single Direction paper
https://arxiv.org/abs/2406.11717

We take this concept further by experimenting with politeness concept, interlanguage concept understanding and creating an interface for convenient vector shifting.

As a base model Qwen-1_8B-chat was used to build a Gradio web-interface and implement the original paper method. For experimental purposes we picked 4bit-quantized YandexGPT-5-Lite-8B-instruct to test how well direction vectors obtained from English examples would work for Russian language.

## Visualization

![alt]()

![alt](./imgs/TransformerLens_Diagram.svg)

## Data

https://github.com/llm-attacks/llm-attacks

## How to run

#### Setup

```bash
uv python install 3.12
```

```bash
uv sync
```

#### Gradio App

```bash
uvx gradio app/demo.py
```
