As part of an exploration project during my MSBA program at the University of Virginia, I experimented with multiple smaller GPT models on different tasks.

Here is what I did:

# Transformer Architecture Exploration: DistilBERT vs. XLNet

This repository serves as a foundational exploration into transformer-based architectures. Before moving into production-level NLP tools, I implemented and compared two distinct models—DistilBERT and XLNet—to evaluate their performance across a variety of Natural Language Processing tasks.

## Project Scope

This project isn't just a simple chatbot; it's a comparative analysis of how different transformer designs handle linguistic nuance. By using the Hugging Face Transformers library, I tested these models across four core NLP pillars:

Text Generation: Evaluating the creativity and coherence of autoregressive (XLNet) vs. distilled (DistilBERT) outputs.

Sentiment Analysis: Benchmarking the accuracy of sentiment classification on diverse text samples.

Text Classification: Testing the ability to categorize complex documents into predefined intents.

Question Answering (QA): Implementing a context-aware QA system to extract precise answers from text blocks.

## The Comparison: DistilBERT vs. XLNet

A key goal of this project was understanding the trade-offs between different transformer philosophies:

| Feature | **DistilBERT** | **XLNet** |
| :--- | :--- | :--- |
| **Philosophy** | Efficiency & Speed (Distilled BERT) | Performance & Context (Generalized Autoregressive) |
| **Mechanism** | Bidirectional (Masked LM) | Permutation-based (Captures long-range dependencies) |
| **Model Size** | ~66M Parameters (40% smaller than BERT) | ~110M+ Parameters (Base version) |
| **Performance** | Retains ~97% of BERT's performance | Often outperforms BERT/RoBERTa on complex tasks |
| **Best For** | Edge devices, low-latency apps, and speed | Nuanced reasoning and long-form text analysis |
| **Training** | Knowledge Distillation from BERT | Permutation Language Modeling (PLM) |

## Interactive Features

To move beyond static code, I built an interactive UI using ipywidgets. This allows for live testing of the Question Answering (QA) capabilities. Users can input a custom context paragraph and ask the model questions in real-time, observing how each architecture interprets the relationship between tokens.

## License
This project is licensed under the MIT License.
