# SpeechAudio LM — Project Brief

## Basic Information

**Project name:** SpeechAudio LM
**Possible venue:** ICASSP 2027
**GitHub repository:** TBD
**Collaborators:** Pooneh Mousavi, Yingzhi Wang
**Dataset:** `ttas`

## Overview

SpeechAudio LM is a generative modeling project focused on learning bidirectional mappings between environmental audio and natural speech.

The central goal is to build a model that can:

1. Given an environmental audio scene, generate a plausible spoken utterance that could naturally occur in that acoustic context.
2. Given a spoken utterance, generate a plausible environmental audio scene that matches the speech content.

Examples:

- **Audio:** Rainfall
  **Speech:** “Don’t forget your umbrella.”

- **Audio:** Crackling fire
  **Speech:** “Let’s sit closer — it’s getting cold.”

- **Audio:** City traffic
  **Speech:** “We’ll never make it on time at this rate.”

This differs from standard text-to-speech, speech synthesis, and audio captioning. The task is not only to describe an audio scene or synthesize speech from text; it requires reasoning about what kinds of speech and situations are plausible in a given acoustic environment, and vice versa.

The task is inherently generative, creative, and context-dependent.

## Expected Contributions

### 1. Bidirectional Speech-Audio Generation Model

Develop a model that can infer plausible audio scenes from spoken utterances and plausible spoken utterances from audio scenes.

Main directions:

- **Audio → Speech:** generate dialogue or utterances based on ambient/environmental sound.
- **Speech → Audio:** generate a matching environmental sound scene from speech.

The model should go beyond simple TTS or audio captioning by learning correspondences between acoustic context and speech content.

### 2. Reasoning About Audio Context and Speech Content

The task requires reasoning about the relationship between:

- environmental audio
- speech semantics
- likely situations
- speaker intent
- emotional tone
- contextually plausible utterances

For example, traffic noise may imply urgency, delay, frustration, or movement; rain may imply umbrellas, shelter, cold weather, or caution. The model should learn these contextual relationships.

### 3. Synthetic Dataset Generation for Downstream Use

A successful model could generate diverse paired speech-audio samples for downstream applications such as:

- video game audio and natural dialogue generation
- creative media and storytelling tools
- sound separation and source enhancement
- synthetic data generation for multimodal audio research

### 4. Representation Learning

The project may also contribute to representation learning by forcing models to learn useful latent structure connecting speech, audio scenes, and semantic context.

## Dataset

The dataset is `ttas`.

The dataset generation stage is considered complete for the current project phase.

The intended dataset construction process is:

1. Start with environmental audio scenes.
2. For each audio scene, generate 1–2 plausible natural speech sentences using an LLM.
3. Synthesize those sentences using a TTS model.
4. Create paired samples of the form:

```text
environmental audio ↔ synthetic speech
````

The dataset may include multiple plausible speech samples for one audio environment, although the current cleaned manifest should be treated according to the verified dataset structure.

## Current Dataset Location

The dataset is stored outside the Git repository.

Current Rorqual dataset root:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted
```

Current best clean manifest:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl
```

The repository may contain a copied small manifest file, but it must not contain audio files, zip files, checkpoints, or large generated outputs.

## Methodology

The proposal includes several modeling directions.

### 1. Autoregressive Discrete Audio Language Models

The primary initial direction is to train an autoregressive model on discrete audio representations.

Relevant model families:

* VALL-E-style models
* AudioGen-style models
* AudioCraft-based models

Relevant tokenizers:

* EnCodec
* WavTokenizer

The model will operate on discrete audio tokens and use cross-entropy loss.

### 2. Generation Tasks

The training procedure may include four generation tasks:

#### Speech → Audio

Generate an environmental audio scene related to a spoken utterance or conversation.

Example:

```text
Input: “Let’s sit closer — it’s getting cold.”
Output: crackling fire ambience
```

#### Audio → Speech

Generate a plausible spoken utterance based on an ambient audio scene.

Example:

```text
Input: rainfall
Output: “Don’t forget your umbrella.”
```

#### Text → Speech

Use text-to-speech as a warm-up or auxiliary task.

#### Text → Audio

Use text-to-audio as another warm-up or auxiliary task.

A likely training curriculum is:

1. Begin with simpler text-to-speech and/or text-to-audio tasks.
2. Transition to the core cross-modal tasks:

   * Speech → Audio
   * Audio → Speech

### 3. Training Objective

Primary objective:

* cross-entropy loss over discrete audio tokens

Possible auxiliary objectives:

* WER-based auxiliary signal for speech generation
* sound classification accuracy for audio generation
* alignment-related auxiliary losses

These auxiliary objectives are exploratory and should not be assumed unless explicitly implemented.

### 4. Continuous or Hybrid Generative Models

The proposal also leaves room for continuous generative models, including:

* diffusion models
* flow matching
* hybrid approaches with continuous input and discrete output

Motivation:

* overcome limitations of token-based representations
* improve audio quality
* improve robustness
* explore alternatives to fully discrete autoregressive modeling

## Evaluation

Evaluation is challenging because the task is generative and one-to-many. For a given audio scene, there may be many valid utterances. For a given utterance, there may be many valid audio scenes.

Therefore, evaluation should not rely only on a single reference.

### Evaluation Settings

Planned evaluation includes:

1. Performance on a held-out test set of audio-speech pairs.
2. Zero-shot generalization to unseen audio environments.
3. Qualitative inspection of generated samples.
4. Human or subjective evaluation.

### Evaluation Metrics and Methods

#### 1. Qualitative and Human Evaluation

Human evaluation is important because many outputs may be plausible even if they do not exactly match a reference.

Possible human evaluation questions:

* Is the generated speech plausible for the input audio?
* Is the generated audio plausible for the input speech?
* Are the speech and audio semantically aligned?
* Is the output natural and coherent?

#### 2. Semantic Similarity for Generated Speech

For generated speech, one option is to transcribe the generated speech and compare it against speech examples available for the same or similar input audio.

Because there may be multiple valid speech outputs per audio input, evaluation may compare the generated speech against multiple references and average or select the best similarity score.

#### 3. Model Judge

Possible model-judge approaches include:

* SALMON-style models
* Qwen-style speech/audio QA models
* prompt-based relevance/alignment judgments

Possible judgment dimensions:

* background consistency
* background alignment
* relevance between audio and speech
* whether the speech plausibly belongs in the audio scene
* whether the generated audio plausibly matches the speech

Example prompt idea:

```text
Given this audio and this speech, are the two inputs relevant or aligned?
```

#### 4. Pretrained Relevance or Alignment Models

Possible pretrained models/datasets:

* CLAP
* co-reasoning datasets
* relevant vs. irrelevant classifiers

These can be used to classify or score whether a speech-audio pair is aligned.

#### 5. Objective Metrics for Audio

Possible audio metrics:

* Fréchet Audio Distance (FAD)
* KL divergence for semantic consistency
* CLAP score
* ViSQOL

#### 6. Objective Metrics for Speech

Possible speech metrics:

* UTMOS
* WER-related metrics, where applicable

Important caveat: reference-based speech metrics are hard because generated speech can be valid even if it differs substantially from the reference.

### Distribution-Based Evaluation

Because reference-based evaluation is difficult, distributional evaluation may be useful.

Examples:

* compare distributions using FAD-like metrics
* generate multiple samples per input
* average metrics over multiple possible references
* choose best-of-N when multiple valid outputs exist

## Baselines and Architectures

The proposal suggests considering three broad architecture families.

### 1. Autoregressive Model with Delay Pattern

Potential implementation direction:

* AudioGen from AudioCraft
* AR modeling with delay pattern

Relevant reference implementation:

```text
https://github.com/gallilmaimon/audiocraft/blob/main/audiocraft/models/audiogen.py
```

### 2. VALL-E-Style Model

Potential implementation direction:

* SpeechBrain implementation of VALL-E
* discrete codec language modeling for speech generation

### 3. Diffusion / Flow Matching

Potential implementation direction:

* diffusion or flow-matching models for high-fidelity audio generation
* related direction: “From Discrete Tokens to High-Fidelity Audio Using Multi-Band Diffusion”
* Pooneh may explore recent flow-matching directions

## Tokenizers

Potential tokenizers:

* EnCodec
* WavTokenizer

Tokenizer choice should be treated as an experimental variable where feasible.

## Near-Term Implementation Priorities

The current implementation branch should focus on:

1. Repository setup.
2. Dataset manifest cleanup and validation.
3. Fixed train/validation/test split.
4. AudioCraft/AudioGen setup.
5. Initial baseline inference.
6. Initial training runs to understand the pipeline.
7. Saving generated samples for review.
8. Preparing concise progress reports.

## Dataset Split Requirement

The dataset must be split consistently across experiments.

Planned split:

```text
80% train / 10% validation / 10% test
```

The split should be deterministic and version-controlled.

Useful metadata for stratification:

* `speaker_gender`
* `speaker_emotion`

The current preferred strategy is to consider stratification by the joint label:

```text
speaker_gender × speaker_emotion
```

Any final split decision should be discussed and documented before implementation.

## Important Engineering Rules

* Do not commit audio files, speech files, zip files, checkpoints, model outputs, generated samples, or secrets to git.
* Keep large data outside the repository.
* Store small manifests, scripts, configs, and reports in git.
* The clean manifest copied into the repo may be versioned because it is small and defines the dataset state.
* Use a deterministic split seed.
* Keep split files fixed across all experiments.
* Do not run heavy training or large preprocessing on login nodes.
* Use compute nodes or SLURM jobs for heavy workloads.
