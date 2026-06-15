# Prometheus G-Eval: Evaluation Prompts and Scoring Protocol

## 1. Evaluation Problem

SpeechAudio LM performs bidirectional generation between environmental audio and
speech:

- **Audio-to-speech:** given a textual description of an environmental audio
  scene, generate a plausible human utterance that could occur in, after, or
  because of that scene.
- **Speech-to-audio:** given a speech utterance, generate a plausible
  environmental audio description that could contextualize, accompany,
  explain, or be caused by that speech.

These tasks are open-ended. Multiple responses may be valid even when they do
not match a single reference exactly. The evaluator must therefore measure
cross-modal compatibility, plausibility, coherence, specificity, and
naturalness without treating evaluation as exact text matching.

The implemented G-Eval evaluator uses Prometheus 8x7B v2.0 as a judge. It makes
one independent direct-assessment call per criterion, requests concise
feedback and an integer score from 1 to 5, and combines the criterion scores
using a direction-specific weighted mean.

Two variants are implemented:

1. **Plausibility G-Eval:** evaluates the input-response pair without consulting
   a reference output.
2. **Reference-grounded G-Eval:** applies the same core evaluation and adds a
   separate criterion measuring whether the response belongs to the intended
   output space represented by one or more reference anchors.

## 2. Judge Message and System Prompt

There is **no explicit system-role prompt** in the implementation. Each
criterion prompt is passed as one `user` message:

```python
[{"role": "user", "content": prompt}]
```

The tokenizer's model-specific chat template is then applied with
`add_generation_prompt=True`. Consequently, the exact serialized control
tokens and assistant prefix come from the Prometheus model tokenizer, not from
a separately authored system message in this project.

Conceptually, the message sequence is:

```text
SYSTEM: <none supplied by the evaluator>
USER: <criterion-specific direct-assessment prompt>
ASSISTANT: <generated feedback and result>
```

## 3. Common Evaluation Protocol

Every call evaluates exactly one criterion. The prompt tells the judge:

- multiple outputs can be valid;
- judge only the requested criterion;
- do not allow qualities covered by other criteria to affect the score;
- provide concise criterion-specific feedback;
- return an integer score from 1 to 5, or `N/A` only when the criterion is
  genuinely not assessable.

Required output:

```text
Feedback: <feedback> [RESULT] <integer score from 1 to 5>
```

Permitted abstention:

```text
Feedback: <feedback> [RESULT] N/A
```

`N/A` must not be used merely because an output is poor, short, vague, or
low-quality. Assessable weak outputs should receive a score of 1 or 2.

### General Scale Interpretation

Each criterion has its own explicit rubric, but the scale follows this common
pattern:

| Score | General interpretation |
|---:|---|
| 5 | Strong, fully compatible, highly plausible, coherent, or natural |
| 4 | Good, with only a minor weakness, ambiguity, or omission |
| 3 | Partial, generic, weak, incomplete, or mildly questionable |
| 2 | Substantial weakness, tenuous connection, likely conflict, or implausibility |
| 1 | Complete failure on the criterion, clear contradiction, or incoherence |

The criterion-specific definitions below take precedence over this summary.

## 4. Criteria Shared by Both Directions

### 4.1 Non-Contradiction

**Criterion prompt**

> Evaluate the degree to which the response conflicts with the input text.

| Score | Definition |
|---:|---|
| 5 | No contradiction or meaningful tension; the response is fully compatible with the input. |
| 4 | No likely conflict, with only minor ambiguity or mild tension. |
| 3 | Mostly compatible, but includes detail that creates a genuine possible inconsistency with the input. |
| 2 | Likely or significant conflict with the input, though not a direct explicit contradiction. |
| 1 | Clear contradiction of the input. |

**Criterion boundary:** Judge only conflict with the input. Do not penalize weak
relatedness, generic wording, missing details, or merely plausible unsupported
details unless they create a genuine possible, likely, or actual conflict.

### 4.2 Specificity Without Over-Invention

**Criterion prompt**

> Evaluate whether the response is concrete and useful without inventing
> unsupported excessive detail.

| Score | Definition |
|---:|---|
| 5 | Concrete, informative, and appropriately specific without unsupported over-invention. |
| 4 | Mostly useful and specific, with minor vagueness or minor extra detail. |
| 3 | Understandable but generic, under-specified, or slightly over-invented. |
| 2 | Too vague or substantially over-invented relative to the input. |
| 1 | Empty, uninformative, or dominated by unsupported invented details. |

**Criterion boundary:** Judge only whether the amount of supported detail is
appropriate. Do not treat every reasonable unstated detail as a contradiction,
and do not score general cross-modal relevance here.

## 5. Audio-to-Speech Criteria

In this direction, the input is an environmental audio description and the
response is a generated human utterance.

### 5.1 Meaningful Scene-to-Utterance Connection

**Criterion prompt**

> Evaluate whether the generated utterance has a meaningful situational,
> causal, contextual, or reactive relationship to the input audio scene.

| Score | Definition |
|---:|---|
| 5 | Strong, direct, specific relationship; the utterance clearly makes sense in, after, or because of the audio scene. |
| 4 | Clear plausible relationship with minor weakness or missing specificity. |
| 3 | Identifiable but generic, weak, or incomplete relationship. |
| 2 | Very tenuous relationship; the utterance is mostly independent of the audio scene. |
| 1 | No meaningful relationship to the audio scene. |

**Criterion boundary:** Judge only the scene-to-utterance relationship. Do not
score language fluency, emotional tone, or reference similarity.

### 5.2 Human Reaction Plausibility

**Criterion prompt**

> Evaluate whether a human would plausibly say the generated utterance in
> response to, during, or because of the described audio scene.

| Score | Definition |
|---:|---|
| 5 | Highly plausible human utterance for the situation implied by the audio scene. |
| 4 | Plausible utterance with minor contextual or pragmatic weakness. |
| 3 | Possible but weakly motivated, atypical, or pragmatically strained for the situation. |
| 2 | Unlikely or forced human reaction to the audio scene. |
| 1 | Implausible as something a human would say in relation to the audio scene. |

**Criterion boundary:** Judge whether this is a plausible thing for a human to
say in the situation. Do not require every input detail to be mentioned or a
reference utterance to be matched.

### 5.3 Contextual and Emotional Fit

**Criterion prompt**

> Evaluate whether the response fits the implied situation, intent, mood,
> urgency, or emotional context of the input.

| Score | Definition |
|---:|---|
| 5 | Strong fit to the implied context, mood, intent, or human reaction. |
| 4 | Good fit with minor weakness or missing nuance. |
| 3 | Partially fitting but generic, emotionally flat, or only weakly aligned. |
| 2 | Poor contextual or emotional fit. |
| 1 | Completely mismatched context, mood, intent, or emotional reaction. |

**Criterion boundary:** Judge only situational, pragmatic, emotional, and
urgency fit. Do not score wording naturalness, exact reference similarity, or
omission of acoustically irrelevant details.

### 5.4 Spoken Naturalness

**Criterion prompt**

> Evaluate whether the generated speech text sounds like a natural human
> utterance.

| Score | Definition |
|---:|---|
| 5 | Natural, fluent, and plausible as spoken human language. |
| 4 | Mostly natural with minor awkwardness. |
| 3 | Understandable but somewhat stiff, repetitive, or template-like. |
| 2 | Unnatural, awkward, or difficult to accept as human speech. |
| 1 | Broken, incoherent, excessively repetitive, or not a plausible utterance. |

**Criterion boundary:** Judge only the utterance's naturalness and fluency.
Ignore relevance, factual grounding, and reference similarity.

## 6. Speech-to-Audio Criteria

In this direction, the input is a speech utterance and the response is a
generated environmental audio description.

This is the main direction-dependent difference. The shared
`non_contradiction` and `specificity_without_over_invention` criteria remain,
but the four audio-to-speech criteria above are replaced by criteria appropriate
for an environmental audio scene.

### 6.1 Meaningful Utterance-to-Scene Connection

**Criterion prompt**

> Evaluate whether the generated audio description has a meaningful
> situational, causal, contextual, or accompanying relationship to the input
> speech utterance.

| Score | Definition |
|---:|---|
| 5 | Strong, direct, specific relationship; the audio scene clearly contextualizes, accompanies, explains, or could be caused by the speech. |
| 4 | Clear plausible relationship with minor weakness or missing specificity. |
| 3 | Identifiable but generic, weak, or incomplete relationship. |
| 2 | Very tenuous relationship; the audio scene is mostly independent of the speech. |
| 1 | No meaningful relationship to the speech. |

**Criterion boundary:** Judge only the utterance-to-scene relationship. Do not
score prose quality, internal acoustic realism, or reference similarity.

### 6.2 Causal and Contextual Plausibility

**Criterion prompt**

> Evaluate whether the generated audio scene could plausibly contextualize,
> accompany, explain, or be caused by the input speech.

| Score | Definition |
|---:|---|
| 5 | Highly plausible causal/contextual relationship between the speech and the audio scene. |
| 4 | Plausible relationship with minor weakness or missing detail. |
| 3 | Possible but generic, weak, or only partially motivated relationship. |
| 2 | Unlikely or forced relationship between the speech and the audio scene. |
| 1 | Implausible relationship between the speech and the audio scene. |

**Criterion boundary:** Judge only causal/contextual plausibility. Do not score
prose quality, acoustic realism, or reference similarity.

### 6.3 Physical and Acoustic Coherence

**Criterion prompt**

> Evaluate whether the generated audio scene is physically and acoustically
> coherent.

| Score | Definition |
|---:|---|
| 5 | Sound sources, events, and acoustic scene are realistic and internally coherent. |
| 4 | Mostly coherent with only minor physical or acoustic weakness. |
| 3 | Generally possible but includes mildly questionable interactions among sound sources, events, or acoustic conditions. |
| 2 | Contains significant physical or acoustic implausibility. |
| 1 | Physically impossible, acoustically incoherent, or internally contradictory. |

**Criterion boundary:** Judge only whether the generated scene is internally
physically and acoustically possible. Do not penalize weak input relevance or
disagreement with a reference anchor.

### 6.4 Audio-Description Naturalness

**Criterion prompt**

> Evaluate whether the generated audio description is a coherent natural
> description of an audio scene.

| Score | Definition |
|---:|---|
| 5 | Natural, coherent, and well-formed audio scene description. |
| 4 | Mostly natural with minor awkwardness or wording issue. |
| 3 | Understandable but somewhat stiff, repetitive, or template-like. |
| 2 | Unnatural, awkward, or poorly formed as an audio description. |
| 1 | Broken, incoherent, excessively repetitive, or not a meaningful audio description. |

**Criterion boundary:** Judge only whether the text is a natural and coherent
audio-scene description. Ignore relevance, factual grounding, and reference
similarity.

## 7. Criterion Sets and Weights

### Plausibility G-Eval

| Audio-to-speech criterion | Weight | Speech-to-audio criterion | Weight |
|---|---:|---|---:|
| Non-contradiction | 0.20 | Non-contradiction | 0.20 |
| Meaningful scene-to-utterance connection | 0.25 | Meaningful utterance-to-scene connection | 0.25 |
| Human reaction plausibility | 0.20 | Causal/contextual plausibility | 0.20 |
| Contextual/emotional fit | 0.15 | Physical/acoustic coherence | 0.15 |
| Specificity without over-invention | 0.10 | Specificity without over-invention | 0.10 |
| Spoken naturalness | 0.10 | Audio-description naturalness | 0.10 |

### Reference-Grounded G-Eval

| Audio-to-speech criterion | Weight | Speech-to-audio criterion | Weight |
|---|---:|---|---:|
| Non-contradiction | 0.15 | Non-contradiction | 0.15 |
| Meaningful scene-to-utterance connection | 0.20 | Meaningful utterance-to-scene connection | 0.20 |
| **Reference-anchor grounding** | **0.25** | **Reference-anchor grounding** | **0.25** |
| Human reaction plausibility | 0.15 | Causal/contextual plausibility | 0.15 |
| Contextual/emotional fit | 0.10 | Physical/acoustic coherence | 0.10 |
| Specificity without over-invention | 0.075 | Specificity without over-invention | 0.075 |
| Spoken naturalness | 0.075 | Audio-description naturalness | 0.075 |

The final score is:

```text
final_score = sum(criterion_score_i * weight_i) / sum(weight_i)
```

All listed criteria are required. If any criterion abstains or fails to produce
a valid score, the implementation does not compute a final score for that row.

## 8. Full Runtime Prompt Template

The following is the complete logical prompt supplied for each criterion. The
criterion boundary and criterion rubric are placeholders because their exact
contents are documented above.

```text
###Task Description:
You are evaluating a creative cross-modal generation system.

{TASK_DESCRIPTION}

Judge only the requested criterion using the score rubric.
Do not let qualities assigned to other criteria affect this score.

Criterion boundary:
{CRITERION_BOUNDARY}

Direction: {DIRECTION}

For audio_to_speech, the input is an environmental audio description and the
response is a generated human utterance that could occur in, after, or because
of the audio scene.

For speech_to_audio, the input is a speech utterance and the response is a
generated environmental audio description that could contextualize, accompany,
explain, or be caused by the speech.

Write concise feedback strictly based on the criterion and rubric. Then output
the final score as:
Feedback: <feedback> [RESULT] <integer score from 1 to 5>

If the criterion genuinely cannot be assessed from the response, you may
instead output [RESULT] N/A. Do not use N/A merely because the response is
poor, short, vague, or low quality; use scores 1 or 2 for assessable weak
outputs.

###The instruction to evaluate:
Input text:
{INPUT_TEXT}

###Response to evaluate:
{GENERATED_TEXT}
{OPTIONAL_REFERENCE_SECTION}

###Score Rubric:
Criterion: {CRITERION_NAME}

{CRITERION_RUBRIC}

###Feedback:
```

For every non-reference criterion, `{TASK_DESCRIPTION}` is:

```text
Multiple outputs can be valid. Judge only the requested criterion using the
score rubric. No reference comparison is part of this criterion.
```

`{OPTIONAL_REFERENCE_SECTION}` is empty for all non-reference criteria.

## 9. Reference-Grounded G-Eval Addition

Reference-grounded G-Eval does not insert references into every criterion.
Instead, it adds one independent `reference_anchor_grounding` judgment. Only
that judgment receives the reference-specific task description, the formatted
reference anchors, and the direction-specific reference rubric.

### Reference-Specific Task Description

```text
The task is not exact reference matching. Multiple outputs can be valid. The
reference anchors are examples of the intended target-output space, not the
only correct answers. Do not require exact wording or exact scene matching.
Use them only to judge whether the response is grounded in the dataset's
intended paired output space.
```

### Reference Section Appended After the Response

```text
###Reference Anchors:
[Anchor 1 | source={SOURCE} | id={ANCHOR_ID}]
{REFERENCE_TEXT}

[Anchor 2 | source={SOURCE} | id={ANCHOR_ID}]
{REFERENCE_TEXT}

...
```

### Reference Criterion Boundary

```text
Judge grounding in the intended output space represented by the anchor, not
exact matching. A different utterance or scene may receive a high score when
it preserves the anchor's important situation, intent, sound source, event, or
communicative function. Do not penalize wording or harmless realization
changes.
```

### Audio-to-Speech Reference Rubric

**Criterion prompt**

> Evaluate whether the generated utterance is grounded in the reference speech
> anchor(s) while allowing valid alternative wording and reasonable variation.

The anchors are not exact gold answers. A high score requires preservation of
the important intended speech-output space, such as the core situation, intent,
reaction, or communicative function, rather than copied wording.

| Score | Definition |
|---:|---|
| 5 | Strongly grounded in the reference anchor(s); preserves the important intended speech situation, intent, or reaction while allowing natural variation. |
| 4 | Mostly grounded in the reference anchor(s), with minor omission, shift, or extra detail. |
| 3 | Partially grounded, but misses or changes an important aspect of the reference output space. |
| 2 | Weakly grounded; mostly an alternative utterance not well supported by the reference anchor(s). |
| 1 | Not grounded in, or clearly conflicts with, the reference anchor(s). |

### Speech-to-Audio Reference Rubric

**Criterion prompt**

> Evaluate whether the generated audio description is grounded in the
> reference audio-caption anchor(s) while allowing valid alternative
> realizations.

The anchors are not exact gold answers. A high score requires preservation of
the important intended audio-output space, such as core sound sources, scene
type, events, or acoustic context, rather than copied wording.

| Score | Definition |
|---:|---|
| 5 | Strongly grounded in the reference anchor(s); preserves the important intended audio scene, sound sources, or events while allowing reasonable variation. |
| 4 | Mostly grounded in the reference anchor(s), with minor omission, shift, or extra detail. |
| 3 | Partially grounded, but misses or changes an important aspect of the reference output space. |
| 2 | Weakly grounded; mostly an alternative audio scene not well supported by the reference anchor(s). |
| 1 | Not grounded in, or clearly conflicts with, the reference anchor(s). |

Reference-grounded evaluation requires at least one non-empty reference anchor.
A row without a valid anchor is marked incomplete and is not sent to the judge.

## 10. Output Repair Prompt

If the initial model output cannot be parsed, the same judge receives a second,
format-repair prompt:

```text
Repair the evaluator response into the native Prometheus direct-assessment
format.

Return exactly one of:
Feedback: <non-empty concise feedback> [RESULT] <one integer from 1 to 5>
Feedback: <non-empty concise feedback> [RESULT] N/A

Do not return JSON. Preserve an explicit N/A rather than forcing a score.

Response to repair:
{RAW_RESPONSE}
```

This prompt repairs formatting only. It is not an additional evaluation
criterion.

## 11. Implementation Source

The authoritative implementation is
`scripts/run_geval_prometheus_criterion_v2.py`:

- prompt versions and criterion weights: lines 17-64;
- criterion rubrics: lines 66-158;
- criterion boundaries: lines 160-208;
- chat-template invocation: lines 276-286;
- full prompt builder: lines 347-403;
- output-repair prompt: lines 413-423.

Current prompt versions:

- `geval_prometheus_plausibility_direct_assessment_v3`
- `geval_prometheus_reference_grounded_direct_assessment_v3`
