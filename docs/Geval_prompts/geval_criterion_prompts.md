# GEval Criterion Prompts

Rendered from the current prompt builders in:

- `scripts/run_geval_prometheus_criterion_v2.py`
- `scripts/run_geval_audio_qwen3_criterion.py`

The examples below use placeholder input/output text and placeholder audio file paths. The structure, criterion names, criterion boundaries, and rubrics are rendered exactly from the code.

Generated at: `2026-06-26 14:48:42 UTC`

## GEval Text

- Evaluator version: `geval_prometheus_criterion_v3`
- Template: `geval_text_direct_assessment_v1`

### Plausibility

- Metric id: `geval_text_plausibility`
- Prompt version: `geval_text_plausibility_v2_indirect_links`

#### A2S / audio_to_speech

##### `non_contradiction`

- Criterion boundary: Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: non_contradiction

Criterion boundary:
Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

Rubric:
Evaluate whether the generated utterance avoids contradicting the input audio caption.

Focus only on contradiction. The utterance may add reasonable unstated context and may be simple or indirect. Do not penalize the utterance merely because it does not mention the sound source, is generic, or has weak relevance. Penalize only when it denies, reverses, or conflicts with explicit or strongly implied facts from the audio scene.

Score 1: The utterance directly and severely contradicts the audio caption, or is incoherent with the scene.
Score 2: The utterance contains a major contradiction with the audio caption or with a strongly implied fact from the scene.
Score 3: The utterance has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The utterance is mostly compatible with the audio caption, with only minor ambiguity or weak tension.
Score 5: The utterance is fully compatible with the audio caption and contains no meaningful contradiction.

###Feedback:
```

##### `conditional_relevance`

- Criterion boundary: Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: conditional_relevance

Criterion boundary:
Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

Rubric:
Evaluate whether the generated utterance is meaningfully grounded in the input audio caption.

The utterance should use, respond to, or be motivated by the described sound event, acoustic context, or situation. The connection may be indirect: a departure can fit a fading vehicle, a celebration can fit cheering, and a command can fit a sound-producing action. It does not need to restate the caption or mention the sound source. Simple utterances can still be relevant when a plausible situational bridge is clear.

Score 1: The utterance is unrelated to the audio caption.
Score 2: The utterance has only a very weak, vague, or accidental connection to the audio caption.
Score 3: The utterance has a plausible but generic, indirect, or underdeveloped connection to the audio caption.
Score 4: The utterance is clearly connected to the audio caption through an explicit or implicit situation, with only minor genericness or missed opportunity.
Score 5: The utterance is strongly grounded in the audio caption through a natural explicit or implicit situation.

###Feedback:
```

##### `scenario_plausibility`

- Criterion boundary: Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: scenario_plausibility

Criterion boundary:
Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

Rubric:
Evaluate whether the generated utterance and its minimally implied situation are believable human speech.

This criterion is about plausibility of the utterance as a human situation, not about how strongly it is grounded in the input; grounding is evaluated by conditional_relevance. Consider temporal, causal, emotional, and pragmatic fit only when they make the utterance or its minimal scenario believable or unbelievable. Do not assign a very low score merely because the utterance is simple, generic, or does not name the sound source.

Score 1: The utterance or implied situation is absurd, impossible, incoherent, or extremely unnatural as human behavior.
Score 2: The utterance is difficult to believe because of a major temporal, causal, emotional, pragmatic, or human-reaction mismatch.
Score 3: The utterance is possible but somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The utterance forms a plausible human situation, requiring at most a small reasonable assumption.
Score 5: The utterance feels like a very natural and believable human thing to say in its implied situation.

###Feedback:
```

##### `appropriate_specificity`

- Criterion boundary: Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: appropriate_specificity

Criterion boundary:
Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

Rubric:
Evaluate whether the generated utterance has an appropriate level of specificity: not too vague, and not too over-specific.

The utterance may infer reasonable context from the audio caption because the task is one-to-many. It does not need to explicitly mention the sound source. Simple everyday utterances can have acceptable specificity if they provide a usable action, intent, reaction, or situation. Penalize only when the output is nearly empty or so generic that it gives no usable scenario, or when it invents highly specific unsupported details, identities, locations, causes, or events.

Score 1: The utterance is nearly empty, has no usable situational content, or is wildly over-invented with unsupported details.
Score 2: The utterance is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The utterance is simple or generic but still gives some usable action, intent, reaction, or situation; or it includes somewhat unsupported detail.
Score 4: The utterance has mostly appropriate specificity, with only minor vagueness or minor over-inference.
Score 5: The utterance is appropriately specific: concrete enough to be useful, but not over-invented beyond reasonable inference from the audio caption.

###Feedback:
```

##### `target_form_validity`

- Criterion boundary: Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: target_form_validity

Criterion boundary:
Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

Rubric:
Evaluate whether the generated text is valid as a speech transcript / human utterance.

Judge only form, not semantic alignment with the audio caption. The output should read like something a person could actually say aloud: direct speech, a question, command, exclamation, or first-person utterance can all be valid. It should not be an audio caption, third-person report of what someone said, scene description, narrator comment, analysis, label, or explanation of what is happening.

Score 1: The output is not a valid utterance, is gibberish, or is completely in the wrong format.
Score 2: The output is mostly not a spoken utterance; it is mainly a caption, third-person report, description, analysis, or label.
Score 3: The output is partly utterance-like, but noticeably unnatural, caption-like, written, or not clearly speakable.
Score 4: The output is a mostly natural spoken utterance, with only minor stiffness or formatting issues.
Score 5: The output is a natural, fluent, clearly speakable human utterance.

###Feedback:
```

#### S2A / speech_to_audio

##### `non_contradiction`

- Criterion boundary: Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: non_contradiction

Criterion boundary:
Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

Rubric:
Evaluate whether the generated audio caption avoids contradicting the input utterance.

Focus only on contradiction. The audio caption may add reasonable unstated context and may use an implied reading of the utterance. Do not penalize the caption merely because the link is generic, indirect, or not explicitly named in the utterance. Penalize only when it denies, reverses, or conflicts with explicit or strongly implied facts, intent, or context from the utterance.

Score 1: The audio caption directly and severely contradicts the utterance, or is incoherent with it.
Score 2: The audio caption contains a major contradiction with the utterance or with a strongly implied fact from it.
Score 3: The audio caption has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The audio caption is mostly compatible with the utterance, with only minor ambiguity or weak tension.
Score 5: The audio caption is fully compatible with the utterance and contains no meaningful contradiction.

###Feedback:
```

##### `conditional_relevance`

- Criterion boundary: Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: conditional_relevance

Criterion boundary:
Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

Rubric:
Evaluate whether the generated audio caption is meaningfully grounded in the input utterance.

The audio scene should contextualize, accompany, explain, or be caused by the utterance. The connection may be indirect through common-sense implications, domain words, actions, consequences, or pragmatic context. Do not force an unrelated interpretation when an obvious reading creates a strong audio-scene link. The audio caption does not need to be the only possible scene.

Score 1: The audio caption is unrelated to the utterance.
Score 2: The audio caption has only a very weak, vague, or accidental connection to the utterance.
Score 3: The audio caption has a plausible but generic, indirect, or underdeveloped connection to the utterance.
Score 4: The audio caption is clearly connected to the utterance through an explicit or implicit situation, with only minor genericness or missed opportunity.
Score 5: The audio caption is strongly grounded in the utterance through a natural explicit or implicit situation.

###Feedback:
```

##### `scenario_plausibility`

- Criterion boundary: Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: scenario_plausibility

Criterion boundary:
Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

Rubric:
Evaluate whether the generated audio caption and its minimally implied scene are believable as an audio situation.

This criterion is about plausibility of the described audio scene, not about how strongly it is grounded in the input; grounding is evaluated by conditional_relevance. Consider temporal, causal, emotional, pragmatic, and acoustic-world fit only when they make the described scene believable or unbelievable. Do not assign a very low score merely because the caption is simple, generic, or indirectly linked.

Score 1: The described audio scene is absurd, impossible, incoherent, or extremely unnatural.
Score 2: The described audio scene is difficult to believe because of a major temporal, causal, pragmatic, or acoustic-world problem.
Score 3: The described audio scene is possible but somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The described audio scene is plausible, requiring at most a small reasonable assumption.
Score 5: The described audio scene feels very natural and believable.

###Feedback:
```

##### `appropriate_specificity`

- Criterion boundary: Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: appropriate_specificity

Criterion boundary:
Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

Rubric:
Evaluate whether the generated audio caption has an appropriate level of specificity: not too vague, and not too over-specific.

The audio caption may infer reasonable context from the utterance because the task is one-to-many. It does not need to restate the utterance. A short caption can have acceptable specificity if it names a useful sound event, source, or acoustic action. Penalize only when the caption is nearly empty or so generic that it gives no usable audio scene, or when it invents highly specific unsupported details, identities, locations, causes, or events.

Score 1: The audio caption is nearly empty, has no usable sound-scene content, or is wildly over-invented with unsupported details.
Score 2: The audio caption is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The audio caption is simple or generic but still gives a usable sound event, source, or acoustic action; or it includes somewhat unsupported detail.
Score 4: The audio caption has mostly appropriate specificity, with only minor vagueness or minor over-inference.
Score 5: The audio caption is appropriately specific: concrete enough to be useful, but not over-invented beyond reasonable inference from the utterance.

###Feedback:
```

##### `target_form_validity`

- Criterion boundary: Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: target_form_validity

Criterion boundary:
Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

Rubric:
Evaluate whether the generated text is valid as an environmental audio caption.

Judge only form, not semantic alignment with the input utterance. The output should describe audible events, sound sources, acoustic actions, or sound context. Sound-producing actions such as something being shot, flushed, knocked, loaded, opened, started, or running can be valid audio-caption wording when the event is audibly recognizable. It should not merely infer the speaker's feelings, continue the speech, answer the speaker, or explain the situation without describing sound.

Score 1: The output is not a valid audio caption, is gibberish, or is completely in the wrong format.
Score 2: The output is mostly not an audio caption; it is mainly a speech continuation, explanation, emotion label, visual-only scene, or analysis.
Score 3: The output is partly audio-caption-like, but includes substantial non-audio content or unnatural caption style.
Score 4: The output is a mostly valid environmental audio caption, with only minor style issues.
Score 5: The output is a natural, clear environmental audio caption describing audible events, sound sources, acoustic actions, or acoustic context.

###Feedback:
```

### Reference-Grounded

- Metric id: `geval_text_reference`
- Prompt version: `geval_text_reference_v2_indirect_links`

#### A2S / audio_to_speech

##### `non_contradiction`

- Criterion boundary: Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: non_contradiction

Criterion boundary:
Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

Rubric:
Evaluate whether the generated utterance avoids contradicting the input audio caption.

Focus only on contradiction. The utterance may add reasonable unstated context and may be simple or indirect. Do not penalize the utterance merely because it does not mention the sound source, is generic, or has weak relevance. Penalize only when it denies, reverses, or conflicts with explicit or strongly implied facts from the audio scene.

Score 1: The utterance directly and severely contradicts the audio caption, or is incoherent with the scene.
Score 2: The utterance contains a major contradiction with the audio caption or with a strongly implied fact from the scene.
Score 3: The utterance has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The utterance is mostly compatible with the audio caption, with only minor ambiguity or weak tension.
Score 5: The utterance is fully compatible with the audio caption and contains no meaningful contradiction.

###Feedback:
```

##### `conditional_relevance`

- Criterion boundary: Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: conditional_relevance

Criterion boundary:
Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

Rubric:
Evaluate whether the generated utterance is meaningfully grounded in the input audio caption.

The utterance should use, respond to, or be motivated by the described sound event, acoustic context, or situation. The connection may be indirect: a departure can fit a fading vehicle, a celebration can fit cheering, and a command can fit a sound-producing action. It does not need to restate the caption or mention the sound source. Simple utterances can still be relevant when a plausible situational bridge is clear.

Score 1: The utterance is unrelated to the audio caption.
Score 2: The utterance has only a very weak, vague, or accidental connection to the audio caption.
Score 3: The utterance has a plausible but generic, indirect, or underdeveloped connection to the audio caption.
Score 4: The utterance is clearly connected to the audio caption through an explicit or implicit situation, with only minor genericness or missed opportunity.
Score 5: The utterance is strongly grounded in the audio caption through a natural explicit or implicit situation.

###Feedback:
```

##### `scenario_plausibility`

- Criterion boundary: Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: scenario_plausibility

Criterion boundary:
Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

Rubric:
Evaluate whether the generated utterance and its minimally implied situation are believable human speech.

This criterion is about plausibility of the utterance as a human situation, not about how strongly it is grounded in the input; grounding is evaluated by conditional_relevance. Consider temporal, causal, emotional, and pragmatic fit only when they make the utterance or its minimal scenario believable or unbelievable. Do not assign a very low score merely because the utterance is simple, generic, or does not name the sound source.

Score 1: The utterance or implied situation is absurd, impossible, incoherent, or extremely unnatural as human behavior.
Score 2: The utterance is difficult to believe because of a major temporal, causal, emotional, pragmatic, or human-reaction mismatch.
Score 3: The utterance is possible but somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The utterance forms a plausible human situation, requiring at most a small reasonable assumption.
Score 5: The utterance feels like a very natural and believable human thing to say in its implied situation.

###Feedback:
```

##### `appropriate_specificity`

- Criterion boundary: Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: appropriate_specificity

Criterion boundary:
Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

Rubric:
Evaluate whether the generated utterance has an appropriate level of specificity: not too vague, and not too over-specific.

The utterance may infer reasonable context from the audio caption because the task is one-to-many. It does not need to explicitly mention the sound source. Simple everyday utterances can have acceptable specificity if they provide a usable action, intent, reaction, or situation. Penalize only when the output is nearly empty or so generic that it gives no usable scenario, or when it invents highly specific unsupported details, identities, locations, causes, or events.

Score 1: The utterance is nearly empty, has no usable situational content, or is wildly over-invented with unsupported details.
Score 2: The utterance is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The utterance is simple or generic but still gives some usable action, intent, reaction, or situation; or it includes somewhat unsupported detail.
Score 4: The utterance has mostly appropriate specificity, with only minor vagueness or minor over-inference.
Score 5: The utterance is appropriately specific: concrete enough to be useful, but not over-invented beyond reasonable inference from the audio caption.

###Feedback:
```

##### `target_form_validity`

- Criterion boundary: Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance. It should be something a human could plausibly say in, after, or because of the described audio scene.

This is a one-to-many task. A valid utterance may connect to the audio indirectly through a plausible situation, action, reaction, implication, or shared context. Do not require the utterance to name, quote, or literally describe the sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: target_form_validity

Criterion boundary:
Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

Rubric:
Evaluate whether the generated text is valid as a speech transcript / human utterance.

Judge only form, not semantic alignment with the audio caption. The output should read like something a person could actually say aloud: direct speech, a question, command, exclamation, or first-person utterance can all be valid. It should not be an audio caption, third-person report of what someone said, scene description, narrator comment, analysis, label, or explanation of what is happening.

Score 1: The output is not a valid utterance, is gibberish, or is completely in the wrong format.
Score 2: The output is mostly not a spoken utterance; it is mainly a caption, third-person report, description, analysis, or label.
Score 3: The output is partly utterance-like, but noticeably unnatural, caption-like, written, or not clearly speakable.
Score 4: The output is a mostly natural spoken utterance, with only minor stiffness or formatting issues.
Score 5: The output is a natural, fluent, clearly speakable human utterance.

###Feedback:
```

##### `reference_anchor_compatibility`

- Criterion boundary: Judge compatibility with the paired dataset reference as a soft anchor, not exact wording or exact scene matching.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input audio caption:
[INPUT AUDIO CAPTION PLACEHOLDER]

Paired dataset reference speech transcript:
[PAIRED DATASET REFERENCE SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated speech transcript / human utterance.

The reference is a soft anchor, not an exact gold answer. The generated utterance does not need to match the reference wording or express the exact same sentence. Because the task is one-to-many, different utterances may be valid if they are compatible with the input audio scene and the reference's core situation, reaction, intent, or communicative role.

Evaluate only the specified criterion. Do not evaluate overall quality.

###Response to evaluate:
[GENERATED SPEECH TRANSCRIPT / HUMAN UTTERANCE PLACEHOLDER]

###Score Rubric:
Criterion: reference_anchor_compatibility

Criterion boundary:
Judge compatibility with the paired dataset reference as a soft anchor, not exact wording or exact scene matching.

Rubric:
Evaluate whether the generated utterance is compatible with the paired dataset reference as a soft semantic and pragmatic anchor.

The generated utterance should preserve or remain compatible with the reference's core situation, human reaction, intent, emotional stance, or communicative function when those are clear. Do not require exact wording, exact length, exact style, or exact semantic matching. Penalize only when the generated utterance ignores, conflicts with, or drifts away from the main anchor provided by the reference.

Score 1: The generated utterance is incompatible with the reference's core situation, reaction, intent, or communicative role.
Score 2: The generated utterance has only a weak or strained compatibility with the reference, and misses or conflicts with an important reference anchor.
Score 3: The generated utterance is partially compatible with the reference, but changes, weakens, or omits a noticeable part of the reference's core anchor.
Score 4: The generated utterance is mostly compatible with the reference's core anchor, with only minor differences in emphasis, specificity, emotion, or wording.
Score 5: The generated utterance is strongly compatible with the reference's core anchor while allowing natural variation in wording, phrasing, and details.

###Feedback:
```

#### S2A / speech_to_audio

##### `non_contradiction`

- Criterion boundary: Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: non_contradiction

Criterion boundary:
Judge only whether the response conflicts with the input. Do not lower the score merely because the response is weakly related, generic, missing details, or contains plausible unsupported details. A generic or indirect response with no actual conflict should usually score high on this criterion. Unsupported detail matters here only when it creates a genuine possible, likely, or actual conflict with the input.

Rubric:
Evaluate whether the generated audio caption avoids contradicting the input utterance.

Focus only on contradiction. The audio caption may add reasonable unstated context and may use an implied reading of the utterance. Do not penalize the caption merely because the link is generic, indirect, or not explicitly named in the utterance. Penalize only when it denies, reverses, or conflicts with explicit or strongly implied facts, intent, or context from the utterance.

Score 1: The audio caption directly and severely contradicts the utterance, or is incoherent with it.
Score 2: The audio caption contains a major contradiction with the utterance or with a strongly implied fact from it.
Score 3: The audio caption has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The audio caption is mostly compatible with the utterance, with only minor ambiguity or weak tension.
Score 5: The audio caption is fully compatible with the utterance and contains no meaningful contradiction.

###Feedback:
```

##### `conditional_relevance`

- Criterion boundary: Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: conditional_relevance

Criterion boundary:
Judge only whether the response is meaningfully grounded in the direction-specific input. Accept indirect, pragmatic, causal, or common-sense links; do not require explicit mention of the sound source. Do not score target-form validity, scenario plausibility, or appropriate specificity here.

Rubric:
Evaluate whether the generated audio caption is meaningfully grounded in the input utterance.

The audio scene should contextualize, accompany, explain, or be caused by the utterance. The connection may be indirect through common-sense implications, domain words, actions, consequences, or pragmatic context. Do not force an unrelated interpretation when an obvious reading creates a strong audio-scene link. The audio caption does not need to be the only possible scene.

Score 1: The audio caption is unrelated to the utterance.
Score 2: The audio caption has only a very weak, vague, or accidental connection to the utterance.
Score 3: The audio caption has a plausible but generic, indirect, or underdeveloped connection to the utterance.
Score 4: The audio caption is clearly connected to the utterance through an explicit or implicit situation, with only minor genericness or missed opportunity.
Score 5: The audio caption is strongly grounded in the utterance through a natural explicit or implicit situation.

###Feedback:
```

##### `scenario_plausibility`

- Criterion boundary: Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: scenario_plausibility

Criterion boundary:
Judge only whether the response and its minimal implied situation are believable. Do not score grounding strength or exact input relevance here; those belong to conditional_relevance.

Rubric:
Evaluate whether the generated audio caption and its minimally implied scene are believable as an audio situation.

This criterion is about plausibility of the described audio scene, not about how strongly it is grounded in the input; grounding is evaluated by conditional_relevance. Consider temporal, causal, emotional, pragmatic, and acoustic-world fit only when they make the described scene believable or unbelievable. Do not assign a very low score merely because the caption is simple, generic, or indirectly linked.

Score 1: The described audio scene is absurd, impossible, incoherent, or extremely unnatural.
Score 2: The described audio scene is difficult to believe because of a major temporal, causal, pragmatic, or acoustic-world problem.
Score 3: The described audio scene is possible but somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The described audio scene is plausible, requiring at most a small reasonable assumption.
Score 5: The described audio scene feels very natural and believable.

###Feedback:
```

##### `appropriate_specificity`

- Criterion boundary: Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: appropriate_specificity

Criterion boundary:
Judge only whether the response has an appropriate level of specificity: neither too vague nor too over-specific for an open-ended one-to-many task. Do not require the response to name the sound source.

Rubric:
Evaluate whether the generated audio caption has an appropriate level of specificity: not too vague, and not too over-specific.

The audio caption may infer reasonable context from the utterance because the task is one-to-many. It does not need to restate the utterance. A short caption can have acceptable specificity if it names a useful sound event, source, or acoustic action. Penalize only when the caption is nearly empty or so generic that it gives no usable audio scene, or when it invents highly specific unsupported details, identities, locations, causes, or events.

Score 1: The audio caption is nearly empty, has no usable sound-scene content, or is wildly over-invented with unsupported details.
Score 2: The audio caption is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The audio caption is simple or generic but still gives a usable sound event, source, or acoustic action; or it includes somewhat unsupported detail.
Score 4: The audio caption has mostly appropriate specificity, with only minor vagueness or minor over-inference.
Score 5: The audio caption is appropriately specific: concrete enough to be useful, but not over-invented beyond reasonable inference from the utterance.

###Feedback:
```

##### `target_form_validity`

- Criterion boundary: Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

The response to evaluate is a generated environmental audio caption. It should describe an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input utterance.

This is a one-to-many task. A valid audio caption may connect to the utterance indirectly through a plausible situation, action, consequence, implication, or shared context. Use common-sense readings of the utterance; do not require it to explicitly name every sound source.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not require exact wording from any reference.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: target_form_validity

Criterion boundary:
Judge only whether the generated text has the required direction-specific target form, either a speech transcript / human utterance or an environmental audio caption. Ignore whether it is semantically aligned with the input.

Rubric:
Evaluate whether the generated text is valid as an environmental audio caption.

Judge only form, not semantic alignment with the input utterance. The output should describe audible events, sound sources, acoustic actions, or sound context. Sound-producing actions such as something being shot, flushed, knocked, loaded, opened, started, or running can be valid audio-caption wording when the event is audibly recognizable. It should not merely infer the speaker's feelings, continue the speech, answer the speaker, or explain the situation without describing sound.

Score 1: The output is not a valid audio caption, is gibberish, or is completely in the wrong format.
Score 2: The output is mostly not an audio caption; it is mainly a speech continuation, explanation, emotion label, visual-only scene, or analysis.
Score 3: The output is partly audio-caption-like, but includes substantial non-audio content or unnatural caption style.
Score 4: The output is a mostly valid environmental audio caption, with only minor style issues.
Score 5: The output is a natural, clear environmental audio caption describing audible events, sound sources, acoustic actions, or acoustic context.

###Feedback:
```

##### `reference_anchor_compatibility`

- Criterion boundary: Judge compatibility with the paired dataset reference as a soft anchor, not exact wording or exact scene matching.

```text
###Task Description:
An instruction, a response to evaluate, and a score rubric representing one evaluation criterion are given.
1. Write clear, objective feedback assessing the response strictly according to the given score rubric, not in general.
2. After writing the feedback, write a score that is an integer between 1 and 5.
3. The output format should be exactly:
Feedback: (write feedback for the criterion) [RESULT] (integer between 1 and 5)
4. Do not generate any other opening, closing, or explanation.
5. If the criterion genuinely cannot be assessed from the response, write [RESULT] N/A instead of an integer. Do not use N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an intermediate text transformation for SpeechAudio LM.

Input speech transcript / utterance:
[INPUT SPEECH TRANSCRIPT PLACEHOLDER]

Paired dataset reference audio caption:
[PAIRED DATASET REFERENCE AUDIO CAPTION PLACEHOLDER]

The response to evaluate is a generated environmental audio caption.

The reference is a soft anchor, not an exact gold answer. The generated audio caption does not need to match the reference wording or describe the exact same scene in every detail. Because the task is one-to-many, different audio scenes may be valid if they are compatible with the input utterance and the reference's core acoustic situation, sound event, environment, cause, or contextual role.

Evaluate only the specified criterion. Do not evaluate overall quality.

###Response to evaluate:
[GENERATED ENVIRONMENTAL AUDIO CAPTION PLACEHOLDER]

###Score Rubric:
Criterion: reference_anchor_compatibility

Criterion boundary:
Judge compatibility with the paired dataset reference as a soft anchor, not exact wording or exact scene matching.

Rubric:
Evaluate whether the generated audio caption is compatible with the paired dataset reference as a soft acoustic and contextual anchor.

The generated audio caption should preserve or remain compatible with the reference's core sound event, acoustic environment, causal context, scene type, or contextual role when those are clear. Do not require exact wording, exact sound list, exact scene details, or exact semantic matching. Penalize only when the generated audio caption ignores, conflicts with, or drifts away from the main acoustic/contextual anchor provided by the reference.

Score 1: The generated audio caption is incompatible with the reference's core sound event, acoustic environment, causal context, or scene role.
Score 2: The generated audio caption has only a weak or strained compatibility with the reference, and misses or conflicts with an important reference anchor.
Score 3: The generated audio caption is partially compatible with the reference, but changes, weakens, or omits a noticeable part of the reference's core acoustic/contextual anchor.
Score 4: The generated audio caption is mostly compatible with the reference's core anchor, with only minor differences in emphasis, specificity, sound details, or wording.
Score 5: The generated audio caption is strongly compatible with the reference's core anchor while allowing natural variation in wording, sound detail, and scene description.

###Feedback:
```

## GEval Audio

- Evaluator version: `geval_audio_qwen3_criterion_v1`
- Template: `geval_audio_end_to_end_direct_assessment_v1`

### Plausibility

- Metric id: `geval_audio_plausibility`
- Prompt version: `geval_audio_plausibility_v4_directional_clip_labels`

#### A2S / audio_to_speech

##### `cross_modal_non_contradiction`

- Prompt-facing criterion name: `output_speech_non_contradiction_with_input_environment`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_non_contradiction_with_input_environment
Internal criterion id: cross_modal_non_contradiction

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

Rubric:
Evaluate whether the generated speech audio avoids contradicting the heard input environmental audio.

Focus only on contradiction. The speech may add reasonable unstated context, but it should not deny, reverse, or conflict with explicit or strongly implied events, conditions, or situations heard in the input audio.

Score 1: The generated speech directly and severely contradicts the input audio scene, or is incoherent with it.
Score 2: The generated speech contains a major contradiction with the input audio scene or with a strongly implied fact from it.
Score 3: The generated speech has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The generated speech is mostly compatible with the input audio scene, with only minor ambiguity or weak tension.
Score 5: The generated speech is fully compatible with the input audio scene and contains no meaningful contradiction.

###Feedback:
```

##### `cross_modal_grounding`

- Prompt-facing criterion name: `output_speech_grounding_in_input_environment`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_grounding_in_input_environment
Internal criterion id: cross_modal_grounding

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

Rubric:
Evaluate whether the generated speech audio is meaningfully grounded in the heard input environmental audio.

The speech should use, respond to, or be motivated by the sound event, acoustic context, or situation heard in the input audio. It does not need to describe the audio literally, but it should not be generic, unrelated, or only accidentally compatible.

Score 1: The generated speech is unrelated to the input audio.
Score 2: The generated speech has only a very weak, vague, or accidental connection to the input audio.
Score 3: The generated speech is somewhat connected, but the connection is generic, indirect, or underdeveloped.
Score 4: The generated speech is clearly connected to the input audio, with only minor genericness or missed opportunity.
Score 5: The generated speech is strongly and specifically grounded in the input audio.

###Feedback:
```

##### `scenario_plausibility`

- Prompt-facing criterion name: `input_environment_and_output_speech_scenario_plausibility`
- Criterion boundary: Judge only whether Clip 1 and Clip 2 form a believable shared situation.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: input_environment_and_output_speech_scenario_plausibility
Internal criterion id: scenario_plausibility

Criterion boundary:
Judge only whether Clip 1 and Clip 2 form a believable shared situation.

Rubric:
Evaluate whether the input environmental audio and generated speech audio form a believable shared situation.

The generated speech should sound like something a human could plausibly say during, after, or because of the heard audio scene. Consider temporal fit, causal fit, contextual emotional fit, pragmatic fit, and human reaction plausibility. These are not separate sub-scores; use them only to judge overall scenario plausibility.

Score 1: The input audio and generated speech form an absurd, impossible, incoherent, or extremely unnatural situation.
Score 2: The generated speech is difficult to believe in the heard scene because of a major temporal, causal, emotional, pragmatic, or human-reaction mismatch.
Score 3: The generated speech could occur in the heard scene, but the situation feels somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The generated speech forms a plausible human situation with the heard scene, requiring at most a small reasonable assumption.
Score 5: The generated speech feels like a very natural and believable human reaction to the heard scene.

###Feedback:
```

##### `controlled_detail`

- Prompt-facing criterion name: `output_speech_controlled_detail`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_controlled_detail
Internal criterion id: controlled_detail

Criterion boundary:
Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

Rubric:
Evaluate whether the generated speech audio uses an appropriate amount of inferred detail: not too vague, and not too over-specific.

The speech may infer reasonable context from the heard input audio because the task is one-to-many. However, it should not invent highly specific unsupported details, identities, locations, causes, or events. It should also not be so vague that it barely uses the input audio.

Score 1: The generated speech is either almost empty/generic or wildly over-invented with unsupported details.
Score 2: The generated speech is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The generated speech has some useful detail, but is noticeably too generic or includes somewhat unsupported details.
Score 4: The generated speech has mostly appropriate detail, with only minor vagueness or minor over-inference.
Score 5: The generated speech is appropriately detailed: concrete enough to be useful, but not over-invented beyond the input audio.

###Feedback:
```

##### `target_modality_success`

- Prompt-facing criterion name: `output_speech_modality_success`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_modality_success
Internal criterion id: target_modality_success

Criterion boundary:
Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

Rubric:
Evaluate whether the output audio succeeds as generated human speech.

The output should contain intelligible, speakable human speech. It should not be only environmental sound, music, silence, noise, a non-speech vocalization, a spoken audio caption, or an incoherent waveform. Judge the success of the target modality, not whether the content is semantically perfect.

Score 1: The output is not recognizable as valid human speech, is silent/noise, or is completely in the wrong modality.
Score 2: The output contains little usable speech, or the speech is mostly unintelligible, broken, or not a plausible utterance.
Score 3: The output is partly valid speech, but has noticeable intelligibility, modality, or utterance-form problems.
Score 4: The output is mostly successful speech audio, with only minor intelligibility or utterance-form issues.
Score 5: The output is clearly successful human speech audio: intelligible, speakable, and in the correct target modality.

###Feedback:
```

##### `perceptual_audio_quality`

- Prompt-facing criterion name: `output_speech_perceptual_quality`
- Criterion boundary: Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_perceptual_quality
Internal criterion id: perceptual_audio_quality

Criterion boundary:
Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

Rubric:
Evaluate the perceptual quality and naturalness of the generated speech audio.

Focus on audio quality, not semantic alignment. Consider naturalness, clarity, distortion, clipping, glitches, robotic artifacts, unstable voice, excessive noise, unnatural pauses, and synthesis artifacts.

Score 1: The generated speech audio is extremely degraded, unnatural, distorted, or difficult to listen to.
Score 2: The generated speech audio has major quality problems such as strong artifacts, distortion, instability, or poor clarity.
Score 3: The generated speech audio is understandable but has noticeable artifacts, unnaturalness, noise, or synthesis issues.
Score 4: The generated speech audio is mostly natural and clear, with only minor artifacts or quality issues.
Score 5: The generated speech audio is highly natural, clear, stable, and free of meaningful perceptual artifacts.

###Feedback:
```

##### `expressive_delivery_fit`

- Prompt-facing criterion name: `output_speech_expressive_delivery_fit`
- Criterion boundary: Use Clip 1 as context and judge only non-textual expressive delivery fit of Clip 2, the generated output speech.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_expressive_delivery_fit
Internal criterion id: expressive_delivery_fit

Criterion boundary:
Use Clip 1 as context and judge only non-textual expressive delivery fit of Clip 2, the generated output speech.

Rubric:
Evaluate whether the expressive delivery of the generated speech fits the heard input audio scene.

Focus on non-textual speech cues: emotion, prosody, tone, urgency, loudness, speaking pace, rhythm, hesitation, stress, surprise, fear, calmness, laughter, or other vocal expression. The words may be plausible, but the delivery should also fit the situation implied by the input audio. Do not require strong emotion when the situation is neutral.

Score 1: The expressive delivery is completely inappropriate for the heard scene, or the speech has no usable expressive character.
Score 2: The delivery has a major mismatch in emotion, prosody, urgency, loudness, pace, or tone.
Score 3: The delivery is somewhat plausible, but feels flat, exaggerated, emotionally mismatched, or weakly adapted to the scene.
Score 4: The delivery mostly fits the heard scene, with only minor expressive mismatch or missed nuance.
Score 5: The delivery strongly fits the heard scene, with natural emotion, prosody, urgency, pace, and tone for the situation.

###Feedback:
```

#### S2A / speech_to_audio

##### `cross_modal_non_contradiction`

- Prompt-facing criterion name: `output_environmental_audio_non_contradiction_with_input_speech`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_non_contradiction_with_input_speech
Internal criterion id: cross_modal_non_contradiction

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

Rubric:
Evaluate whether the generated environmental audio avoids contradicting the heard input speech.

Focus only on contradiction. The environmental audio may add reasonable unstated context, but it should not deny, reverse, or conflict with explicit or strongly implied facts, intent, emotion, or context from the spoken utterance.

Score 1: The generated environmental audio directly and severely contradicts the input speech, or is incoherent with it.
Score 2: The generated environmental audio contains a major contradiction with the input speech or with a strongly implied fact from it.
Score 3: The generated environmental audio has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The generated environmental audio is mostly compatible with the input speech, with only minor ambiguity or weak tension.
Score 5: The generated environmental audio is fully compatible with the input speech and contains no meaningful contradiction.

###Feedback:
```

##### `cross_modal_grounding`

- Prompt-facing criterion name: `output_environmental_audio_grounding_in_input_speech`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_grounding_in_input_speech
Internal criterion id: cross_modal_grounding

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

Rubric:
Evaluate whether the generated environmental audio is meaningfully grounded in the heard input speech.

The audio scene should contextualize, accompany, explain, or be caused by the spoken utterance. It does not need to be the only possible scene, but it should not be generic, unrelated, or only accidentally compatible.

Score 1: The generated environmental audio is unrelated to the input speech.
Score 2: The generated environmental audio has only a very weak, vague, or accidental connection to the input speech.
Score 3: The generated environmental audio is somewhat connected, but the connection is generic, indirect, or underdeveloped.
Score 4: The generated environmental audio is clearly connected to the input speech, with only minor genericness or missed opportunity.
Score 5: The generated environmental audio is strongly and specifically grounded in the input speech.

###Feedback:
```

##### `scenario_plausibility`

- Prompt-facing criterion name: `input_speech_and_output_environmental_audio_scenario_plausibility`
- Criterion boundary: Judge only whether Clip 1 and Clip 2 form a believable shared situation.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: input_speech_and_output_environmental_audio_scenario_plausibility
Internal criterion id: scenario_plausibility

Criterion boundary:
Judge only whether Clip 1 and Clip 2 form a believable shared situation.

Rubric:
Evaluate whether the input speech and generated environmental audio form a believable shared situation.

The generated audio scene should realistically contextualize, accompany, explain, or result from the heard utterance. Consider temporal fit, causal fit, contextual emotional fit, pragmatic fit, and whether the described sound world is believable. These are not separate sub-scores; use them only to judge overall scenario plausibility.

Score 1: The input speech and generated environmental audio form an absurd, impossible, incoherent, or extremely unnatural situation.
Score 2: The generated audio scene is difficult to believe with the input speech because of a major temporal, causal, emotional, pragmatic, or sound-world mismatch.
Score 3: The generated audio scene could fit the input speech, but the situation feels somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The generated audio scene forms a plausible situation with the input speech, requiring at most a small reasonable assumption.
Score 5: The generated audio scene feels like a very natural and believable context, accompaniment, explanation, or consequence of the input speech.

###Feedback:
```

##### `controlled_detail`

- Prompt-facing criterion name: `output_environmental_audio_controlled_detail`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_controlled_detail
Internal criterion id: controlled_detail

Criterion boundary:
Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

Rubric:
Evaluate whether the generated environmental audio uses an appropriate amount of acoustic detail: not too sparse, and not too overloaded or over-specific.

The audio may infer reasonable context from the heard input speech because the task is one-to-many. However, it should not introduce many unsupported or overly specific sound events, identities, locations, or causes. It should also not be so sparse or generic that it barely uses the input speech.

Score 1: The generated audio is either almost empty/generic or wildly overloaded with unsupported acoustic details.
Score 2: The generated audio is mostly too sparse/generic or mostly too over-specific/overcrowded, making it poorly controlled.
Score 3: The generated audio has some useful acoustic detail, but is noticeably too generic, too sparse, or somewhat over-invented.
Score 4: The generated audio has mostly appropriate acoustic detail, with only minor sparseness or minor over-inference.
Score 5: The generated audio is appropriately detailed: rich enough to be useful, but not overloaded or over-invented beyond the input speech.

###Feedback:
```

##### `target_modality_success`

- Prompt-facing criterion name: `output_environmental_audio_modality_success`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_modality_success
Internal criterion id: target_modality_success

Criterion boundary:
Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

Rubric:
Evaluate whether the output audio succeeds as generated environmental audio.

The output should contain an environmental sound scene, sound event, ambience, or acoustic context. It should not be generated speech replying to the speaker, a spoken caption, pure silence, unrelated music, random noise, or an incoherent waveform. Judge the success of the target modality, not whether the semantic alignment is perfect.

Score 1: The output is not recognizable as valid environmental audio, is silent/noise, or is completely in the wrong modality.
Score 2: The output contains little usable environmental audio, or is mostly speech, music, silence, random noise, or broken audio.
Score 3: The output is partly valid environmental audio, but has noticeable modality, scene, or acoustic-content problems.
Score 4: The output is mostly successful environmental audio, with only minor modality or acoustic-content issues.
Score 5: The output is clearly successful environmental audio: audible, scene-like, and in the correct target modality.

###Feedback:
```

##### `perceptual_audio_quality`

- Prompt-facing criterion name: `output_environmental_audio_perceptual_quality`
- Criterion boundary: Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_perceptual_quality
Internal criterion id: perceptual_audio_quality

Criterion boundary:
Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

Rubric:
Evaluate the perceptual quality and realism of the generated environmental audio.

Focus on audio quality, not semantic alignment. Consider realism, clarity, natural sound texture, distortion, clipping, glitches, looping artifacts, harsh synthetic noise, poor mixing, unstable ambience, and unnatural transitions.

Score 1: The generated environmental audio is extremely degraded, unrealistic, distorted, or difficult to listen to.
Score 2: The generated environmental audio has major quality problems such as strong artifacts, distortion, poor realism, or unstable sound texture.
Score 3: The generated environmental audio is usable but has noticeable artifacts, unnaturalness, poor mixing, looping, or synthetic texture.
Score 4: The generated environmental audio is mostly realistic and clear, with only minor artifacts or quality issues.
Score 5: The generated environmental audio is highly realistic, clear, coherent, and free of meaningful perceptual artifacts.

###Feedback:
```

##### `contextual_acoustic_fit`

- Prompt-facing criterion name: `output_environmental_audio_contextual_acoustic_fit`
- Criterion boundary: Use Clip 1 as context and judge only non-textual acoustic character fit of Clip 2, the generated output environmental audio.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_contextual_acoustic_fit
Internal criterion id: contextual_acoustic_fit

Criterion boundary:
Use Clip 1 as context and judge only non-textual acoustic character fit of Clip 2, the generated output environmental audio.

Rubric:
Evaluate whether the acoustic character of the generated environmental audio fits the heard input speech.

Focus on non-textual acoustic cues: loudness, intensity, ambience, mood, tension, calmness, event timing, foreground/background balance, density, and soundscape energy. The audio scene may be semantically plausible, but its acoustic character should also fit the utterance's intent, emotion, urgency, and pragmatic context. Do not require dramatic sound when the speech implies a neutral or quiet situation.

Score 1: The acoustic character is completely inappropriate for the heard speech, or the output has no usable acoustic character.
Score 2: The generated audio has a major mismatch in loudness, intensity, ambience, timing, density, or acoustic mood.
Score 3: The generated audio is somewhat plausible, but feels too flat, too intense, emotionally mismatched, poorly timed, or weakly adapted to the speech.
Score 4: The generated audio mostly fits the heard speech, with only minor acoustic mismatch or missed nuance.
Score 5: The generated audio strongly fits the heard speech, with natural loudness, intensity, ambience, timing, density, and acoustic mood for the situation.

###Feedback:
```

### Reference-Grounded

- Metric id: `geval_audio_reference`
- Prompt version: `geval_audio_reference_v4_directional_clip_labels`

#### A2S / audio_to_speech

##### `cross_modal_non_contradiction`

- Prompt-facing criterion name: `output_speech_non_contradiction_with_input_environment`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_non_contradiction_with_input_environment
Internal criterion id: cross_modal_non_contradiction

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

Rubric:
Evaluate whether the generated speech audio avoids contradicting the heard input environmental audio.

Focus only on contradiction. The speech may add reasonable unstated context, but it should not deny, reverse, or conflict with explicit or strongly implied events, conditions, or situations heard in the input audio.

Score 1: The generated speech directly and severely contradicts the input audio scene, or is incoherent with it.
Score 2: The generated speech contains a major contradiction with the input audio scene or with a strongly implied fact from it.
Score 3: The generated speech has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The generated speech is mostly compatible with the input audio scene, with only minor ambiguity or weak tension.
Score 5: The generated speech is fully compatible with the input audio scene and contains no meaningful contradiction.

###Feedback:
```

##### `cross_modal_grounding`

- Prompt-facing criterion name: `output_speech_grounding_in_input_environment`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_grounding_in_input_environment
Internal criterion id: cross_modal_grounding

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

Rubric:
Evaluate whether the generated speech audio is meaningfully grounded in the heard input environmental audio.

The speech should use, respond to, or be motivated by the sound event, acoustic context, or situation heard in the input audio. It does not need to describe the audio literally, but it should not be generic, unrelated, or only accidentally compatible.

Score 1: The generated speech is unrelated to the input audio.
Score 2: The generated speech has only a very weak, vague, or accidental connection to the input audio.
Score 3: The generated speech is somewhat connected, but the connection is generic, indirect, or underdeveloped.
Score 4: The generated speech is clearly connected to the input audio, with only minor genericness or missed opportunity.
Score 5: The generated speech is strongly and specifically grounded in the input audio.

###Feedback:
```

##### `scenario_plausibility`

- Prompt-facing criterion name: `input_environment_and_output_speech_scenario_plausibility`
- Criterion boundary: Judge only whether Clip 1 and Clip 2 form a believable shared situation.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: input_environment_and_output_speech_scenario_plausibility
Internal criterion id: scenario_plausibility

Criterion boundary:
Judge only whether Clip 1 and Clip 2 form a believable shared situation.

Rubric:
Evaluate whether the input environmental audio and generated speech audio form a believable shared situation.

The generated speech should sound like something a human could plausibly say during, after, or because of the heard audio scene. Consider temporal fit, causal fit, contextual emotional fit, pragmatic fit, and human reaction plausibility. These are not separate sub-scores; use them only to judge overall scenario plausibility.

Score 1: The input audio and generated speech form an absurd, impossible, incoherent, or extremely unnatural situation.
Score 2: The generated speech is difficult to believe in the heard scene because of a major temporal, causal, emotional, pragmatic, or human-reaction mismatch.
Score 3: The generated speech could occur in the heard scene, but the situation feels somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The generated speech forms a plausible human situation with the heard scene, requiring at most a small reasonable assumption.
Score 5: The generated speech feels like a very natural and believable human reaction to the heard scene.

###Feedback:
```

##### `controlled_detail`

- Prompt-facing criterion name: `output_speech_controlled_detail`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_controlled_detail
Internal criterion id: controlled_detail

Criterion boundary:
Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

Rubric:
Evaluate whether the generated speech audio uses an appropriate amount of inferred detail: not too vague, and not too over-specific.

The speech may infer reasonable context from the heard input audio because the task is one-to-many. However, it should not invent highly specific unsupported details, identities, locations, causes, or events. It should also not be so vague that it barely uses the input audio.

Score 1: The generated speech is either almost empty/generic or wildly over-invented with unsupported details.
Score 2: The generated speech is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The generated speech has some useful detail, but is noticeably too generic or includes somewhat unsupported details.
Score 4: The generated speech has mostly appropriate detail, with only minor vagueness or minor over-inference.
Score 5: The generated speech is appropriately detailed: concrete enough to be useful, but not over-invented beyond the input audio.

###Feedback:
```

##### `target_modality_success`

- Prompt-facing criterion name: `output_speech_modality_success`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_modality_success
Internal criterion id: target_modality_success

Criterion boundary:
Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

Rubric:
Evaluate whether the output audio succeeds as generated human speech.

The output should contain intelligible, speakable human speech. It should not be only environmental sound, music, silence, noise, a non-speech vocalization, a spoken audio caption, or an incoherent waveform. Judge the success of the target modality, not whether the content is semantically perfect.

Score 1: The output is not recognizable as valid human speech, is silent/noise, or is completely in the wrong modality.
Score 2: The output contains little usable speech, or the speech is mostly unintelligible, broken, or not a plausible utterance.
Score 3: The output is partly valid speech, but has noticeable intelligibility, modality, or utterance-form problems.
Score 4: The output is mostly successful speech audio, with only minor intelligibility or utterance-form issues.
Score 5: The output is clearly successful human speech audio: intelligible, speakable, and in the correct target modality.

###Feedback:
```

##### `perceptual_audio_quality`

- Prompt-facing criterion name: `output_speech_perceptual_quality`
- Criterion boundary: Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_perceptual_quality
Internal criterion id: perceptual_audio_quality

Criterion boundary:
Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

Rubric:
Evaluate the perceptual quality and naturalness of the generated speech audio.

Focus on audio quality, not semantic alignment. Consider naturalness, clarity, distortion, clipping, glitches, robotic artifacts, unstable voice, excessive noise, unnatural pauses, and synthesis artifacts.

Score 1: The generated speech audio is extremely degraded, unnatural, distorted, or difficult to listen to.
Score 2: The generated speech audio has major quality problems such as strong artifacts, distortion, instability, or poor clarity.
Score 3: The generated speech audio is understandable but has noticeable artifacts, unnaturalness, noise, or synthesis issues.
Score 4: The generated speech audio is mostly natural and clear, with only minor artifacts or quality issues.
Score 5: The generated speech audio is highly natural, clear, stable, and free of meaningful perceptual artifacts.

###Feedback:
```

##### `expressive_delivery_fit`

- Prompt-facing criterion name: `output_speech_expressive_delivery_fit`
- Criterion boundary: Use Clip 1 as context and judge only non-textual expressive delivery fit of Clip 2, the generated output speech.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_expressive_delivery_fit
Internal criterion id: expressive_delivery_fit

Criterion boundary:
Use Clip 1 as context and judge only non-textual expressive delivery fit of Clip 2, the generated output speech.

Rubric:
Evaluate whether the expressive delivery of the generated speech fits the heard input audio scene.

Focus on non-textual speech cues: emotion, prosody, tone, urgency, loudness, speaking pace, rhythm, hesitation, stress, surprise, fear, calmness, laughter, or other vocal expression. The words may be plausible, but the delivery should also fit the situation implied by the input audio. Do not require strong emotion when the situation is neutral.

Score 1: The expressive delivery is completely inappropriate for the heard scene, or the speech has no usable expressive character.
Score 2: The delivery has a major mismatch in emotion, prosody, urgency, loudness, pace, or tone.
Score 3: The delivery is somewhat plausible, but feels flat, exaggerated, emotionally mismatched, or weakly adapted to the scene.
Score 4: The delivery mostly fits the heard scene, with only minor expressive mismatch or missed nuance.
Score 5: The delivery strongly fits the heard scene, with natural emotion, prosody, urgency, pace, and tone for the situation.

###Feedback:
```

##### `reference_audio_compatibility`

- Prompt-facing criterion name: `output_speech_reference_audio_compatibility`
- Criterion boundary: Use reference clips as soft examples and judge compatibility of Clip 2, the generated output clip. Do not require exact matching.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation with reference audio anchors. Listen carefully to the ordered clips.

Clip 1 is the input environmental audio scene. Clip 2 is the generated output speech audio to evaluate. Reference speech audio anchors, when present, start at Clip 3. Use Clip 1 and any reference anchors as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output speech audio should be generated human speech that could plausibly be spoken in, after, or because of the input environmental audio scene, and should remain compatible with the reference speech audio anchors as soft examples of valid output space.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_input_environmental_audio.wav
Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated_output_speech_audio.wav
Clip 3 - REFERENCE SPEECH AUDIO ANCHOR 1: attached file reference_audio_1_reference_speech_audio_anchor_1.wav

###Score Rubric:
Criterion: output_speech_reference_audio_compatibility
Internal criterion id: reference_audio_compatibility

Criterion boundary:
Use reference clips as soft examples and judge compatibility of Clip 2, the generated output clip. Do not require exact matching.

Rubric:
Evaluate whether the generated speech audio is compatible with the attached reference speech audio anchor(s) as soft examples of valid responses.

The generated speech does not need to match the exact wording, voice, timing, or delivery of the reference audio. It should remain compatible with the reference's core audible situation, communicative role, reaction, intent, or emotional stance when those are clear.

Score 1: The generated speech is incompatible with the reference audio's core response, reaction, intent, or communicative role.
Score 2: The generated speech has only weak or strained compatibility with the reference audio and misses or conflicts with an important anchor.
Score 3: The generated speech is partially compatible with the reference audio, but changes, weakens, or omits a noticeable part of the reference anchor.
Score 4: The generated speech is mostly compatible with the reference audio's core anchor, with only minor differences in emphasis, specificity, emotion, wording, or delivery.
Score 5: The generated speech is strongly compatible with the reference audio's core anchor while allowing natural variation in wording, phrasing, voice, and delivery.

###Feedback:
```

#### S2A / speech_to_audio

##### `cross_modal_non_contradiction`

- Prompt-facing criterion name: `output_environmental_audio_non_contradiction_with_input_speech`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_non_contradiction_with_input_speech
Internal criterion id: cross_modal_non_contradiction

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, contradicts it.

Rubric:
Evaluate whether the generated environmental audio avoids contradicting the heard input speech.

Focus only on contradiction. The environmental audio may add reasonable unstated context, but it should not deny, reverse, or conflict with explicit or strongly implied facts, intent, emotion, or context from the spoken utterance.

Score 1: The generated environmental audio directly and severely contradicts the input speech, or is incoherent with it.
Score 2: The generated environmental audio contains a major contradiction with the input speech or with a strongly implied fact from it.
Score 3: The generated environmental audio has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The generated environmental audio is mostly compatible with the input speech, with only minor ambiguity or weak tension.
Score 5: The generated environmental audio is fully compatible with the input speech and contains no meaningful contradiction.

###Feedback:
```

##### `cross_modal_grounding`

- Prompt-facing criterion name: `output_environmental_audio_grounding_in_input_speech`
- Criterion boundary: Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_grounding_in_input_speech
Internal criterion id: cross_modal_grounding

Criterion boundary:
Use Clip 1 as context and judge only whether Clip 2, the generated output clip, is meaningfully grounded in it.

Rubric:
Evaluate whether the generated environmental audio is meaningfully grounded in the heard input speech.

The audio scene should contextualize, accompany, explain, or be caused by the spoken utterance. It does not need to be the only possible scene, but it should not be generic, unrelated, or only accidentally compatible.

Score 1: The generated environmental audio is unrelated to the input speech.
Score 2: The generated environmental audio has only a very weak, vague, or accidental connection to the input speech.
Score 3: The generated environmental audio is somewhat connected, but the connection is generic, indirect, or underdeveloped.
Score 4: The generated environmental audio is clearly connected to the input speech, with only minor genericness or missed opportunity.
Score 5: The generated environmental audio is strongly and specifically grounded in the input speech.

###Feedback:
```

##### `scenario_plausibility`

- Prompt-facing criterion name: `input_speech_and_output_environmental_audio_scenario_plausibility`
- Criterion boundary: Judge only whether Clip 1 and Clip 2 form a believable shared situation.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: input_speech_and_output_environmental_audio_scenario_plausibility
Internal criterion id: scenario_plausibility

Criterion boundary:
Judge only whether Clip 1 and Clip 2 form a believable shared situation.

Rubric:
Evaluate whether the input speech and generated environmental audio form a believable shared situation.

The generated audio scene should realistically contextualize, accompany, explain, or result from the heard utterance. Consider temporal fit, causal fit, contextual emotional fit, pragmatic fit, and whether the described sound world is believable. These are not separate sub-scores; use them only to judge overall scenario plausibility.

Score 1: The input speech and generated environmental audio form an absurd, impossible, incoherent, or extremely unnatural situation.
Score 2: The generated audio scene is difficult to believe with the input speech because of a major temporal, causal, emotional, pragmatic, or sound-world mismatch.
Score 3: The generated audio scene could fit the input speech, but the situation feels somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The generated audio scene forms a plausible situation with the input speech, requiring at most a small reasonable assumption.
Score 5: The generated audio scene feels like a very natural and believable context, accompaniment, explanation, or consequence of the input speech.

###Feedback:
```

##### `controlled_detail`

- Prompt-facing criterion name: `output_environmental_audio_controlled_detail`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_controlled_detail
Internal criterion id: controlled_detail

Criterion boundary:
Judge only whether Clip 2, the generated output clip, uses an appropriate amount of inferred detail.

Rubric:
Evaluate whether the generated environmental audio uses an appropriate amount of acoustic detail: not too sparse, and not too overloaded or over-specific.

The audio may infer reasonable context from the heard input speech because the task is one-to-many. However, it should not introduce many unsupported or overly specific sound events, identities, locations, or causes. It should also not be so sparse or generic that it barely uses the input speech.

Score 1: The generated audio is either almost empty/generic or wildly overloaded with unsupported acoustic details.
Score 2: The generated audio is mostly too sparse/generic or mostly too over-specific/overcrowded, making it poorly controlled.
Score 3: The generated audio has some useful acoustic detail, but is noticeably too generic, too sparse, or somewhat over-invented.
Score 4: The generated audio has mostly appropriate acoustic detail, with only minor sparseness or minor over-inference.
Score 5: The generated audio is appropriately detailed: rich enough to be useful, but not overloaded or over-invented beyond the input speech.

###Feedback:
```

##### `target_modality_success`

- Prompt-facing criterion name: `output_environmental_audio_modality_success`
- Criterion boundary: Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_modality_success
Internal criterion id: target_modality_success

Criterion boundary:
Judge only whether Clip 2, the generated output clip, succeeds in the required target modality.

Rubric:
Evaluate whether the output audio succeeds as generated environmental audio.

The output should contain an environmental sound scene, sound event, ambience, or acoustic context. It should not be generated speech replying to the speaker, a spoken caption, pure silence, unrelated music, random noise, or an incoherent waveform. Judge the success of the target modality, not whether the semantic alignment is perfect.

Score 1: The output is not recognizable as valid environmental audio, is silent/noise, or is completely in the wrong modality.
Score 2: The output contains little usable environmental audio, or is mostly speech, music, silence, random noise, or broken audio.
Score 3: The output is partly valid environmental audio, but has noticeable modality, scene, or acoustic-content problems.
Score 4: The output is mostly successful environmental audio, with only minor modality or acoustic-content issues.
Score 5: The output is clearly successful environmental audio: audible, scene-like, and in the correct target modality.

###Feedback:
```

##### `perceptual_audio_quality`

- Prompt-facing criterion name: `output_environmental_audio_perceptual_quality`
- Criterion boundary: Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_perceptual_quality
Internal criterion id: perceptual_audio_quality

Criterion boundary:
Judge only perceptual audio quality and naturalness/realism of Clip 2, the generated output clip. Do not score Clip 1 quality or semantic alignment.

Rubric:
Evaluate the perceptual quality and realism of the generated environmental audio.

Focus on audio quality, not semantic alignment. Consider realism, clarity, natural sound texture, distortion, clipping, glitches, looping artifacts, harsh synthetic noise, poor mixing, unstable ambience, and unnatural transitions.

Score 1: The generated environmental audio is extremely degraded, unrealistic, distorted, or difficult to listen to.
Score 2: The generated environmental audio has major quality problems such as strong artifacts, distortion, poor realism, or unstable sound texture.
Score 3: The generated environmental audio is usable but has noticeable artifacts, unnaturalness, poor mixing, looping, or synthetic texture.
Score 4: The generated environmental audio is mostly realistic and clear, with only minor artifacts or quality issues.
Score 5: The generated environmental audio is highly realistic, clear, coherent, and free of meaningful perceptual artifacts.

###Feedback:
```

##### `contextual_acoustic_fit`

- Prompt-facing criterion name: `output_environmental_audio_contextual_acoustic_fit`
- Criterion boundary: Use Clip 1 as context and judge only non-textual acoustic character fit of Clip 2, the generated output environmental audio.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Use Clip 1 as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_contextual_acoustic_fit
Internal criterion id: contextual_acoustic_fit

Criterion boundary:
Use Clip 1 as context and judge only non-textual acoustic character fit of Clip 2, the generated output environmental audio.

Rubric:
Evaluate whether the acoustic character of the generated environmental audio fits the heard input speech.

Focus on non-textual acoustic cues: loudness, intensity, ambience, mood, tension, calmness, event timing, foreground/background balance, density, and soundscape energy. The audio scene may be semantically plausible, but its acoustic character should also fit the utterance's intent, emotion, urgency, and pragmatic context. Do not require dramatic sound when the speech implies a neutral or quiet situation.

Score 1: The acoustic character is completely inappropriate for the heard speech, or the output has no usable acoustic character.
Score 2: The generated audio has a major mismatch in loudness, intensity, ambience, timing, density, or acoustic mood.
Score 3: The generated audio is somewhat plausible, but feels too flat, too intense, emotionally mismatched, poorly timed, or weakly adapted to the speech.
Score 4: The generated audio mostly fits the heard speech, with only minor acoustic mismatch or missed nuance.
Score 5: The generated audio strongly fits the heard speech, with natural loudness, intensity, ambience, timing, density, and acoustic mood for the situation.

###Feedback:
```

##### `reference_audio_compatibility`

- Prompt-facing criterion name: `output_environmental_audio_reference_audio_compatibility`
- Criterion boundary: Use reference clips as soft examples and judge compatibility of Clip 2, the generated output clip. Do not require exact matching.

```text
###Task Description:
An instruction, ordered audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips in the listed order.
2. Treat Clip 1 as the input/context clip and Clip 2 as the generated output clip to evaluate. Reference clips, when present, come after Clip 2.
3. Write concise, objective feedback assessing the generated output clip strictly according to the given score rubric, not in general.
4. After writing the feedback, write a score that is an integer between 1 and 5.
5. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
6. Do not generate any other opening, closing, or explanation.
7. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the generated output clip is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
You are evaluating an end-to-end SpeechAudio LM generation with reference audio anchors. Listen carefully to the ordered clips.

Clip 1 is the input speech audio. Clip 2 is the generated output environmental audio to evaluate. Reference environmental audio anchors, when present, start at Clip 3. Use Clip 1 and any reference anchors as context, but for output-quality or target-modality criteria, score Clip 2 only.

The output environmental audio should be an audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance, and should remain compatible with the reference environmental audio anchors as soft examples of valid output space.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the ordered clips.

###Audio clips:
The attached clips are provided in this exact order:
Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_input_speech_audio.wav
Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated_output_environmental_audio.wav
Clip 3 - REFERENCE ENVIRONMENTAL AUDIO ANCHOR 1: attached file reference_audio_1_reference_environmental_audio_anchor_1.wav

###Score Rubric:
Criterion: output_environmental_audio_reference_audio_compatibility
Internal criterion id: reference_audio_compatibility

Criterion boundary:
Use reference clips as soft examples and judge compatibility of Clip 2, the generated output clip. Do not require exact matching.

Rubric:
Evaluate whether the generated environmental audio is compatible with the attached reference environmental audio anchor(s) as soft examples of valid output scenes.

The generated environmental audio does not need to match the exact same sound list, timing, recording conditions, or scene details. It should remain compatible with the reference's core sound event, acoustic environment, causal context, scene type, or contextual role when those are clear.

Score 1: The generated environmental audio is incompatible with the reference audio's core sound event, acoustic environment, causal context, or scene role.
Score 2: The generated environmental audio has only weak or strained compatibility with the reference audio and misses or conflicts with an important anchor.
Score 3: The generated environmental audio is partially compatible with the reference audio, but changes, weakens, or omits a noticeable part of the reference acoustic/contextual anchor.
Score 4: The generated environmental audio is mostly compatible with the reference audio's core anchor, with only minor differences in emphasis, specificity, sound details, timing, or texture.
Score 5: The generated environmental audio is strongly compatible with the reference audio's core anchor while allowing natural variation in sound detail, scene realization, and acoustic texture.

###Feedback:
```
