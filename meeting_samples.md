# Modular Baseline Meeting Samples

Generated from existing manifests under `/scratch/l/lovenya/salm_migration_clean/results`.
Duplicate manifest rows with the same sample/output path are collapsed.
Note: several current TamIA smokes were one-sample validations, so those sections currently have one available unique sample rather than two.

## A2T / captioning: `qwen3_omni_captioner`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_qwen3_caption_smoke/cascaded/A2T/audio_captioning-qwen3_omni_captioner/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: A motorcycle engine is idling
- Model text output: A loud, rapid, and rhythmic mechanical chugging dominates the audio, characteristic of a small, single-cylinder engine idling or running at low speed. The sound is harsh, percussive, and metallic, with a steady tempo and no variation in pitch or volume, suggesting a simple, utilitarian machine such as a generator, pump, or small tractor. The recording is of low fidelity, with significant digital clipping and distortion, and is presented in mono with no spatial cues or ambient background noise. The environment is acoustically neutral, likely an open outdoor space or a large, empty indoor area, and the absence of any human or environmental sounds creates a stark, isolated atmosphere. The mood is tense and mechanical, evoking a sense of industrial monotony and urgency, with the relentless engine sound dominating the listener’s attention.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_qwen3_caption_smoke/cascaded/A2T/audio_captioning-qwen3_omni_captioner/ttas/test/text/000000_0V0MxUBtx9g.json

## S2T / ASR: `whisper`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_fixed_adapter_retry/cascaded/S2T/asr-whisper/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_fixed_adapter_retry/cascaded/S2T/asr-whisper/ttas/test/text/000000_0V0MxUBtx9g.json

## T2T / text conversion: `qwen3_32b`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_text_llm_smoke/cascaded/T2T/text_to_text-qwen3_32b/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: A quiet outdoor scene with the faint hum of a vehicle engine idling, the metallic clink of adjusting mirrors, and the occasional rustle of wind or distant
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_text_llm_smoke/cascaded/T2T/text_to_text-qwen3_32b/ttas/test/text/000000_0V0MxUBtx9g.json

## T2T / text conversion: `llama3_3_70b_instruct`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_asr_llama_smoke/cascaded/T2T/text_to_text-llama3_3_70b_instruct/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: "Generate the sound of a car engine idling, with a driver adjusting side mirrors and preparing to depart, set in a quiet outdoor environment."
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_asr_llama_smoke/cascaded/T2T/text_to_text-llama3_3_70b_instruct/ttas/test/text/000000_0V0MxUBtx9g.json

## T2S / speech generation: `f5_tts`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_fixed_adapter_retry/cascaded/T2S/text_to_speech-f5_tts/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_fixed_adapter_retry/cascaded/T2S/text_to_speech-f5_tts/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 24000
- Reference speech file: /scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted/speech/0V0MxUBtx9g.wav

## T2S / speech generation: `qwen3_omni`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_smoke/cascaded/T2S/text_to_speech-qwen3_omni/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_cascaded_smoke/cascaded/T2S/text_to_speech-qwen3_omni/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 24000

## T2A / audio generation: `audiogen`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_kad2_smoke/audiogen/T2A/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 2

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: A motorcycle engine is idling
- Model text output: A motorcycle engine is idling
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_kad2_smoke/audiogen/T2A/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 16000

### Sample 2: `IAK4nfn5ofI`

- Dataset audio caption: The clopping of a horse, the distant roar of a car engine, and men shouting to each other
- Dataset speech transcript: Hector's ready to go, but that car's gonna be late. Tell Joe to check the fence by the barn. Should be here any minute.
- Input prompt: The clopping of a horse, the distant roar of a car engine, and men shouting to each other
- Model text output: The clopping of a horse, the distant roar of a car engine, and men shouting to each other
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_kad2_smoke/audiogen/T2A/ttas/test/audio/000001_IAK4nfn5ofI.wav
- Sample rate: 16000

## T2A / audio generation: `tangoflux`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_smoke/tangoflux/T2A/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: A motorcycle engine is idling
- Model text output: A motorcycle engine is idling
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_smoke/tangoflux/T2A/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 44100

## T2A / audio generation: `audio_omni`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_smoke/audio_omni/T2A/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: A motorcycle engine is idling
- Model text output: A motorcycle engine is idling
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_smoke/audio_omni/T2A/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 44100

## A2S / direct/cascade baseline: `qwen3_omni`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_smoke_retry/qwen3_omni/A2S/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 1

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_smoke_retry/qwen3_omni/A2S/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 24000

## S2A / direct/cascade baseline: `audio_omni`

- Manifest: `/scratch/l/lovenya/salm_migration_clean/results/baselines_kad2_smoke/audio_omni/S2A/ttas/test/manifest.jsonl`
- Available unique successful samples in this manifest: 2

### Sample 1: `0V0MxUBtx9g`

- Dataset audio caption: A motorcycle engine is idling
- Dataset speech transcript: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Input prompt: Need to adjust these mirrors before hitting the road. This engine's been idling too long.
- Model text output: The speaker is saying that they need to adjust the mirrors before driving because the engine has been idling too long. If you have any other questions or need further clarification, feel free to let me know.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_kad2_smoke/audio_omni/S2A/ttas/test/audio/000000_0V0MxUBtx9g.wav
- Sample rate: 44100

### Sample 2: `IAK4nfn5ofI`

- Dataset audio caption: The clopping of a horse, the distant roar of a car engine, and men shouting to each other
- Dataset speech transcript: Hector's ready to go, but that car's gonna be late. Tell Joe to check the fence by the barn. Should be here any minute.
- Input prompt: Hector's ready to go, but that car's gonna be late. Tell Joe to check the fence by the barn. Should be here any minute.
- Model text output: Hector's ready to go, but the car's gonna be late. Tell Joe to check the fence by the barn. It should be here any minute.
- Generated output file: /scratch/l/lovenya/salm_migration_clean/results/baselines_kad2_smoke/audio_omni/S2A/ttas/test/audio/000001_IAK4nfn5ofI.wav
- Sample rate: 44100

