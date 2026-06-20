# TTAS Audio Tagging and Primary Audio Class Pooling Report

Status: keep as a dataset-metadata decision record. It documents the PANNs tag
to coarse audio-class pooling process used by reference-pool and presentation
work. Do not rewrite it as current session context; update only if the metadata
artifacts are regenerated.

Generated: 2026-06-05

## Summary

This report documents the metadata-preparation step used to convert TTAS audio samples into coarse audio classes for later reference-pool construction. The process has two stages: first, each audio sample is tagged with PANNs/CNN14 AudioSet tags; second, retained AudioSet tags are pooled into one `primary_audio_class` using a curated class map and evidence-aware assignment rule.

The original TTAS split manifests are not modified. The tag and class manifests are enriched copies used for metadata inspection and future pooling.

Key final outputs:

- Total samples processed: `22,780`
- Final assignment rule: `priority_with_evidence_guard`
- Final unknown samples: `339` (1.49%)
- Number of coarse non-unknown pools: `19`
- Evidence-guard disagreement diagnostics retained per row: `1,095` cases

## Inputs and Splits

| Split | Rows | Input tag manifest | Output class manifest |
| --- | --- | --- | --- |
| train | 18,226 | data/manifests/audio_tags/ttas_train_panns_cnn14.jsonl | data/manifests/audio_classes/ttas_train_panns_cnn14_classes.jsonl |
| val | 2,277 | data/manifests/audio_tags/ttas_val_panns_cnn14.jsonl | data/manifests/audio_classes/ttas_val_panns_cnn14_classes.jsonl |
| test | 2,277 | data/manifests/audio_tags/ttas_test_panns_cnn14.jsonl | data/manifests/audio_classes/ttas_test_panns_cnn14_classes.jsonl |

## Stage 1: PANNs/CNN14 AudioSet Tagging

Audio tagging was performed with `panns_cnn14_audioset`, using PANNs/CNN14 AudioSet predictions as the backend. For each sample, all model predictions were sorted by score and reduced to retained tags for metadata use.

Final tagging settings:

- Tag threshold: `0.6`
- Minimum desired retained tags: `2`
- Fallback top-k: `3`
- Fallback max score gap: `0.25`
- Stored top tag scores per row: `10`
- Fallback used: `15,150` samples (66.51%)
- No valid retained tags during tagging: `227` samples (1.00%)

The fallback rate is high because many PANNs scores are below `0.6`, but the fallback is intentionally gap-limited so a weak second or third tag is not always forced into the retained set.

### Most Frequent Retained AudioSet Tags

| Rank | AudioSet tag | Samples | Share of samples |
| --- | --- | --- | --- |
| 1 | `Vehicle` | 7,262 | 31.88% |
| 2 | `Speech` | 4,203 | 18.45% |
| 3 | `Animal` | 2,528 | 11.10% |
| 4 | `Car` | 1,781 | 7.82% |
| 5 | `Engine` | 1,275 | 5.60% |
| 6 | `Domestic animals, pets` | 1,114 | 4.89% |
| 7 | `Train` | 1,061 | 4.66% |
| 8 | `Music` | 1,050 | 4.61% |
| 9 | `Rail transport` | 915 | 4.02% |
| 10 | `Bird` | 872 | 3.83% |
| 11 | `Dog` | 753 | 3.31% |
| 12 | `Railroad car, train wagon` | 753 | 3.31% |
| 13 | `Emergency vehicle` | 655 | 2.88% |
| 14 | `Water` | 605 | 2.66% |
| 15 | `Train horn` | 594 | 2.61% |
| 16 | `Snoring` | 553 | 2.43% |
| 17 | `Rain` | 517 | 2.27% |
| 18 | `Boat, Water vehicle` | 516 | 2.27% |
| 19 | `Toilet flush` | 481 | 2.11% |
| 20 | `Tick-tock` | 457 | 2.01% |
| 21 | `Idling` | 449 | 1.97% |
| 22 | `Inside, small room` | 439 | 1.93% |
| 23 | `Cat` | 427 | 1.87% |
| 24 | `Tools` | 426 | 1.87% |
| 25 | `Pigeon, dove` | 410 | 1.80% |
| 26 | `Siren` | 380 | 1.67% |
| 27 | `Thunder` | 363 | 1.59% |
| 28 | `Waves, surf` | 362 | 1.59% |
| 29 | `Wood` | 361 | 1.58% |
| 30 | `Steam` | 356 | 1.56% |

## Stage 2: Primary Audio Class Pooling

The retained PANNs tags were mapped into coarse classes using `configs/audio_tag_class_pooling.json`. This is deliberately not the same thing as `primary_audio_tag`: the tag is the tagger output, while the class is our coarser pooling label.

Example:

```json
{
  "retained_audio_tags": ["Vehicle", "Car", "Engine"],
  "primary_audio_tag": "Vehicle",
  "primary_audio_class": "road_vehicle_traffic"
}
```

### Assignment Rule

The first implementation used `priority_first_v1`: the highest-priority class with any matching retained tag won. Sanity checks showed this was too brittle because a single weak high-priority tag could override stronger evidence from another class. Examples included snoring being assigned to pets, ticking with a mouse-click tag being assigned to animals, and bells being assigned to music.

The final rerun uses `priority_with_evidence_guard`. A class passes the evidence guard if it satisfies at least one of these conditions:

- `score_max >= 0.6`
- `score_sum >= 0.6`
- `score_sum >= 0.5 * strongest_class_score_sum`

Among classes that pass, priority order still decides the winner. If no candidate class passes, the strongest class by score is used. This preserves priority for genuinely ambiguous cases while preventing very weak priority matches from dominating stronger evidence.

### Final Primary Audio Class Counts

| Rank | Primary audio class | Samples | Share of samples |
| --- | --- | --- | --- |
| 1 | `road_vehicle_traffic` | 5,868 | 25.76% |
| 2 | `human_voice_vocal` | 2,943 | 12.92% |
| 3 | `other_animals_insects_farm` | 1,611 | 7.07% |
| 4 | `indoor_household` | 1,444 | 6.34% |
| 5 | `domestic_pets` | 1,273 | 5.59% |
| 6 | `birds` | 1,169 | 5.13% |
| 7 | `rail_transport` | 1,108 | 4.86% |
| 8 | `tools_machinery_construction` | 1,089 | 4.78% |
| 9 | `rain_thunder_weather` | 813 | 3.57% |
| 10 | `emergency_siren` | 790 | 3.47% |
| 11 | `office_typing_clock` | 749 | 3.29% |
| 12 | `water_nature` | 715 | 3.14% |
| 13 | `music_party` | 667 | 2.93% |
| 14 | `weapon_explosion_fireworks` | 619 | 2.72% |
| 15 | `aircraft` | 571 | 2.51% |
| 16 | `water_vehicle` | 462 | 2.03% |
| 17 | `unknown` | 339 | 1.49% |
| 18 | `bells_alarms_phones` | 305 | 1.34% |
| 19 | `noise_silence_static` | 203 | 0.89% |
| 20 | `nature_ambient` | 42 | 0.18% |

## Class Pool Definitions

The table below lists every constructed pool and the AudioSet/PANNs tags currently mapped into it. `unknown` is not a pool; it is assigned only when no retained tag matches any pool.

| Pool | Assigned samples | Mapped tags | AudioSet/PANNs tags in pool |
| --- | --- | --- | --- |
| `emergency_siren` | 790 | 10 | `Emergency vehicle`, `Siren`, `Fire engine, fire truck (siren)`, `Police car (siren)`, `Ambulance (siren)`, `Civil defense siren`, `Fire alarm`, `Smoke detector, smoke alarm`, `Alarm`, `Alarm clock` |
| `rail_transport` | 1,108 | 8 | `Train`, `Rail transport`, `Railroad car, train wagon`, `Train horn`, `Steam whistle`, `Clickety-clack`, `Train wheels squealing`, `Subway, metro, underground` |
| `aircraft` | 571 | 6 | `Aircraft`, `Helicopter`, `Fixed-wing aircraft, airplane`, `Jet engine`, `Aircraft engine`, `Propeller, airscrew` |
| `water_vehicle` | 462 | 6 | `Boat, Water vehicle`, `Motorboat, speedboat`, `Rowboat, canoe, kayak`, `Sailboat, sailing ship`, `Ship`, `Foghorn` |
| `weapon_explosion_fireworks` | 619 | 16 | `Gunshot, gunfire`, `Machine gun`, `Explosion`, `Fusillade`, `Artillery fire`, `Fireworks`, `Firecracker`, `Cap gun`, `Boom`, `Burst, pop`, `Bang`, `Crack`, `Crackle`, `Smash, crash`, `Shatter`, `Breaking` |
| `rain_thunder_weather` | 813 | 9 | `Rain`, `Thunder`, `Thunderstorm`, `Wind`, `Wind noise (microphone)`, `Rain on surface`, `Raindrop`, `Patter`, `Drip` |
| `nature_ambient` | 42 | 2 | `Rustling leaves`, `Rustle` |
| `road_vehicle_traffic` | 5,868 | 24 | `Vehicle`, `Car`, `Engine`, `Idling`, `Accelerating, revving, vroom`, `Engine starting`, `Race car, auto racing`, `Motorcycle`, `Truck`, `Bus`, `Vehicle horn, car horn, honking`, `Honk`, `Air horn, truck horn`, `Traffic noise, roadway noise`, `Skidding`, `Motor vehicle (road)`, `Light engine (high frequency)`, `Medium engine (mid frequency)`, `Heavy engine (low frequency)`, `Engine knocking`, `Air brake`, `Reversing beeps`, `Car alarm`, `Tire squeal` |
| `domestic_pets` | 1,273 | 11 | `Domestic animals, pets`, `Dog`, `Cat`, `Meow`, `Bark`, `Bow-wow`, `Purr`, `Caterwaul`, `Whimper (dog)`, `Yip`, `Howl` |
| `birds` | 1,169 | 17 | `Bird`, `Pigeon, dove`, `Bird vocalization, bird call, bird song`, `Chirp, tweet`, `Duck`, `Goose`, `Fowl`, `Chicken, rooster`, `Crow`, `Owl`, `Coo`, `Quack`, `Crowing, cock-a-doodle-doo`, `Squawk`, `Hoot`, `Caw`, `Bird flight, flapping wings` |
| `other_animals_insects_farm` | 1,611 | 23 | `Animal`, `Horse`, `Goat`, `Frog`, `Insect`, `Fly, housefly`, `Bee, wasp, etc.`, `Sheep`, `Pig`, `Cricket`, `Snake`, `Mouse`, `Rodents, rats, mice`, `Wild animals`, `Livestock, farm animals, working animals`, `Cattle, bovinae`, `Neigh, whinny`, `Clip-clop`, `Bleat`, `Oink`, `Moo`, `Roaring cats (lions, tigers)`, `Whale vocalization` |
| `indoor_household` | 1,444 | 28 | `Inside, small room`, `Inside, large room or hall`, `Toilet flush`, `Water tap, faucet`, `Sink (filling or washing)`, `Dishes, pots, and pans`, `Cutlery, silverware`, `Frying (food)`, `Sizzle`, `Stir`, `Microwave oven`, `Blender`, `Vacuum cleaner`, `Door`, `Sliding door`, `Cupboard open or close`, `Drawer open or close`, `Bathtub (filling or washing)`, `Chewing, mastication`, `Chopping (food)`, `Boiling`, `Electric toothbrush`, `Toothbrush`, `Hair dryer`, `Air conditioning`, `Mechanical fan`, `Crumpling, crinkling`, `Tearing` |
| `tools_machinery_construction` | 1,089 | 23 | `Tools`, `Power tool`, `Drill`, `Dental drill, dentist's drill`, `Sawing`, `Chainsaw`, `Jackhammer`, `Hammer`, `Sanding`, `Filing (rasp)`, `Sewing machine`, `Printer`, `Mechanisms`, `Gears`, `Ratchet, pawl`, `Lawn mower`, `Crushing`, `Scrape`, `Scratch`, `Scissors`, `Wood`, `Rub`, `Splinter` |
| `office_typing_clock` | 749 | 11 | `Typing`, `Computer keyboard`, `Typewriter`, `Tick-tock`, `Tick`, `Clock`, `Writing`, `Mouse`, `Clicking`, `Keys jangling`, `Cash register` |
| `music_party` | 667 | 36 | `Music`, `Musical instrument`, `Singing`, `Synthetic singing`, `Male singing`, `Female singing`, `Child singing`, `Guitar`, `Piano`, `Drum`, `Drum kit`, `Rock music`, `Choir`, `Bass`, `Beatboxing`, `Percussion`, `Brass instrument`, `Wind instrument, woodwind instrument`, `Violin, fiddle`, `Trumpet`, `Trombone`, `Ukulele`, `Glockenspiel`, `Organ`, `Harmonica`, `Maraca`, `Marimba, xylophone`, `Rattle (instrument)`, `Singing bowl`, `Tuning fork`, `Cowbell`, `Tubular bells`, `Jingle bell`, `Gong`, `Bell`, `Chime` |
| `bells_alarms_phones` | 305 | 14 | `Bell`, `Church bell`, `Telephone bell ringing`, `Telephone`, `Ringtone`, `Telephone dialing, DTMF`, `Doorbell`, `Beep, bleep`, `Buzzer`, `Chime`, `Ding`, `Ding-dong`, `Dial tone`, `Busy signal` |
| `water_nature` | 715 | 14 | `Water`, `Waves, surf`, `Ocean`, `Stream`, `Waterfall`, `Trickle, dribble`, `Gush`, `Liquid`, `Pour`, `Splash, splatter`, `Slosh`, `Gurgling`, `Fill (with liquid)`, `Pump (liquid)` |
| `human_voice_vocal` | 2,943 | 30 | `Speech`, `Male speech, man speaking`, `Female speech, woman speaking`, `Child speech, kid speaking`, `Narration, monologue`, `Laughter`, `Baby laughter`, `Baby cry, infant cry`, `Crying, sobbing`, `Sneeze`, `Cough`, `Snoring`, `Burping, eructation`, `Whispering`, `Screaming`, `Cheering`, `Applause`, `Babbling`, `Breathing`, `Gasp`, `Groan`, `Grunt`, `Pant`, `Sigh`, `Sniff`, `Wheeze`, `Hiccup`, `Fart`, `Whistling`, `Whistle` |
| `noise_silence_static` | 203 | 13 | `Pink noise`, `White noise`, `Static`, `Noise`, `Silence`, `Environmental noise`, `Hiss`, `Hum`, `Mains hum`, `Sine wave`, `Sound effect`, `Field recording`, `Cacophony` |

## Unknown Handling

A sample is `unknown` when none of its retained audio tags appears in the class map. This is sample-level unknown, not tag-level unknown: a row can contain unmapped tags and still receive a known `primary_audio_class` if at least one retained tag maps to a pool.

During sanity review, several frequent unknown tags were added to existing pools, and `nature_ambient` was added for leaf/rustle sounds. After the refinement, unknown samples dropped to:

- `unknown`: `339` samples (1.49%)

### Top Retained Tags Still Seen in Unknown Samples

| Retained tag | Unknown samples containing tag |
| --- | --- |
| `Spray` | 35 |
| `Clang` | 8 |
| `Whoosh, swoosh, swish` | 8 |
| `Whip` | 8 |
| `Arrow` | 7 |
| `Thunk` | 6 |
| `Chop` | 5 |
| `Bicycle bell` | 5 |
| `Change ringing (campanology)` | 4 |
| `Radio` | 3 |
| `Zipper (clothing)` | 2 |
| `Camera` | 2 |
| `Squeal` | 2 |
| `Whack, thwack` | 2 |
| `Chirp tone` | 2 |
| `Battle cry` | 2 |
| `Crowd` | 2 |
| `Knock` | 2 |
| `Basketball bounce` | 2 |
| `Clapping` | 1 |

These remaining unknown tags are intentionally left for manual review rather than forcing them into weak pools too early.

## Sanity Check Artifacts

Manual-inspection samples were regenerated after switching to the evidence guard and adding the missing mappings.

| Inspection group | Available rows | Sampled rows | JSONL path |
| --- | --- | --- | --- |
| `human_voice_vocal` | 2,943 | 50 | data/reports/audio_classes/sanity_checks/sample_human_voice_vocal.jsonl |
| `road_vehicle_traffic` | 5,868 | 50 | data/reports/audio_classes/sanity_checks/sample_road_vehicle_traffic.jsonl |
| `unknown` | 339 | 50 | data/reports/audio_classes/sanity_checks/sample_unknown.jsonl |
| `evidence_guard_disagreements` | 1,095 | 50 | data/reports/audio_classes/sanity_checks/sample_evidence_guard_disagreements.jsonl |

The sanity summary is stored at `data/reports/audio_classes/sanity_checks/sanity_check_summary.json`.

## Generated Artifacts

Tag manifests:

- `data/manifests/audio_tags/ttas_train_panns_cnn14.jsonl`
- `data/manifests/audio_tags/ttas_val_panns_cnn14.jsonl`
- `data/manifests/audio_tags/ttas_test_panns_cnn14.jsonl`

Class manifests:

- `data/manifests/audio_classes/ttas_train_panns_cnn14_classes.jsonl`
- `data/manifests/audio_classes/ttas_val_panns_cnn14_classes.jsonl`
- `data/manifests/audio_classes/ttas_test_panns_cnn14_classes.jsonl`

Reports and configs:

- `data/reports/audio_tags/ttas_panns_cnn14_tag_frequency_summary.json`
- `data/reports/audio_classes/ttas_panns_cnn14_primary_audio_class_counts.json`
- `data/reports/audio_classes/ttas_panns_cnn14_unknown_audio_class_samples.jsonl`
- `data/reports/audio_classes/sanity_checks/`
- `configs/audio_tag_class_pooling.json`

## Notes for Presentation

- The tagger provides detailed AudioSet-level evidence; pooling turns that evidence into task-oriented coarse classes.
- The class map is intentionally auditable. Each pool has explicit tag membership and a stable priority order.
- The evidence guard was introduced after manual sanity checks found priority-only errors. This is an important quality-control step, not a cosmetic change.
- The final outputs are ready for reference-pool construction, but remaining unknown tags should be revisited if they become important for downstream evaluation coverage.
