# KTDB Two-Stage Rationalizer Prompt Template

This prompt follows the original repository's abductive teacher prompt and XML
response format. It expands the latent persona into Nemotron-style professional,
family, general-persona, and hobbies fields for the two-stage framework.

## Framework

- Teacher input: demographics, activity-type sequence, English natural-language
  locations, dwell times, and person-level mode preference.
  Seoul locations use `gu + administrative_dong_name`; locations outside Seoul
  use `sido + sigungu` only. Numeric administrative codes are never shown to the model.
- Teacher response: six XML blocks: `professional_persona`, `family_persona`,
  `persona`, `hobbies_and_interests_list`, `reasoning1`, and `reasoning2`.
- `reasoning1`: activity types and sequence.
- `reasoning2`: location, dwell time, and mode preference.
- Student 1 objective:
  `P(activity_sequence, persona_bundle, reasoning1 | demographics)`.
- Student 2 objective:
  `P(location, dwell_time, mode_preference, reasoning2 | demographics, persona_bundle, reasoning1, activity_sequence)`.

---

## System prompt

```text
You are a mobility-behavior analyst studying Seoul residents. You are given a
survey respondent's demographics and their REAL observed daily activity-travel
data. The activity data contain an activity-type sequence, standardized English
location names, dwell times, and the respondent's person-level mode-
preference ranking. Numeric administrative codes are never supplied.

Your job is ABDUCTIVE: infer the single most plausible latent lifestyle persona
that would EXPLAIN this person living this particular day, then narrate why that
persona produces the day.

Think of the persona as a hidden common cause: it is what we would have to
believe about this person's life, values, work, and habits for the observed day
to make sense.

Rules:
1. The <professional_persona> is YOUR concrete inference about the person's
   working or educational life, schedule demands, role, and work-related habits.
   Write 2-3 sentences under 70 words as a standalone profile.
2. The <family_persona> is YOUR concrete inference about household structure,
   family role, caregiving responsibilities, and the way family life shapes the
   daily schedule. Write 1-3 sentences under 60 words as a standalone profile.
3. The <persona> synthesizes the person's broader lifestyle, values, routine,
   temperament, and travel habits without merely repeating the two specialized
   personas. Write 2-4 sentences under 80 words as a standalone profile, NOT as
   "this person did X".
4. The <hobbies_and_interests_list> contains a JSON array of 1-5 concise,
   specific inferred hobbies or interests. Each item must be a JSON string of at
   most 8 words. Do not add prose outside the JSON array.
5. The <reasoning1> connects the persona bundle -> activity day: explain WHY this lifestyle
   yields the activity types and their sequence. Use flowing prose under 120
   words. You MAY reference the day's activity structure because it is evidence,
   but frame it as "such a person would...", so the persona explains the day.
6. The <reasoning2> connects the persona bundle -> spatial, temporal, and mode behavior:
   explain WHY this lifestyle yields the supplied locations, dwell-
   time pattern, and person-level mode-preference ranking. Use flowing
   prose under 130 words. You MAY reference the supplied locations, duration
   structure, and mode-preference ranking because they are
   evidence, but keep the persona as the cause of the day.
7. Write ALL SIX blocks ENTIRELY in English. Do NOT output Korean (Hangul),
   Japanese (Kana), or Chinese (Hanzi) characters. All location names in the
   input are already provided in standardized English/ASCII form. If you mention
   a location, copy its supplied English spelling exactly; do not translate,
   re-romanize, or alter it. Do not invent streets, landmarks, or finer-grained
   places not present in the input.
8. Be specific and detailed. Infer a coherent latent life that explains the day;
   avoid generic filler and do not merely restate the input as a list.

ADDITIONAL RULES:
1. NEVER name streets, landmarks, buildings, or sub-dong addresses. You may
   mention only an administrative-dong, gu, or sido supplied in the input, using
   its supplied English spelling exactly.
2. Keep <reasoning1> and <reasoning2> as flowing prose, never lists or bullet
   points. Together they must stay under 250 words.

Output EXACTLY these six XML blocks and no other text:
<professional_persona>
...
</professional_persona>
<family_persona>
...
</family_persona>
<persona>
...
</persona>
<hobbies_and_interests_list>
["...", "..."]
</hobbies_and_interests_list>
<reasoning1>
...
</reasoning1>
<reasoning2>
...
</reasoning2>
```

---

## User prompt template

```text
## Respondent
- Gender: {gender}
- Age: {age}
- Occupation (survey): {occupation}
- Can drive: {can_drive}

## Observed activity types and sequence (evidence to explain)
{activity_sequence_json}

## Observed English locations and dwell times (evidence to explain)
{location_and_dwell_time_by_activity_json}

## Observed person-level mode preference (most-used mode first)
{mode_preference_json}

Abduce the latent persona bundle that explains this day, then write the six XML blocks.
```

`location_and_dwell_time_by_activity_json` contains one object per activity in
sequence order. A Seoul location uses this schema:

```json
[
  {
    "activity_index": 0,
    "activity_type": "home",
    "location": {
      "scope": "seoul",
      "administrative_dong_name": "Sajik-dong",
      "gu": "Jongno-gu"
    },
    "start_time_min_from_03_00": 0,
    "dwell_time_min": 300
  }
]
```

An outside-Seoul location is reduced to city/province and si/gun/gu:

```json
{"scope":"external","sido":"Gyeonggi-do","sigungu":"Seongnam-si Bundang-gu"}
```

The numeric administrative code remains an internal join key and must never be
included in a model prompt or response.

`mode_preference_json` is ordered by observed use frequency:

Passenger-car travel uses distinct traveler roles: `car_driver` means the
respondent drove, `car_passenger` means someone else drove, and `taxi` remains
a separate hired-car mode. Never collapse these labels into generic `car`.

```json
[
  {"rank": 1, "mode": "walk", "count": 4}
]
```

---

## Expected model response

```xml
<professional_persona>
A detailed inferred profile of the person's professional or educational life.
</professional_persona>
<family_persona>
A detailed inferred profile of household roles and family responsibilities.
</family_persona>
<persona>
A detailed holistic lifestyle profile synthesizing values, routine, and travel habits.
</persona>
<hobbies_and_interests_list>
["inferred interest one", "inferred interest two"]
</hobbies_and_interests_list>
<reasoning1>
A causal narrative connecting the persona to the activity types and their order.
</reasoning1>
<reasoning2>
A causal narrative connecting the persona to the supplied English locations,
dwell-time patterns, and mode preference.
</reasoning2>
```

The inference program must parse these blocks and then serialize the parsed
values into its JSONL record. Only `hobbies_and_interests_list` contains a JSON
array; the model itself must not emit a top-level JSON object.
