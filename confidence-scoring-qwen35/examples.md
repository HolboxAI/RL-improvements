# Worked examples — overconfident misses and under-confident hits

Concrete rows from `results/confidence_full.csv`. The first group is the dangerous
case: the model is **nearly certain (≈0.999) and wrong**. The second is the inverse:
**low confidence (≈0.7–0.8) but actually correct**.

## High confidence, wrong answer (conf ≈ 0.999)

### 1. `number` — undercounted, very sure
- **Question:** How many Sponges appear in this frame? Please provide a number.
- **Gold:** `2`
- **Model:** `1`
- **Confidence:** `0.999325`

The model is 99.9% sure it saw one sponge, but there were two — the classic
counting-collapse failure.

### 2. `fo_class` — dropped one object
- **Question:** List all foreign objects that are visible in this video frame. Please provide the class names or answer with none.
- **Gold:** `Clip, Sponge`
- **Model:** `Sponge`
- **Confidence:** `0.999588`

Confidently lists a single object and silently omits the second (multi-label miss).

### 3. `fo_class` — dropped the hard class
- **Question:** List all foreign objects that are visible in this video frame. Please provide the class names or answer with none.
- **Gold:** `Clip, Gallstone`
- **Model:** `Clip`
- **Confidence:** `0.999416`

Same failure mode, but the dropped object is **Gallstone** — the known
underrepresented/error-prone class.

### 4. `fo_class` — wrong class, full confidence
- **Question:** There is one surgical foreign object visible in the frame. What surgical foreign object is visible in this video frame? Please provide a class name.
- **Gold:** `Specimen bag`
- **Model:** `Clip`
- **Confidence:** `0.999523`

### 5. `fo_class` — wrong class on a positional question
- **Question:** What class is the foreign object located in the top/right relative to the image center? Please provide a class name.
- **Gold:** `Clip`
- **Model:** `Specimen`
- **Confidence:** `0.999464`

### 6. `open_ended` — structure confused
- **Question:** Where is the first foreign object placed in this clip? Please answer with exactly two fields in the format: anatomical structure, [proximal|distal].
- **Gold:** `Cystic duct, proximal`
- **Model:** `Cystic artery, proximal`
- **Confidence:** `0.999545`

Right spatial relation, wrong anatomy (artery vs. duct) — still 99.9% confident.

---

## Low confidence, correct answer (conf ≈ 0.7–0.8)

### 1. `number` — right count, hesitant
- **Question:** How many different foreign object instances appear in this frame? Please provide a number.
- **Gold:** `3`
- **Model:** `3`
- **Confidence:** `0.766652`

### 2. `number` — right count, hesitant
- **Question:** How many Clips appear in this frame? Please provide a number.
- **Gold:** `2`
- **Model:** `2`
- **Confidence:** `0.768007`

### 3. `fo_class` — correct rare class, low confidence
- **Question:** There is one surgical foreign object visible in the frame. What surgical foreign object is visible in this video frame? Please provide a class name.
- **Gold:** `External drain`
- **Model:** `External drain`
- **Confidence:** `0.762007`

Correct on the rare `External drain` class, but the model is comparatively unsure.

### 4. `open_ended` — correct "none", low confidence
- **Question:** Which of the foreign object classes is partially occluded/not fully visible because of an anatomical structure at this frame?
- **Gold:** `none`
- **Model:** `none`
- **Confidence:** `0.740865`

Correctly detects absence, but with low certainty — negative answers are where the
model hedges.

### 5. `fo_class` — correct class, low confidence
- **Question:** There is one surgical foreign object visible in the frame. What surgical foreign object is visible in this video frame? Please provide a class name.
- **Gold:** `Specimen`
- **Model:** `Specimen`
- **Confidence:** `0.783801`

---

## What this shows

- The overconfident misses cluster in two failure modes already known for this task:
  **multi-label omissions** (list-all questions dropping one object, especially
  `Gallstone`) and **counting errors**.
- The under-confident hits cluster on **rare classes** (`External drain`) and
  **negative answers** (`none`) — the model hedges exactly where its training signal is
  weakest, even when it lands on the right answer.
- This is why raw token-logprob confidence is a poor "trust this answer" gate: the
  highest-confidence answers include some of the most confident mistakes, and the
  lowest-confidence answers are frequently right.
