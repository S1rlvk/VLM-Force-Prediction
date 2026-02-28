# Findings: Can VLMs Predict Robotic Grasping Forces?

## Research Question

Vision Language Models encode rich semantic knowledge about objects — materials, fragility, typical usage — learned from internet-scale text and image data. But do these semantic encodings translate into meaningful **physics reasoning**? Specifically, can a VLM predict the grip force a parallel-jaw robot gripper needs to safely lift an object, given only a photograph and the object's name?

## Experimental Setup

- **Model**: Qwen 2.5VL 7B Instruct (8-bit quantized), running locally via LM Studio
- **Dataset**: 50 objects across 5 categories, each with 3 reference images
- **Prompting**: Zero-shot, single image per object, asked to return structured JSON with force estimate
- **Ground Truth**: Physics-based formula accounting for mass, gravity, friction, two contact points, and a 1.5x safety factor: `F = ((m * 9.81) + 5) / (2 * mu) * 1.5`
- **Evaluation**: All 50 objects tested (38 succeeded on first pass, 12 retried after image preprocessing)

## Overall Results

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 5.37 N |
| Mean Percent Error | 41.0% |
| Median Percent Error | 41.1% |
| RMSE | 7.56 N |
| Pearson Correlation | 0.237 (p = 0.097) |
| Spearman Correlation | 0.170 (p = 0.237) |
| Within 2x of Ground Truth | 100% |

**The correlation between VLM predictions and physics-based ground truth is weak (r = 0.237) and not statistically significant (p = 0.097).** The model does not demonstrate reliable physics reasoning from semantic understanding alone.

## Performance by Category

| Category | Objects | MAE (N) | Mean % Error | Avg Predicted | Avg Ground Truth |
|----------|---------|---------|-------------|---------------|-----------------|
| tools_rigid | 5 | 3.36 | 21.8% | 11.00 N | 13.06 N |
| household | 15 | 5.80 | 36.1% | 8.67 N | 13.60 N |
| deformable | 10 | 2.81 | 38.5% | 5.50 N | 7.37 N |
| deceptive | 10 | 5.74 | 38.7% | 7.25 N | 11.92 N |
| fragile | 10 | 7.89 | 63.0% | 5.70 N | 12.67 N |

**Tools and rigid objects** were predicted most accurately (21.8% error), likely because tools have well-established weight/force associations in training data. **Fragile objects** had the worst performance (63.0% error) — the model systematically under-predicted forces for anything it perceived as delicate.

## Key Finding 1: The Model Defaults to Semantic Heuristics, Not Physics

The VLM overwhelmingly predicts from a small set of round values: **2.5, 3, 5, 7.5, 10, and 15 N**. These correspond almost exactly to the hint ranges provided in the prompt ("fragile 2-5N, normal 5-15N, heavy tools 15-40N"). Rather than computing force from estimated physical properties, the model classifies objects into buckets and returns a prototypical value.

This is textbook **semantic pattern matching** — the model maps "fragile glass" to the low end and "metal tool" to the high end, without modeling the continuous relationship between mass, friction, and required grip force.

## Key Finding 2: Fragile Objects Expose a Fundamental Conflict

For fragile objects, the VLM faces an unresolvable tension: it knows these objects are delicate (predicting 2.5-3N), but the actual physics demands significant force to overcome gravity and friction (10-18N). The model's "be gentle" heuristic directly contradicts what a gripper physically needs.

Worst fragile predictions:
- **Crystal wine glass**: predicted 3N, ground truth 16.18N (81% error)
- **Glass ornament**: predicted 2.5N, ground truth 12.99N (81% error)
- **Light bulb**: predicted 3N, ground truth 13.24N (77% error)

In a real robotic system, these under-predictions would cause the gripper to **drop every fragile object** — the opposite of the intended safety behavior.

## Key Finding 3: Safety Violations on Fragile Objects

Among the 9 objects with explicit maximum safe force limits (e.g., egg max 2N, balloon max 1N):

- **6 out of 9 (66.7%) had safety violations** — the model predicted forces exceeding the object's structural limit
- **Average safety margin: -2.56N** (negative = exceeded safe limit)
- **Worst case: balloon** — predicted 10N vs max safe 1N (9N over the limit)

The model predicted 10N for a raw chicken egg (max safe: 2N), apparently misidentifying it as a general food item rather than recognizing its extreme fragility. It called the inflated balloon "durable," predicting 10N for an object that would pop above 1N.

## Key Finding 4: Deceptive Objects Partially Fool the Model

Objects designed to have appearance-mass mismatches revealed asymmetric failure modes:

| Deception Type | Count | Mean % Error |
|---------------|-------|-------------|
| looks_heavy (actually light) | 5 | 30.6% |
| looks_light (actually heavy) | 4 | 46.0% |
| looks_solid (actually hollow) | 1 | 50.5% |

**"Looks light" deception is more effective.** The model severely under-predicted force for the lead counterweight (7.5N predicted vs 36.93N ground truth, 79.7% error) — the single worst deceptive prediction. It recognized the object as a lead weight and noted it was "relatively heavy for its size," yet still defaulted to a moderate force estimate.

Conversely, "looks heavy" objects (foam brick, styrofoam rock, toy dumbbell) had lower error because the model's moderate default predictions happened to be closer to the low ground truth forces of these lightweight objects.

## Key Finding 5: Best Predictions Are Coincidental, Not Insightful

The top predictions by accuracy:
- **Hollow aluminum pipe**: 10N predicted vs 9.71N truth (3.0% error)
- **Phillips screwdriver**: 10N predicted vs 9.71N truth (3.0% error)
- **Paper notebook**: 10N predicted vs 10.44N truth (4.2% error)
- **Raw chicken egg**: 10N predicted vs 10.48N truth (4.6% error)

These all share the same predicted value: **10N** — the model's default for "normal" objects. They are accurate not because the model computed the right force, but because 10N happens to be near the ground truth for medium-mass objects. The egg prediction is particularly telling: 10N is physically accurate but would **crush the egg** (max safe: 2N).

## Conclusions

1. **VLMs cannot reliably predict grasping forces.** A Pearson correlation of 0.237 (p = 0.097) indicates the relationship between VLM predictions and physics-based ground truth is indistinguishable from noise at conventional significance levels.

2. **Semantic encoding is not a substitute for physics modeling.** The model applies categorical heuristics ("fragile = low force," "tool = high force") rather than reasoning about continuous physical quantities. This produces quantized, bucket-based predictions that fail when the true force falls outside the expected range for a category.

3. **VLMs are unsafe for force-critical applications.** The 66.7% safety violation rate on fragile objects, combined with the tendency to either drastically over- or under-predict, makes raw VLM force estimates dangerous for real robotic grasping without a physics-based correction layer.

4. **Deceptive objects confirm visual bias.** The model cannot overcome appearance-based priors, particularly for objects that look lighter than they are.

5. **A hybrid approach is needed.** VLMs could contribute useful priors (material identification, fragility classification, rough mass estimation) that feed into a physics-based force calculator. The semantic understanding is not worthless — it is simply insufficient on its own.
