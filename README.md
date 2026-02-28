# VLM Force Prediction Experiment

Testing whether Vision Language Models can predict robotic grasping forces using only semantic understanding, without explicit physics knowledge.

## Dataset

- **50 objects** across 5 categories: household, fragile, deformable, deceptive, tools_rigid
- **150 images** (3 angles per object)
- **Ground truth forces** calculated using physics: `((mass * 9.81) + 5) / (2 * friction) * 1.5`
- **Special test cases**: Deceptive objects (foam brick vs lead weight), fragile items with safety limits

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run experiment:**
```bash
python experiment_runner.py
```

3. **Analyze results:**
```bash
python evaluation.py experiment_results_TIMESTAMP.json
```

## Files

- `grasp_force_dataset.json` - Object dataset with ground truth forces
- `images/` - Object images (001_angle1.jpg format)
- `vlm_interface.py` - Qwen 2.5VL model wrapper
- `experiment_runner.py` - Main experiment execution
- `evaluation.py` - Results analysis and visualization

## Key Research Questions

1. **Semantic Correlation**: Do VLM predictions correlate with physics-based ground truth?
2. **Deception Handling**: Can it overcome visual biases (heavy-looking foam vs light-looking lead)?
3. **Safety Reasoning**: Does it predict conservative forces for fragile objects?
4. **Material Understanding**: Does it encode material-friction relationships?

## Expected Outputs

- Prediction accuracy metrics (MAE, correlation)
- Performance by object category
- Safety analysis for fragile objects
- Visualization plots (predicted vs ground truth, error by category)