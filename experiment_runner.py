import json
import os
import time
from typing import List, Dict, Any
from vlm_interface import QwenVLMInterface
import pandas as pd
from datetime import datetime

class ForceExperiment:
    def __init__(self, dataset_path: str, images_dir: str):
        """Initialize experiment with dataset and image paths."""
        self.dataset_path = dataset_path
        self.images_dir = images_dir
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.objects = json.load(f)
        
        print(f"Loaded {len(self.objects)} objects from dataset")
        
        # Initialize VLM
        self.vlm = QwenVLMInterface()
        
        # Results storage
        self.results = []
    
    def run_single_prediction(self, obj: Dict[str, Any], angle: int = 1) -> Dict[str, Any]:
        """Run prediction for a single object."""
        obj_id = obj["id"]
        object_name = obj["name"]
        
        # Use first angle by default
        image_path = os.path.join(self.images_dir, f"{obj_id:03d}_angle{angle}.jpg")
        
        if not os.path.exists(image_path):
            return {
                "object_id": obj_id,
                "success": False,
                "error": f"Image not found: {image_path}"
            }
        
        print(f"[{obj_id:03d}] Predicting force for: {object_name}")
        
        # Get VLM prediction
        vlm_result = self.vlm.predict_force(image_path, object_name)
        
        # Combine with ground truth data
        result = {
            "object_id": obj_id,
            "object_name": object_name,
            "category": obj["category"],
            "ground_truth_force_N": obj["ground_truth_force_N"],
            "ground_truth_mass_kg": obj["mass_kg"],
            "ground_truth_material": obj["material"],
            "ground_truth_fragility": obj["fragility"],
            "deceptive": obj["deceptive"],
            "max_safe_force_N": obj["max_safe_force_N"],
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
            **vlm_result  # Includes success, prediction, raw_response, or error
        }
        
        return result
    
    def run_full_experiment(self, start_id: int = 1, end_id: int = 50, 
                           delay_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Run experiment on all objects in range."""
        print(f"\nStarting experiment on objects {start_id}-{end_id}")
        print("=" * 60)
        
        for obj in self.objects:
            if start_id <= obj["id"] <= end_id:
                result = self.run_single_prediction(obj)
                self.results.append(result)
                
                # Print quick summary
                if result["success"]:
                    pred_force = result["prediction"].get("required_grip_force_newtons", "N/A")
                    gt_force = result["ground_truth_force_N"]
                    print(f"  Predicted: {pred_force}N | Ground Truth: {gt_force}N")
                else:
                    print(f"  ERROR: {result.get('error', 'Unknown error')}")
                
                # Delay between requests to avoid overwhelming the model
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
        
        print("\n" + "=" * 60)
        print(f"Experiment complete! {len(self.results)} predictions made.")
        
        return self.results
    
    def save_results(self, output_path: str = "experiment_results.json"):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    def quick_summary(self) -> Dict[str, Any]:
        """Generate quick summary statistics."""
        if not self.results:
            return {"error": "No results to summarize"}
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        summary = {
            "total_objects": len(self.results),
            "successful_predictions": len(successful),
            "failed_predictions": len(failed),
            "success_rate": len(successful) / len(self.results) * 100
        }
        
        if successful:
            # Calculate basic accuracy metrics
            valid_predictions = []
            for result in successful:
                pred = result["prediction"]
                if pred and pred.get("required_grip_force_newtons") is not None:
                    predicted = pred["required_grip_force_newtons"]
                    ground_truth = result["ground_truth_force_N"]
                    error = abs(predicted - ground_truth)
                    percent_error = (error / ground_truth) * 100
                    
                    valid_predictions.append({
                        "predicted": predicted,
                        "ground_truth": ground_truth,
                        "absolute_error": error,
                        "percent_error": percent_error
                    })
            
            if valid_predictions:
                mae = sum(p["absolute_error"] for p in valid_predictions) / len(valid_predictions)
                mpe = sum(p["percent_error"] for p in valid_predictions) / len(valid_predictions)
                within_2x = sum(1 for p in valid_predictions if p["percent_error"] <= 100) / len(valid_predictions) * 100
                
                summary.update({
                    "valid_numeric_predictions": len(valid_predictions),
                    "mean_absolute_error": round(mae, 2),
                    "mean_percent_error": round(mpe, 1),
                    "within_2x_accuracy": round(within_2x, 1)
                })
        
        return summary

def main():
    """Run the full experiment."""
    # Configuration
    DATASET_PATH = "/Users/sarthak215s/VLM-Force-Prediction/grasp_force_dataset.json"
    IMAGES_DIR = "/Users/sarthak215s/VLM-Force-Prediction/images"
    
    # Initialize experiment
    experiment = ForceExperiment(DATASET_PATH, IMAGES_DIR)
    
    # Run on all objects (or subset for testing)
    # For quick testing, try: start_id=1, end_id=5
    results = experiment.run_full_experiment(start_id=1, end_id=50, delay_seconds=0.5)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiment_results_{timestamp}.json"
    experiment.save_results(output_file)
    
    # Print summary
    summary = experiment.quick_summary()
    print("\nEXPERIMENT SUMMARY:")
    print("=" * 40)
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()