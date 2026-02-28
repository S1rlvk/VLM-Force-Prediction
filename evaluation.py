import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from scipy.stats import pearsonr, spearmanr
import os

class ForceExperimentAnalysis:
    def __init__(self, results_file: str):
        """Initialize analysis with experiment results."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.df = self._prepare_dataframe()
        print(f"Loaded {len(self.results)} results for analysis")
        print(f"Valid predictions: {len(self.df)} / {len(self.results)}")
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        rows = []
        
        for result in self.results:
            if not result.get("success", False):
                continue
                
            prediction = result.get("prediction", {})
            if not prediction or prediction.get("required_grip_force_newtons") is None:
                continue
            
            row = {
                # Object info
                "object_id": result["object_id"],
                "object_name": result["object_name"],
                "category": result["category"],
                "deceptive": result["deceptive"],
                
                # Ground truth
                "gt_force": result["ground_truth_force_N"],
                "gt_mass_kg": result["ground_truth_mass_kg"],
                "gt_material": result["ground_truth_material"],
                "gt_fragility": result["ground_truth_fragility"],
                "max_safe_force": result["max_safe_force_N"],
                
                # VLM predictions
                "pred_force": prediction["required_grip_force_newtons"],
                "pred_material": prediction.get("material", ""),
                "pred_fragility": prediction.get("fragility", ""),
                "pred_mass_grams": prediction.get("estimated_mass_grams", None),
                "confidence": prediction.get("confidence", ""),
                "reasoning": prediction.get("reasoning", ""),
                
                # Error metrics
                "absolute_error": abs(prediction["required_grip_force_newtons"] - result["ground_truth_force_N"]),
                "percent_error": abs(prediction["required_grip_force_newtons"] - result["ground_truth_force_N"]) / result["ground_truth_force_N"] * 100,
                "within_2x": abs(prediction["required_grip_force_newtons"] - result["ground_truth_force_N"]) / result["ground_truth_force_N"] <= 1.0
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def overall_metrics(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        if self.df.empty:
            return {"error": "No valid predictions to analyze"}
        
        metrics = {
            "total_predictions": len(self.df),
            "mean_absolute_error": self.df["absolute_error"].mean(),
            "mean_percent_error": self.df["percent_error"].mean(),
            "median_percent_error": self.df["percent_error"].median(),
            "within_2x_percent": (self.df["within_2x"].sum() / len(self.df)) * 100,
            "rmse": np.sqrt((self.df["absolute_error"] ** 2).mean())
        }
        
        # Correlation coefficients
        if len(self.df) > 2:
            pearson_r, pearson_p = pearsonr(self.df["pred_force"], self.df["gt_force"])
            spearman_r, spearman_p = spearmanr(self.df["pred_force"], self.df["gt_force"])
            
            metrics.update({
                "pearson_correlation": pearson_r,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_r,
                "spearman_p_value": spearman_p
            })
        
        return metrics
    
    def category_analysis(self) -> pd.DataFrame:
        """Analyze performance by object category."""
        if self.df.empty:
            return pd.DataFrame()
        
        category_stats = self.df.groupby("category").agg({
            "absolute_error": ["count", "mean", "std", "median"],
            "percent_error": ["mean", "median"],
            "within_2x": "mean",
            "pred_force": "mean",
            "gt_force": "mean"
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ["_".join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.rename(columns={
            "absolute_error_count": "n_objects",
            "absolute_error_mean": "mae",
            "absolute_error_std": "mae_std",
            "absolute_error_median": "mae_median",
            "percent_error_mean": "mean_percent_error",
            "percent_error_median": "median_percent_error",
            "within_2x_mean": "within_2x_rate",
            "pred_force_mean": "avg_predicted_force",
            "gt_force_mean": "avg_ground_truth_force"
        })
        
        return category_stats.reset_index()
    
    def deceptive_objects_analysis(self) -> pd.DataFrame:
        """Special analysis for deceptive objects."""
        deceptive_df = self.df[self.df["deceptive"].notna()].copy()
        
        if deceptive_df.empty:
            return pd.DataFrame()
        
        # Group by deceptive type
        deceptive_stats = deceptive_df.groupby("deceptive").agg({
            "percent_error": ["count", "mean", "median", "std"],
            "absolute_error": "mean",
            "within_2x": "mean"
        }).round(2)
        
        deceptive_stats.columns = ["_".join(col).strip() for col in deceptive_stats.columns]
        
        return deceptive_stats.reset_index()
    
    def safety_analysis(self) -> Dict[str, Any]:
        """Analyze safety for fragile objects."""
        fragile_df = self.df[self.df["max_safe_force"].notna()].copy()
        
        if fragile_df.empty:
            return {"message": "No fragile objects with safety limits found"}
        
        # Check safety violations
        fragile_df["safety_violation"] = fragile_df["pred_force"] > fragile_df["max_safe_force"]
        fragile_df["safety_margin"] = fragile_df["max_safe_force"] - fragile_df["pred_force"]
        
        safety_stats = {
            "total_fragile_objects": len(fragile_df),
            "safety_violations": fragile_df["safety_violation"].sum(),
            "violation_rate_percent": (fragile_df["safety_violation"].sum() / len(fragile_df)) * 100,
            "avg_safety_margin": fragile_df["safety_margin"].mean(),
            "min_safety_margin": fragile_df["safety_margin"].min(),
            "objects_with_violations": fragile_df[fragile_df["safety_violation"]]["object_name"].tolist()
        }
        
        return safety_stats
    
    def worst_predictions(self, n: int = 10) -> pd.DataFrame:
        """Get the worst predictions by percent error."""
        if self.df.empty:
            return pd.DataFrame()
        
        worst = self.df.nlargest(n, "percent_error")[
            ["object_name", "category", "pred_force", "gt_force", "percent_error", "reasoning"]
        ].round(2)
        
        return worst
    
    def best_predictions(self, n: int = 10) -> pd.DataFrame:
        """Get the best predictions by percent error."""
        if self.df.empty:
            return pd.DataFrame()
        
        best = self.df.nsmallest(n, "percent_error")[
            ["object_name", "category", "pred_force", "gt_force", "percent_error", "reasoning"]
        ].round(2)
        
        return best
    
    def generate_plots(self, output_dir: str = "plots"):
        """Generate all analysis plots."""
        if self.df.empty:
            print("No valid data to plot")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plot 1: Predicted vs Ground Truth Scatter
        plt.figure(figsize=(10, 8))
        
        categories = self.df["category"].unique()
        colors = sns.color_palette("husl", len(categories))
        
        for i, category in enumerate(categories):
            cat_data = self.df[self.df["category"] == category]
            plt.scatter(cat_data["gt_force"], cat_data["pred_force"], 
                       label=category, alpha=0.7, s=60, color=colors[i])
        
        # Perfect prediction line
        min_force = min(self.df["gt_force"].min(), self.df["pred_force"].min())
        max_force = max(self.df["gt_force"].max(), self.df["pred_force"].max())
        plt.plot([min_force, max_force], [min_force, max_force], 'k--', alpha=0.5, label='Perfect Prediction')
        
        # 2x error bounds
        x_range = np.linspace(min_force, max_force, 100)
        plt.fill_between(x_range, x_range * 0.5, x_range * 2, alpha=0.1, color='red', label='2x Error Bounds')
        
        plt.xlabel('Ground Truth Force (N)')
        plt.ylabel('Predicted Force (N)')
        plt.title('VLM Force Predictions vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/predicted_vs_ground_truth.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Error by Category Boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x="category", y="percent_error")
        plt.title('Prediction Error by Object Category')
        plt.ylabel('Percent Error (%)')
        plt.xlabel('Category')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/error_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Deceptive Objects Analysis
        deceptive_df = self.df[self.df["deceptive"].notna()]
        if not deceptive_df.empty:
            plt.figure(figsize=(12, 8))
            
            # Create comparison plot
            x_pos = np.arange(len(deceptive_df))
            width = 0.35
            
            plt.bar(x_pos - width/2, deceptive_df["gt_force"], width, 
                   label='Ground Truth', alpha=0.8)
            plt.bar(x_pos + width/2, deceptive_df["pred_force"], width,
                   label='Predicted', alpha=0.8)
            
            plt.xlabel('Deceptive Objects')
            plt.ylabel('Force (N)')
            plt.title('Deceptive Objects: Predicted vs Ground Truth Forces')
            plt.xticks(x_pos, [f"{row['object_name']}\n({row['deceptive']})" 
                              for _, row in deceptive_df.iterrows()], rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/deceptive_objects_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {output_dir}/")
    
    def generate_report(self, output_file: str = "analysis_report.txt"):
        """Generate comprehensive text report."""
        with open(output_file, 'w') as f:
            f.write("VLM FORCE PREDICTION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            metrics = self.overall_metrics()
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.3f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Category analysis
            f.write("PERFORMANCE BY CATEGORY:\n")
            f.write("-" * 25 + "\n")
            cat_analysis = self.category_analysis()
            f.write(cat_analysis.to_string(index=False))
            f.write("\n\n")
            
            # Deceptive objects
            deceptive_analysis = self.deceptive_objects_analysis()
            if not deceptive_analysis.empty:
                f.write("DECEPTIVE OBJECTS ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(deceptive_analysis.to_string(index=False))
                f.write("\n\n")
            
            # Safety analysis
            f.write("SAFETY ANALYSIS (FRAGILE OBJECTS):\n")
            f.write("-" * 35 + "\n")
            safety = self.safety_analysis()
            for key, value in safety.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Worst predictions
            f.write("WORST 10 PREDICTIONS:\n")
            f.write("-" * 20 + "\n")
            worst = self.worst_predictions(10)
            f.write(worst.to_string(index=False))
            f.write("\n\n")
            
            # Best predictions
            f.write("BEST 10 PREDICTIONS:\n")
            f.write("-" * 19 + "\n")
            best = self.best_predictions(10)
            f.write(best.to_string(index=False))
            f.write("\n")
        
        print(f"Report saved to {output_file}")

def main():
    """Run analysis on experiment results."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <results_file.json>")
        print("Example: python evaluation.py experiment_results_20240228_143022.json")
        return
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    # Run analysis
    analysis = ForceExperimentAnalysis(results_file)
    
    # Generate outputs
    analysis.generate_plots()
    analysis.generate_report()
    
    # Print quick summary
    print("\nQUICK SUMMARY:")
    print("=" * 30)
    metrics = analysis.overall_metrics()
    print(f"Valid predictions: {metrics.get('total_predictions', 0)}")
    print(f"Mean Absolute Error: {metrics.get('mean_absolute_error', 0):.2f}N")
    print(f"Mean Percent Error: {metrics.get('mean_percent_error', 0):.1f}%")
    print(f"Within 2x accuracy: {metrics.get('within_2x_percent', 0):.1f}%")
    print(f"Correlation: {metrics.get('pearson_correlation', 0):.3f}")

if __name__ == "__main__":
    main()