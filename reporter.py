"""
Reporting and documentation system for optimization results.
"""
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from knowledge_archive import KnowledgeArchive
from test_framework import TestFramework


class Reporter:
    """Report generation and documentation system."""
    
    def __init__(self, archive: KnowledgeArchive, test_framework: TestFramework):
        self.archive = archive
        self.test_framework = test_framework
    
    def analyze_stability(
        self, kernel_name: str, params: Dict[str, Any],
        input_sizes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze kernel stability across different input sizes.
        
        Args:
            kernel_name: Name of the kernel
            params: Parameters to test
            input_sizes: List of input size configurations
        
        Returns:
            Dictionary with stability analysis results
        """
        results = []
        
        if kernel_name == "matmul":
            for size_config in input_sizes:
                m, n, k = size_config.get("m", 1024), size_config.get("n", 1024), size_config.get("k", 1024)
                result = self.test_framework.full_test_matmul(params, m=m, n=n, k=k)
                results.append({
                    "input_size": {"m": m, "n": n, "k": k},
                    "speedup": result.get("speedup", 0.0),
                    "correct": result.get("correct", False),
                    "runtime_ms": result.get("median_ms", float('inf')),
                })
        elif kernel_name == "softmax":
            for size_config in input_sizes:
                shape = tuple(size_config.get("shape", (1024, 1024)))
                result = self.test_framework.full_test_softmax(params, shape=shape)
                results.append({
                    "input_size": {"shape": shape},
                    "speedup": result.get("speedup", 0.0),
                    "correct": result.get("correct", False),
                    "runtime_ms": result.get("median_ms", float('inf')),
                })
        
        # Calculate statistics
        speedups = [r["speedup"] for r in results if r["correct"]]
        runtimes = [r["runtime_ms"] for r in results if r["correct"]]
        
        if not speedups:
            return {
                "stable": False,
                "results": results,
                "error": "All tests failed",
            }
        
        mean_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        speedup_variance = sum((s - mean_speedup) ** 2 for s in speedups) / len(speedups)
        speedup_std = speedup_variance ** 0.5
        
        mean_runtime = sum(runtimes) / len(runtimes)
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)
        
        stability_score = 1.0 - (speedup_std / mean_speedup) if mean_speedup > 0 else 0.0
        
        return {
            "stable": stability_score > 0.7 and all(r["correct"] for r in results),
            "stability_score": stability_score,
            "results": results,
            "statistics": {
                "mean_speedup": mean_speedup,
                "min_speedup": min_speedup,
                "max_speedup": max_speedup,
                "speedup_std": speedup_std,
                "mean_runtime_ms": mean_runtime,
                "min_runtime_ms": min_runtime,
                "max_runtime_ms": max_runtime,
                "num_tests": len(results),
                "num_passed": len(speedups),
            },
        }
    
    def generate_report(
        self, kernel_name: str, optimization_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive optimization report.
        
        Args:
            kernel_name: Name of the kernel
            optimization_results: Results from optimization process
            output_file: Optional file path to save report
        
        Returns:
            Report text as string
        """
        # Get best kernel from archive
        best_kernel = self.archive.get_best_kernel(kernel_name)
        stats = self.archive.get_statistics(kernel_name)
        history = self.archive.get_kernel_history(kernel_name)
        
        # Build report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"TRITON KERNEL OPTIMIZATION REPORT: {kernel_name.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Best Speedup vs Baseline: {optimization_results.get('best_speedup', 0.0):.3f}x")
        report_lines.append(f"Best Runtime: {optimization_results.get('best_runtime_ms', 0.0):.3f} ms")
        report_lines.append(f"Best Parameters: {json.dumps(optimization_results.get('best_params', {}), indent=2)}")
        report_lines.append(f"Max Error: {optimization_results.get('max_error', 0.0):.6f}")
        report_lines.append(f"Total Iterations: {optimization_results.get('iterations', 0)}")
        report_lines.append("")
        
        # Statistics
        report_lines.append("## OPTIMIZATION STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Attempts: {stats.get('total_attempts', 0)}")
        report_lines.append(f"Successful Attempts: {stats.get('successful_attempts', 0)}")
        report_lines.append(f"Success Rate: {stats.get('successful_attempts', 0) / max(stats.get('total_attempts', 1), 1) * 100:.1f}%")
        report_lines.append("")
        
        # Parameter Impact Analysis
        report_lines.append("## PARAMETER IMPACT ANALYSIS")
        report_lines.append("-" * 80)
        
        # Analyze which parameters had the biggest impact
        if history and len(history) > 1:
            param_impact = self._analyze_parameter_impact(history)
            report_lines.append("Parameter changes and their impact on speedup:")
            for param_name, impact in sorted(param_impact.items(), key=lambda x: abs(x[1]), reverse=True):
                report_lines.append(f"  {param_name}: {impact:+.3f}x average impact")
        else:
            report_lines.append("Insufficient history for parameter impact analysis.")
        report_lines.append("")
        
        # Top Performers
        report_lines.append("## TOP PERFORMING CONFIGURATIONS")
        report_lines.append("-" * 80)
        
        # Sort by speedup
        successful = [h for h in history if h.get("correctness")]
        successful.sort(key=lambda x: x.get("speedup", 0.0), reverse=True)
        
        for i, config in enumerate(successful[:5], 1):  # Top 5
            report_lines.append(f"\n{i}. Speedup: {config.get('speedup', 0.0):.3f}x")
            report_lines.append(f"   Parameters: {json.dumps(config.get('parameters', {}), indent=4)}")
        
        report_lines.append("")
        
        # Stability Analysis
        report_lines.append("## STABILITY ANALYSIS")
        report_lines.append("-" * 80)
        
        if optimization_results.get("best_params"):
            # Test across different input sizes
            if kernel_name == "matmul":
                input_sizes = [
                    {"m": 512, "n": 512, "k": 512},
                    {"m": 1024, "n": 1024, "k": 1024},
                    {"m": 2048, "n": 2048, "k": 2048},
                    {"m": 4096, "n": 4096, "k": 4096},
                ]
            else:  # softmax
                input_sizes = [
                    {"shape": (512, 512)},
                    {"shape": (1024, 1024)},
                    {"shape": (2048, 2048)},
                    {"shape": (4096, 4096)},
                ]
            
            stability = self.analyze_stability(
                kernel_name, optimization_results["best_params"], input_sizes
            )
            
            if stability.get("stable"):
                report_lines.append("✓ Kernel is stable across input sizes")
            else:
                report_lines.append("⚠ Kernel shows instability across input sizes")
            
            stats_data = stability.get("statistics", {})
            report_lines.append(f"\nStability Score: {stability.get('stability_score', 0.0):.3f}")
            report_lines.append(f"Mean Speedup: {stats_data.get('mean_speedup', 0.0):.3f}x")
            report_lines.append(f"Speedup Range: {stats_data.get('min_speedup', 0.0):.3f}x - {stats_data.get('max_speedup', 0.0):.3f}x")
            report_lines.append(f"Speedup Std Dev: {stats_data.get('speedup_std', 0.0):.3f}")
            report_lines.append(f"Tests Passed: {stats_data.get('num_passed', 0)}/{stats_data.get('num_tests', 0)}")
            
            report_lines.append("\nPer-size results:")
            for result in stability.get("results", []):
                if kernel_name == "matmul":
                    size = result["input_size"]
                    report_lines.append(f"  {size['m']}x{size['n']}x{size['k']}: "
                                      f"{result['speedup']:.3f}x "
                                      f"({'✓' if result['correct'] else '✗'})")
                else:
                    shape = result["input_size"]["shape"]
                    report_lines.append(f"  {shape[0]}x{shape[1]}: "
                                      f"{result['speedup']:.3f}x "
                                      f"({'✓' if result['correct'] else '✗'})")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text
    
    def _analyze_parameter_impact(self, history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which parameters had the biggest impact on performance."""
        param_impacts = {}
        
        if len(history) < 2:
            return param_impacts
        
        # Track parameter changes and their effects
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            if not (prev.get("correctness") and curr.get("correctness")):
                continue
            
            prev_params = prev.get("parameters", {})
            curr_params = curr.get("parameters", {})
            prev_speedup = prev.get("speedup", 0.0)
            curr_speedup = curr.get("speedup", 0.0)
            
            speedup_delta = curr_speedup - prev_speedup
            
            # Find changed parameters
            for param_name in set(list(prev_params.keys()) + list(curr_params.keys())):
                prev_val = prev_params.get(param_name)
                curr_val = curr_params.get(param_name)
                
                if prev_val != curr_val:
                    if param_name not in param_impacts:
                        param_impacts[param_name] = []
                    param_impacts[param_name].append(speedup_delta)
        
        # Average impacts
        avg_impacts = {}
        for param_name, impacts in param_impacts.items():
            if impacts:
                avg_impacts[param_name] = sum(impacts) / len(impacts)
        
        return avg_impacts

