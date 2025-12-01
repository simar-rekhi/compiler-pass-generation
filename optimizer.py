"""
Main optimization loop that ties everything together.
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from test_framework import TestFramework
from knowledge_archive import KnowledgeArchive
from llm_optimizer import LLMOptimizer, get_kernel_code
from reporter import Reporter
from triton_kernels import (
    triton_matmul,
    triton_softmax,
    get_tunable_matmul_params,
    get_tunable_softmax_params,
    get_default_matmul_params,
    get_default_softmax_params,
)


class KernelOptimizer:
    """Main optimizer that orchestrates the optimization process."""
    
    def __init__(
        self,
        kernel_name: str,
        device: str = "cuda",
        archive_dir: str = "archive",
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4",
        max_iterations: int = 20,
    ):
        self.kernel_name = kernel_name
        self.device = device
        self.max_iterations = max_iterations
        
        # Initialize components
        self.test_framework = TestFramework(device=device)
        self.archive = KnowledgeArchive(archive_dir=archive_dir)
        self.llm_optimizer = LLMOptimizer(api_key=llm_api_key, model=llm_model)
        self.reporter = Reporter(archive=self.archive, test_framework=self.test_framework)
        
        # Get kernel-specific functions and parameters
        self._setup_kernel_specifics()
    
    def _setup_kernel_specifics(self):
        """Setup kernel-specific functions and parameters."""
        if self.kernel_name == "matmul":
            self.kernel_func = triton_matmul
            self.get_tunable_params = get_tunable_matmul_params
            self.get_default_params = get_default_matmul_params
            self.test_func = self.test_framework.full_test_matmul
            self.input_size = {"m": 1024, "n": 1024, "k": 1024}
        elif self.kernel_name == "softmax":
            self.kernel_func = triton_softmax
            self.get_tunable_params = get_tunable_softmax_params
            self.get_default_params = get_default_softmax_params
            self.test_func = self.test_framework.full_test_softmax
            self.input_size = {"shape": (1024, 1024)}
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization loop.
        
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting optimization for {self.kernel_name} kernel...")
        print(f"Device: {self.device}")
        print(f"Max iterations: {self.max_iterations}")
        print("-" * 60)
        
        # Start with default parameters
        current_params = self.get_default_params()
        kernel_code = get_kernel_code(self.kernel_name)
        tunable_params = self.get_tunable_params()
        
        # Test baseline
        print("\nTesting baseline with default parameters...")
        baseline_result = self.test_func(current_params, **self.input_size)
        
        if not baseline_result["correct"]:
            print(f"Baseline failed correctness test: {baseline_result.get('error')}")
            return {"success": False, "error": "Baseline failed"}
        
        print(f"Baseline speedup: {baseline_result['speedup']:.3f}x")
        print(f"Baseline runtime: {baseline_result['median_ms']:.3f} ms")
        
        # Store baseline
        self.archive.store_kernel(
            kernel_name=self.kernel_name,
            kernel_code=kernel_code,
            parameters=current_params,
            speedup=baseline_result["speedup"],
            correctness=baseline_result["correct"],
            runtime_ms=baseline_result["median_ms"],
            baseline_ms=baseline_result["baseline_ms"],
            max_error=baseline_result["max_error"],
        )
        
        best_result = baseline_result
        best_params = current_params
        history = []
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # Get parameter suggestions from LLM
            print("Getting parameter suggestions from LLM...")
            suggested_params = self.llm_optimizer.suggest_parameters(
                kernel_name=self.kernel_name,
                kernel_code=kernel_code,
                current_params=current_params,
                tunable_params=tunable_params,
                history=history,
                performance_goal="maximize speedup",
            )
            
            print(f"Suggested parameters: {suggested_params}")
            
            # Test suggested parameters
            print("Testing suggested parameters...")
            result = self.test_func(suggested_params, **self.input_size)
            
            print(f"Correct: {result['correct']}")
            if result['correct']:
                print(f"Speedup: {result['speedup']:.3f}x")
                print(f"Runtime: {result['median_ms']:.3f} ms")
                print(f"Max error: {result['max_error']:.6f}")
            else:
                print(f"Error: {result.get('error', 'Unknown')}")
            
            # Store result
            self.archive.store_kernel(
                kernel_name=self.kernel_name,
                kernel_code=kernel_code,
                parameters=suggested_params,
                speedup=result["speedup"] if result["correct"] else 0.0,
                correctness=result["correct"],
                runtime_ms=result["median_ms"],
                baseline_ms=result.get("baseline_ms", baseline_result["baseline_ms"]),
                max_error=result.get("max_error", float('inf')),
            )
            
            # Update history
            history.append({
                "parameters": suggested_params,
                "speedup": result["speedup"] if result["correct"] else 0.0,
                "correctness": result["correct"],
                "error": result.get("error"),
            })
            
            # Update best if improved
            if result["correct"] and result["speedup"] > best_result["speedup"]:
                print(f"✓ New best speedup: {result['speedup']:.3f}x (was {best_result['speedup']:.3f}x)")
                best_result = result
                best_params = suggested_params
                current_params = suggested_params  # Use best as new starting point
            else:
                # Keep exploring from current best
                if result["correct"]:
                    current_params = suggested_params
                else:
                    # If failed, go back to best
                    current_params = best_params
            
            # Early stopping if we've achieved a good speedup
            if best_result["speedup"] > 2.0:  # 2x speedup threshold
                print(f"\nReached target speedup threshold (2.0x), stopping early.")
                break
        
        # Final results
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Best speedup: {best_result['speedup']:.3f}x")
        print(f"Best runtime: {best_result['median_ms']:.3f} ms")
        print(f"Best parameters: {best_params}")
        print(f"Max error: {best_result['max_error']:.6f}")
        
        # Get statistics
        stats = self.archive.get_statistics(self.kernel_name)
        print(f"\nTotal attempts: {stats.get('total_attempts', 0)}")
        print(f"Successful attempts: {stats.get('successful_attempts', 0)}")
        
        # Generate report
        print("\nGenerating optimization report...")
        optimization_results = {
            "success": True,
            "best_speedup": best_result["speedup"],
            "best_runtime_ms": best_result["median_ms"],
            "best_params": best_params,
            "max_error": best_result["max_error"],
            "iterations": iteration + 1,
            "statistics": stats,
        }
        
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        
        report = self.reporter.generate_report(
            self.kernel_name,
            optimization_results,
            output_file=f"reports/{self.kernel_name}_optimization_report.txt"
        )
        
        print("\n" + report)
        
        return optimization_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Triton kernels using LLM")
    parser.add_argument("--kernel", choices=["matmul", "softmax"], default="matmul",
                        help="Kernel to optimize")
    parser.add_argument("--device", default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Maximum optimization iterations")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    optimizer = KernelOptimizer(
        kernel_name=args.kernel,
        device=args.device,
        max_iterations=args.max_iterations,
        llm_api_key=args.api_key,
    )
    
    results = optimizer.optimize()
    
    if results.get("success"):
        print("\nOptimization completed successfully!")
        print(f"Final speedup: {results['best_speedup']:.3f}x")
    else:
        print("\nOptimization failed!")
        print(f"Error: {results.get('error', 'Unknown error')}")

