"""
Knowledge archive for storing optimized kernel versions and metadata.
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class KnowledgeArchive:
    """Archive for storing kernel optimization results and metadata."""
    
    def __init__(self, archive_dir: str = "archive"):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(exist_ok=True)
        self.kernels_file = self.archive_dir / "kernels.json"
        self.metadata_file = self.archive_dir / "metadata.json"
        
        # Initialize storage
        self.kernels: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Load existing data
        self._load()
    
    def _load(self):
        """Load existing archive data."""
        if self.kernels_file.exists():
            try:
                with open(self.kernels_file, 'r') as f:
                    self.kernels = json.load(f)
            except Exception:
                self.kernels = {}
        
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
    
    def _save(self):
        """Save archive data to disk."""
        with open(self.kernels_file, 'w') as f:
            json.dump(self.kernels, f, indent=2)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def store_kernel(
        self,
        kernel_name: str,
        kernel_code: str,
        parameters: Dict[str, Any],
        speedup: float,
        correctness: bool,
        runtime_ms: float,
        baseline_ms: float,
        max_error: float,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Store a kernel version in the archive.
        
        Args:
            kernel_name: Name of the kernel (e.g., "matmul", "softmax")
            kernel_code: Source code of the kernel (if available)
            parameters: Dictionary of optimization parameters
            speedup: Speedup over baseline
            correctness: Whether the kernel passes correctness tests
            runtime_ms: Median runtime in milliseconds
            baseline_ms: Baseline runtime in milliseconds
            max_error: Maximum numerical error vs reference
            additional_metadata: Any additional metadata to store
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "kernel_code": kernel_code,
            "parameters": parameters,
            "speedup": speedup,
            "correctness": correctness,
            "runtime_ms": runtime_ms,
            "baseline_ms": baseline_ms,
            "max_error": max_error,
            "metadata": additional_metadata or {},
        }
        
        if kernel_name not in self.kernels:
            self.kernels[kernel_name] = []
        
        self.kernels[kernel_name].append(entry)
        
        # Update metadata
        if kernel_name not in self.metadata:
            self.metadata[kernel_name] = {
                "best_speedup": speedup if correctness else 0.0,
                "best_params": parameters if correctness else None,
                "total_attempts": 0,
                "successful_attempts": 0,
            }
        
        self.metadata[kernel_name]["total_attempts"] += 1
        if correctness:
            self.metadata[kernel_name]["successful_attempts"] += 1
            if speedup > self.metadata[kernel_name]["best_speedup"]:
                self.metadata[kernel_name]["best_speedup"] = speedup
                self.metadata[kernel_name]["best_params"] = parameters
        
        self._save()
    
    def get_best_kernel(
        self, kernel_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get the best kernel version for a given kernel name."""
        if kernel_name not in self.kernels or not self.kernels[kernel_name]:
            return None
        
        # Find best correct kernel
        best_entry = None
        best_speedup = 0.0
        
        for entry in self.kernels[kernel_name]:
            if entry["correctness"] and entry["speedup"] > best_speedup:
                best_speedup = entry["speedup"]
                best_entry = entry
        
        return best_entry
    
    def get_all_kernels(
        self, kernel_name: str
    ) -> List[Dict[str, Any]]:
        """Get all kernel versions for a given kernel name."""
        return self.kernels.get(kernel_name, [])
    
    def get_kernel_history(
        self, kernel_name: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get kernel optimization history, optionally limited."""
        kernels = self.kernels.get(kernel_name, [])
        if limit:
            return kernels[-limit:]
        return kernels
    
    def get_statistics(self, kernel_name: str) -> Dict[str, Any]:
        """Get statistics for a kernel optimization process."""
        return self.metadata.get(kernel_name, {})
    
    def export_results(self, kernel_name: str, output_file: str):
        """Export kernel results to a JSON file."""
        results = {
            "kernel_name": kernel_name,
            "statistics": self.get_statistics(kernel_name),
            "best_kernel": self.get_best_kernel(kernel_name),
            "all_kernels": self.get_all_kernels(kernel_name),
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

