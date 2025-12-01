"""
LLM integration for generating optimization parameter suggestions.
"""
import os
import json
from typing import Dict, Any, Optional, List

# Delay OpenAI import until needed
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Don't import from triton_kernels at module level to avoid JITFunction inspection issues
# These will be imported lazily when needed


class LLMOptimizer:
    """LLM-based optimizer for generating kernel parameter suggestions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def get_device_info(self) -> str:
        """Get hardware/device information."""
        import torch
        
        info = []
        info.append(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            info.append(f"CUDA Version: {torch.version.cuda}")
            info.append(f"Device Name: {torch.cuda.get_device_name(0)}")
            info.append(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
            info.append(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        return "\n".join(info)
    
    def build_prompt(
        self,
        kernel_name: str,
        kernel_code: str,
        current_params: Dict[str, Any],
        tunable_params: Dict[str, List],
        history: List[Dict[str, Any]],
        performance_goal: str = "maximize speedup",
        constraints: Optional[str] = None,
    ) -> str:
        """
        Build a structured prompt for the LLM.
        
        Args:
            kernel_name: Name of the kernel (e.g., "matmul", "softmax")
            kernel_code: Triton kernel code
            current_params: Current parameter values
            tunable_params: Valid ranges for tunable parameters
            history: Previous optimization attempts
            performance_goal: Goal for optimization
            constraints: Additional constraints
        """
        device_info = self.get_device_info()
        
        history_str = ""
        if history:
            history_str = "\n### Previous Attempts:\n"
            for i, attempt in enumerate(history[-5:]):  # Last 5 attempts
                history_str += f"\nAttempt {i+1}:\n"
                history_str += f"  Parameters: {json.dumps(attempt.get('parameters', {}), indent=2)}\n"
                history_str += f"  Speedup: {attempt.get('speedup', 0.0):.3f}x\n"
                history_str += f"  Correct: {attempt.get('correctness', False)}\n"
                if attempt.get('error'):
                    history_str += f"  Error: {attempt.get('error')}\n"
        
        prompt = f"""You are an expert in GPU kernel optimization, specifically for Triton kernels.

### Task:
Optimize the following Triton kernel by suggesting new parameter values that will improve performance.

### Kernel Name:
{kernel_name}

### Kernel Code:
```python
{kernel_code}
```

### Hardware Information:
{device_info}

### Current Parameters:
{json.dumps(current_params, indent=2)}

### Tunable Parameters and Valid Values:
{json.dumps(tunable_params, indent=2)}

{history_str}

### Performance Goal:
{performance_goal}

### Constraints:
{constraints or "No additional constraints"}

### Instructions:
1. Analyze the current kernel implementation and parameters
2. Consider the hardware characteristics
3. Learn from previous attempts (if any)
4. Suggest new parameter values that are likely to improve performance
5. Focus on parameters that have the most impact: block sizes, number of warps, pipeline stages
6. Ensure suggested values are within the valid ranges provided

### Response Format:
Respond with ONLY a JSON object containing the new parameter values. Do not include any explanation or markdown formatting.

Example format:
{{
  "BLOCK_SIZE_M": 128,
  "BLOCK_SIZE_N": 64,
  "BLOCK_SIZE_K": 32,
  "GROUP_SIZE_M": 8,
  "num_stages": 4,
  "num_warps": 8
}}

Now suggest optimized parameters:
"""
        return prompt
    
    def suggest_parameters(
        self,
        kernel_name: str,
        kernel_code: str,
        current_params: Dict[str, Any],
        tunable_params: Dict[str, List],
        history: List[Dict[str, Any]],
        performance_goal: str = "maximize speedup",
        constraints: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get parameter suggestions from LLM.
        
        Returns:
            Dictionary of suggested parameters, or None if LLM unavailable
        """
        if not self.client:
            # Fallback to heuristic-based suggestions
            return self._heuristic_suggest(current_params, tunable_params, history)
        
        prompt = self.build_prompt(
            kernel_name, kernel_code, current_params,
            tunable_params, history, performance_goal, constraints
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert GPU kernel optimizer. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            # Parse JSON
            suggested_params = json.loads(content)
            
            # Validate parameters are in valid ranges
            validated_params = {}
            for key, value in suggested_params.items():
                if key in tunable_params:
                    valid_values = tunable_params[key]
                    # Find closest valid value
                    if isinstance(valid_values, list):
                        closest = min(valid_values, key=lambda x: abs(x - value))
                        validated_params[key] = closest
                    else:
                        validated_params[key] = value
                else:
                    validated_params[key] = value
            
            return validated_params
            
        except Exception as e:
            print(f"LLM suggestion failed: {e}, using heuristic fallback")
            return self._heuristic_suggest(current_params, tunable_params, history)
    
    def _heuristic_suggest(
        self,
        current_params: Dict[str, Any],
        tunable_params: Dict[str, List],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Heuristic-based parameter suggestion when LLM is unavailable.
        This is a simple exploration strategy.
        """
        import random
        
        # If we have history, try to improve on the best
        best_speedup = 0.0
        best_params = current_params
        
        for attempt in history:
            if attempt.get("correctness") and attempt.get("speedup", 0) > best_speedup:
                best_speedup = attempt.get("speedup", 0)
                best_params = attempt.get("parameters", current_params)
        
        # Explore around the best parameters
        suggested = best_params.copy()
        
        for param_name, valid_values in tunable_params.items():
            if param_name in suggested:
                current_idx = valid_values.index(suggested[param_name]) if suggested[param_name] in valid_values else 0
                # Try slightly different values
                new_idx = max(0, min(len(valid_values) - 1, current_idx + random.randint(-1, 1)))
                suggested[param_name] = valid_values[new_idx]
        
        return suggested


def get_kernel_code(kernel_name: str) -> str:
    """
    Get kernel source code from file.
    Reads from triton_kernels/{kernel_name}.py
    Uses basic file I/O to avoid any inspection issues.
    """
    file_path = f"triton_kernels/{kernel_name}.py"
    
    try:
        # Use basic file operations to avoid any pathlib/inspect issues
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: Kernel source file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading kernel source file {file_path}: {e}")
        return ""
