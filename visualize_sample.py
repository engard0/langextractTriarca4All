#!/usr/bin/env python3
# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of creating visualization with pre-made data."""

import langextract as lx

def main():
    """Main function to demonstrate visualization with pre-made data."""
    print("Creating visualization with pre-made data...")
    print("-" * 50)
    
    # Generate visualization from the sample data file
    try:
        html_content = lx.visualize("sample_data.jsonl")
        with open("sample_visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Generated visualization in sample_visualization.html")
        print("Open this file in your browser to view the interactive visualization!")
        return True
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)