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

"""Example of creating visualization with LangExtract."""

import langextract as lx
from langextract import data
import textwrap
import json

def create_sample_data():
    """Create sample annotated document data for visualization."""
    # Create a sample annotated document
    sample_text = "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different scientific fields."
    
    # Create extractions
    extractions = [
        data.Extraction(
            extraction_class="scientist",
            extraction_text="Marie Curie",
            attributes={
                "name": "Marie Curie",
                "field": "physics and chemistry"
            }
        ),
        data.Extraction(
            extraction_class="achievement",
            extraction_text="first woman to win a Nobel Prize",
            attributes={
                "type": "Nobel Prize",
                "significance": "first woman recipient"
            }
        ),
        data.Extraction(
            extraction_class="scientific_concept",
            extraction_text="radioactivity",
            attributes={
                "field": "physics",
                "contribution": "pioneering research"
            }
        )
    ]
    
    # Create annotated document
    annotated_doc = data.AnnotatedDocument(
        text=sample_text,
        extractions=extractions
    )
    
    return annotated_doc

def main():
    """Main function to demonstrate visualization."""
    print("Creating sample visualization with LangExtract...")
    print("-" * 50)
    
    # Create sample data
    annotated_doc = create_sample_data()
    
    # Save to JSONL file
    try:
        lx.io.save_annotated_documents([annotated_doc], output_name="sample_results.jsonl", output_dir=".")
        print("Saved sample data to sample_results.jsonl")
    except Exception as e:
        print(f"Could not save data: {e}")
        return
    
    # Generate visualization
    try:
        html_content = lx.visualize("sample_results.jsonl")
        with open("sample_visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Generated visualization in sample_visualization.html")
        print("Open this file in your browser to view the interactive visualization!")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("This might be because the visualization feature requires specific data formats.")
    
    # Show what the JSONL file contains
    print("\nSample data saved to JSONL:")
    try:
        with open("sample_results.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                print(json.dumps(json.loads(line), indent=2))
                break  # Just show the first record
    except Exception as e:
        print(f"Could not read JSONL file: {e}")

if __name__ == "__main__":
    main()