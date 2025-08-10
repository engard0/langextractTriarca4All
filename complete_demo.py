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

"""Complete workflow example for LangExtract."""

import langextract as lx
from langextract import data
import textwrap
import json
import os

def demonstrate_data_structures():
    """Demonstrate the core data structures used by LangExtract."""
    print("=== LangExtract Data Structures ===")
    
    # 1. ExampleData - Used to provide examples to guide the model
    example = data.ExampleData(
        text="J.R.R. Tolkien was an English writer, best known for high-fantasy works.",
        extractions=[
            data.Extraction(
                extraction_class="author",
                extraction_text="J.R.R. Tolkien",
                attributes={
                    "nationality": "English",
                    "genre": "high-fantasy"
                }
            )
        ]
    )
    print("1. ExampleData:")
    print(f"   Text: {example.text}")
    print(f"   Extractions: {len(example.extractions)}")
    print(f"   First extraction class: {example.extractions[0].extraction_class}")
    print(f"   First extraction attributes: {example.extractions[0].attributes}")
    print()
    
    # 2. Document - Represents input text to be processed
    document = data.Document(
        text="Isaac Asimov was a Russian-American writer and professor of biochemistry."
    )
    print("2. Document:")
    print(f"   Text: {document.text}")
    print(f"   Document ID: {document.document_id}")
    print()
    
    # 3. Extraction - Represents an extracted entity or concept
    extraction = data.Extraction(
        extraction_class="scientist",
        extraction_text="Isaac Asimov",
        attributes={
            "field": "biochemistry",
            "nationality": "Russian-American"
        }
    )
    print("3. Extraction:")
    print(f"   Class: {extraction.extraction_class}")
    print(f"   Text: {extraction.extraction_text}")
    print(f"   Attributes: {extraction.attributes}")
    print()
    
    # 4. AnnotatedDocument - Results of processing
    annotated_doc = data.AnnotatedDocument(
        text="Marie Curie was a physicist and chemist.",
        extractions=[extraction]
    )
    print("4. AnnotatedDocument:")
    print(f"   Text: {annotated_doc.text}")
    print(f"   Extractions: {len(annotated_doc.extractions)}")
    print()

def demonstrate_prompting():
    """Demonstrate how to create prompts for LangExtract."""
    print("=== LangExtract Prompting ===")
    
    # Define a clear prompt
    prompt = textwrap.dedent("""\
        Extract scientists and their fields of expertise from the text.
        Include their nationality if mentioned.
        Use exact text for extractions and provide meaningful attributes.""")
    
    print("Prompt:")
    print(prompt)
    print()
    
    # Examples to guide the model
    examples = [
        data.ExampleData(
            text="Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
            extractions=[
                data.Extraction(
                    extraction_class="scientist",
                    extraction_text="Albert Einstein",
                    attributes={
                        "field": "theoretical physics",
                        "nationality": "German-born"
                    }
                )
            ]
        )
    ]
    
    print("Examples:")
    print(f"Number of examples: {len(examples)}")
    print(f"Example text: {examples[0].text}")
    print(f"Example extraction: {examples[0].extractions[0].extraction_text}")
    print()

def demonstrate_io():
    """Demonstrate input/output functionality."""
    print("=== LangExtract I/O ===")
    
    # Create sample annotated document
    annotated_doc = data.AnnotatedDocument(
        text="Niels Bohr was a Danish physicist who made foundational contributions to understanding atomic structure.",
        extractions=[
            data.Extraction(
                extraction_class="scientist",
                extraction_text="Niels Bohr",
                attributes={
                    "field": "physics",
                    "nationality": "Danish",
                    "contribution": "atomic structure"
                }
            )
        ]
    )
    
    # Save to JSONL (would normally require API key or local model to generate)
    try:
        lx.io.save_annotated_documents([annotated_doc], output_name="demo_results.jsonl", output_dir=".")
        print("Saved annotated document to demo_results.jsonl")
    except Exception as e:
        print(f"Note: Saving encountered an issue (likely Windows encoding): {e}")
        # Create the file manually for demonstration
        with open("demo_results.jsonl", "w", encoding="utf-8") as f:
            json.dump({
                "document_id": annotated_doc.document_id,
                "text": annotated_doc.text,
                "extractions": [
                    {
                        "extraction_class": ext.extraction_class,
                        "extraction_text": ext.extraction_text,
                        "attributes": ext.attributes
                    }
                    for ext in annotated_doc.extractions
                ]
            }, f)
        print("Created demo_results.jsonl manually for demonstration")
    
    # Show file contents
    print("\nFile contents:")
    try:
        with open("demo_results.jsonl", "r", encoding="utf-8") as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Could not read file: {e}")
    
    print()

def demonstrate_visualization():
    """Demonstrate visualization functionality."""
    print("=== LangExtract Visualization ===")
    
    # Check if we have the sample data file
    if os.path.exists("sample_data.jsonl"):
        try:
            html_content = lx.visualize("sample_data.jsonl")
            with open("demo_visualization.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print("Generated visualization in demo_visualization.html")
            print("Open this file in your browser to view the interactive visualization!")
        except Exception as e:
            print(f"Could not generate visualization: {e}")
    else:
        print("Sample data file not found. Skipping visualization.")
    
    print()

def main():
    """Main function to demonstrate LangExtract workflow."""
    print("LangExtract Complete Workflow Demo")
    print("=" * 50)
    print()
    
    # Demonstrate data structures
    demonstrate_data_structures()
    
    # Demonstrate prompting
    demonstrate_prompting()
    
    # Demonstrate I/O
    demonstrate_io()
    
    # Demonstrate visualization
    demonstrate_visualization()
    
    print("=" * 50)
    print("Demo complete!")
    print()
    print("To actually run extractions, you need either:")
    print("1. A Google Gemini API key, or")
    print("2. Ollama with a local model installed")
    print()
    print("See comprehensive_example.py for runnable examples.")

if __name__ == "__main__":
    main()