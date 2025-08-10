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

"""Example of using LangExtract with Ollama for local LLM inference."""

import langextract as lx
from langextract import data
import textwrap
import os

def run_ollama_extraction():
    """Run a simple extraction example using Ollama (local models)."""
    print("Running LangExtract with Ollama example...")
    print("-" * 50)
    
    # Input text for extraction
    input_text = "Isaac Asimov was a prolific science fiction writer who authored the Foundation series."
    
    # Prompt describing what to extract
    prompt = "Extract the author's full name and their primary literary genre."
    
    # Examples to guide the model (few-shot learning)
    examples = [
        data.ExampleData(
            text="J.R.R. Tolkien was an English writer, best known for high-fantasy works like The Lord of the Rings.",
            extractions=[
                data.Extraction(
                    extraction_class="author_details",
                    extraction_text="J.R.R. Tolkien was an English writer...",
                    attributes={
                        "name": "J.R.R. Tolkien",
                        "genre": "high-fantasy",
                    },
                )
            ],
        )
    ]
    
    print("Input text:", input_text)
    print("Prompt:", prompt)
    print()
    
    # Check if Ollama is running
    try:
        # This would be the actual extraction call if Ollama was available
        # For demonstration purposes, we'll just show what the call would look like
        print("To run this example with Ollama, you would use:")
        print()
        print("```python")
        print("result = lx.extract(")
        print("    text_or_documents=input_text,")
        print("    prompt_description=prompt,")
        print("    examples=examples,")
        print("    model_id='gemma2:2b',  # or any Ollama model you have installed")
        print("    model_url='http://localhost:11434',")
        print("    fence_output=False,")
        print("    use_schema_constraints=False,")
        print(")")
        print("```")
        print()
        print("Note: You need to have Ollama installed and running with the model pulled:")
        print("  1. Install Ollama from https://ollama.com/")
        print("  2. Pull a model: ollama pull gemma2:2b")
        print("  3. Start Ollama: ollama serve")
        
        return True
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        print("Make sure Ollama is running: 'ollama serve'")
        return False

def run_gemini_extraction():
    """Show how to use LangExtract with Google Gemini models."""
    print("\n\nRunning LangExtract with Google Gemini example...")
    print("-" * 50)
    
    # Input text for extraction
    input_text = "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity."
    
    # Prompt describing what to extract
    prompt = "Extract the person's full name, their scientific field, and their major contribution."
    
    # Examples to guide the model (few-shot learning)
    examples = [
        data.ExampleData(
            text="Albert Einstein was a theoretical physicist who developed the theory of relativity.",
            extractions=[
                data.Extraction(
                    extraction_class="scientist_details",
                    extraction_text="Albert Einstein was a theoretical physicist...",
                    attributes={
                        "name": "Albert Einstein",
                        "field": "theoretical physics",
                        "contribution": "theory of relativity",
                    },
                )
            ],
        )
    ]
    
    print("Input text:", input_text)
    print("Prompt:", prompt)
    print()
    
    print("To run this example with Google Gemini, you would use:")
    print()
    print("```python")
    print("result = lx.extract(")
    print("    text_or_documents=input_text,")
    print("    prompt_description=prompt,")
    print("    examples=examples,")
    print("    model_id='gemini-2.5-flash',  # or gemini-2.5-pro")
    print("    api_key='your-google-ai-studio-api-key',  # Get from https://aistudio.google.com/app/apikey")
    print(")")
    print("```")
    print()
    print("Note: You need to get an API key from Google AI Studio:")
    print("  1. Go to https://aistudio.google.com/app/apikey")
    print("  2. Create an API key")
    print("  3. Set it as an environment variable: export LANGEXTRACT_API_KEY='your-key'")
    
    return True

def main():
    """Main function to run the examples."""
    print("LangExtract Examples")
    print("=" * 50)
    
    # Run Ollama example
    run_ollama_extraction()
    
    # Run Gemini example
    run_gemini_extraction()
    
    print("\n" + "=" * 50)
    print("For more examples, check out:")
    print("- https://github.com/google/langextract/tree/main/examples")
    print("- https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md")
    print("- https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md")

if __name__ == "__main__":
    main()