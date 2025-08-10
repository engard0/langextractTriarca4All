#!/usr/bin/env python3
# Copyright 2025 Triarca Tech
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

"""Example of using LangExtractTriarca4All with various providers including Anthropic, Mistral, and Grok."""

import langextract as lx
from langextract import data
import textwrap
import os


def run_anthropic_extraction():
    """Run a simple extraction example using Anthropic models."""
    print("Running LangExtract with Anthropic example...")
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
    
    print("To run this example with Anthropic, you would use:")
    print()
    print("```python")
    print("result = lx.extract(")
    print("    text_or_documents=input_text,")
    print("    prompt_description=prompt,")
    print("    examples=examples,")
    print("    model_id='claude-3-5-sonnet-latest',  # or any Anthropic model")
    print("    api_key='your-anthropic-api-key',  # Get from https://console.anthropic.com/settings/keys")
    print(")")
    print("```")
    print()
    print("Note: You need to get an API key from Anthropic:")
    print("  1. Go to https://console.anthropic.com/settings/keys")
    print("  2. Create an API key")
    print("  3. Set it as an environment variable: export ANTHROPIC_API_KEY='your-key'")
    
    return True


def run_mistral_extraction():
    """Run a simple extraction example using Mistral models."""
    print("\n\nRunning LangExtract with Mistral example...")
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
    
    print("To run this example with Mistral, you would use:")
    print()
    print("```python")
    print("result = lx.extract(")
    print("    text_or_documents=input_text,")
    print("    prompt_description=prompt,")
    print("    examples=examples,")
    print("    model_id='mistral-large-latest',  # or any Mistral model")
    print("    api_key='your-mistral-api-key',  # Get from https://console.mistral.ai/api-keys/")
    print(")")
    print("```")
    print()
    print("Note: You need to get an API key from Mistral:")
    print("  1. Go to https://console.mistral.ai/api-keys/")
    print("  2. Create an API key")
    print("  3. Set it as an environment variable: export MISTRAL_API_KEY='your-key'")
    
    return True


def run_grok_extraction():
    """Run a simple extraction example using Grok (XAI) models."""
    print("\n\nRunning LangExtract with Grok (XAI) example...")
    print("-" * 50)
    
    # Input text for extraction
    input_text = "Elon Musk founded SpaceX and Tesla, focusing on electric vehicles and space exploration."
    
    # Prompt describing what to extract
    prompt = "Extract the person's name, the companies they founded, and their business focus areas."
    
    # Examples to guide the model (few-shot learning)
    examples = [
        data.ExampleData(
            text="Steve Jobs co-founded Apple Inc. and was focused on personal computers and consumer electronics.",
            extractions=[
                data.Extraction(
                    extraction_class="entrepreneur_details",
                    extraction_text="Steve Jobs co-founded Apple Inc....",
                    attributes={
                        "name": "Steve Jobs",
                        "companies": "Apple Inc.",
                        "focus_areas": "personal computers and consumer electronics",
                    },
                )
            ],
        )
    ]
    
    print("Input text:", input_text)
    print("Prompt:", prompt)
    print()
    
    print("To run this example with Grok (XAI), you would use:")
    print()
    print("```python")
    print("result = lx.extract(")
    print("    text_or_documents=input_text,")
    print("    prompt_description=prompt,")
    print("    examples=examples,")
    print("    model_id='grok-beta',  # or any Grok model")
    print("    api_key='your-xai-api-key',  # Get from https://console.x.ai/")
    print(")")
    print("```")
    print()
    print("Note: You need to get an API key from XAI:")
    print("  1. Go to https://console.x.ai/")
    print("  2. Create an API key")
    print("  3. Set it as an environment variable: export XAI_API_KEY='your-key'")
    
    return True


def main():
    """Main function to run the examples."""
    print("LangExtract Examples with New Providers")
    print("=" * 50)
    
    # Run Anthropic example
    run_anthropic_extraction()
    
    # Run Mistral example
    run_mistral_extraction()
    
    # Run Grok example
    run_grok_extraction()
    
    print("\n" + "=" * 50)
    print("For more examples, check out:")
    print("- https://github.com/google/langextract/tree/main/examples")
    print("- https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md")
    print("- https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md")


if __name__ == "__main__":
    main()