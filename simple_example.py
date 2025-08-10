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

"""Simple example for LangExtract."""

import langextract as lx
from langextract import data
import textwrap

def main():
    """Main function to run a simple extraction example."""
    print("Running LangExtract test...")
    print("-" * 50)

    # 1. Define the prompt and extraction rules
    prompt = textwrap.dedent("""\
        Extract characters and emotions in order of appearance.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context.""")

    # 2. Provide a high-quality example to guide the model
    examples = [
        data.ExampleData(
            text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
            extractions=[
                data.Extraction(
                    extraction_class="character",
                    extraction_text="ROMEO",
                    attributes={"emotional_state": "wonder"}
                ),
                data.Extraction(
                    extraction_class="emotion",
                    extraction_text="But soft!",
                    attributes={"feeling": "gentle awe"}
                ),
            ]
        )
    ]

    # The input text to be processed
    input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

    try:
        # Show what we're working with
        print("Input text:", input_text)
        print("Prompt:", prompt)
        print("Examples:", len(examples))
        print()

        print("SUCCESS! LangExtract is properly installed and imported.")
        print("Note: To run actual extractions, you would need to provide an API key for a supported model.")
        return True

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)