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

"""Test to verify that new providers are properly registered."""

import langextract as lx


def test_provider_registration():
    """Test that new providers are properly registered."""
    # List all registered providers
    providers = lx.providers.registry.list_entries()
    
    # Print all registered providers
    print("Registered providers:")
    for patterns, priority in providers:
        print(f"  Patterns: {patterns}, Priority: {priority}")
    
    # Check that our new providers are registered
    provider_patterns = [pattern for patterns, _ in providers for pattern in patterns]
    
    # Check for Anthropic
    assert any("claude-" in pattern for pattern in provider_patterns), "Anthropic provider not registered"
    print("[PASS] Anthropic provider is registered")
    
    # Check for Mistral
    assert any("mistral-" in pattern for pattern in provider_patterns), "Mistral provider not registered"
    print("[PASS] Mistral provider is registered")
    
    # Check for Grok
    assert any("grok-" in pattern for pattern in provider_patterns), "Grok provider not registered"
    print("[PASS] Grok provider is registered")
    
    print("All new providers are properly registered!")


if __name__ == "__main__":
    test_provider_registration()