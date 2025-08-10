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

"""Anthropic provider for LangExtract."""
# pylint: disable=cyclic-import,duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Any, Iterator, Sequence

from langextract import data
from langextract import exceptions
from langextract import inference
from langextract import schema
from langextract.providers import registry


@registry.register(
    r'^claude-',  # claude-3-5-sonnet, claude-3-opus, claude-3-haiku, etc.
    priority=10,
)
@dataclasses.dataclass(init=False)
class AnthropicLanguageModel(inference.BaseLanguageModel):
  """Language model inference using Anthropic's API with structured output."""

  model_id: str = 'claude-3-5-sonnet-latest'
  api_key: str | None = None
  base_url: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'claude-3-5-sonnet-latest',
      api_key: str | None = None,
      base_url: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the Anthropic language model.

    Args:
      model_id: The Anthropic model ID to use (e.g., 'claude-3-5-sonnet-latest').
      api_key: API key for Anthropic service.
      base_url: Base URL for Anthropic service (if using a proxy).
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    # Lazy import: Anthropic package required
    try:
      # pylint: disable=import-outside-toplevel
      import anthropic
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'Anthropic provider requires anthropic package. '
          'Install with: pip install langextract[anthropic]'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.base_url = base_url
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise exceptions.InferenceConfigError('API key not provided for Anthropic.')

    # Initialize the Anthropic client
    self._client = anthropic.Anthropic(
        api_key=self.api_key,
        base_url=self.base_url,
    )

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> inference.ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      # Prepare the system message for structured output
      system_message = ''
      if self.format_type == data.FormatType.JSON:
        system_message = (
            'You are a helpful assistant that responds in JSON format.'
        )
      elif self.format_type == data.FormatType.YAML:
        system_message = (
            'You are a helpful assistant that responds in YAML format.'
        )

      # Prepare messages
      messages = []
      if system_message:
        messages.append({'role': 'system', 'content': system_message})
      messages.append({'role': 'user', 'content': prompt})

      # Make the API call
      response = self._client.messages.create(
          model=self.model_id,
          messages=messages,
          temperature=config.get('temperature', self.temperature),
          max_tokens=config.get('max_output_tokens', 4096),
          top_p=config.get('top_p'),
      )

      # Extract the response text
      output_text = response.content[0].text

      return inference.ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'Anthropic API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[inference.ScoredOutput]]:
    """Runs inference on a list of prompts via Anthropic's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[inference.ScoredOutput | None] = [None] * len(
            batch_prompts
        )
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'Parallel inference error: {str(e)}', original=e
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]