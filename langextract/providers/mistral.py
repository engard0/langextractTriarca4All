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

"""Mistral provider for LangExtract."""
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
    r'^mistral-',  # mistral-large, mistral-small, mistral-nemo, etc.
    priority=10,
)
@dataclasses.dataclass(init=False)
class MistralLanguageModel(inference.BaseLanguageModel):
  """Language model inference using Mistral's API with structured output."""

  model_id: str = 'mistral-large-latest'
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
      model_id: str = 'mistral-large-latest',
      api_key: str | None = None,
      base_url: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      **kwargs,
  ) -> None:
    """Initialize the Mistral language model.

    Args:
      model_id: The Mistral model ID to use (e.g., 'mistral-large-latest').
      api_key: API key for Mistral service.
      base_url: Base URL for Mistral service (if using a different endpoint).
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    # Lazy import: Mistral package required
    try:
      # pylint: disable=import-outside-toplevel
      from mistralai import Mistral
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'Mistral provider requires mistralai package. '
          'Install with: pip install langextract[mistral]'
      ) from e

    self.model_id = model_id
    self.api_key = api_key
    self.base_url = base_url
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self._extra_kwargs = kwargs or {}

    if not self.api_key:
      raise exceptions.InferenceConfigError('API key not provided for Mistral.')

    # Initialize the Mistral client
    self._client = Mistral(
        api_key=self.api_key,
        endpoint=self.base_url,
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
      response = self._client.chat.complete(
          model=self.model_id,
          messages=messages,
          temperature=config.get('temperature', self.temperature),
          max_tokens=config.get('max_output_tokens'),
          top_p=config.get('top_p'),
      )

      # Extract the response text
      output_text = response.choices[0].message.content

      return inference.ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'Mistral API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[inference.ScoredOutput]]:
    """Runs inference on a list of prompts via Mistral's API.

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