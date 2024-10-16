# Copyright 2024 The android_world Authors.
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

"""Utilities for agents."""

import ast
import re
from typing import Any, Optional
import json


def extract_json(s: str) -> Optional[dict[str, Any]]:
  """Extracts JSON from string.

  Args:
    s: A string with a JSON in it. E.g., "{'hello': 'world'}" or from CoT:
      "let's think step-by-step, ..., {'hello': 'world'}".

  Returns:
    JSON object.
  """
  pattern = r'\{.*?\}'
  match = re.search(pattern, s)
  if match:
    try:
      return ast.literal_eval(match.group())
    except (SyntaxError, ValueError) as error:
      print('Cannot extract JSON, skipping due to error %s', error)
      return None
  else:
    return None


def extract_json_from_action(s: str) -> Optional[dict[str, Any]]:
  start = s.find("<|action_start|>") + len("<|action_start|>")
  end = s.find("<|action_end|>")
  try:
    return eval(s[start: end])
  except Exception as e:
    print(f"extract_json_from_action fail: {e}")
    return None


def extract_json_from_action_v2(s: str) -> Optional[dict[str, Any]]:
  start = s.find("<tool_call>\n") + len("<tool_call>\n")
  end = s.find("\n</tool_call>")
  try:
    return eval(s[start: end])
  except Exception as e:
    print(f"extract_json_from_action fail: {e}")
    return None


def extract_json_from_action_v3(s: str) -> Optional[dict[str, Any]]:
  start = s.find("<tool_call>\n") + len("<tool_call>\n")
  end = s.find("\n</tool_call>")
  try:
    return json.loads(s[start: end])
  except Exception as e:
    print(f"extract_json_from_action fail: {e}")
    return None
