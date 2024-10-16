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

"""A Multimodal Autonomous Agent for Android (M3A)."""

import time
import base64
import numpy as np
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

# PROMPT_PREFIX = """
# all_action = {
#     "tap": r'`{"name": "tap", "point": "<point>(x, y)</point>"}`: Tap a specific point. Use when needing to interact with a precise location.',
#     "click": r'`{"name": "click", "element": "<box>(x_1, y_1, x_2, y_2)</box>"}`: Click on a specific element. Suitable for buttons, links, or interactive areas.',
#     "hover": r'`{"name": "hover", "element": "<box>(x_1, y_1, x_2, y_2)</box>"}`: Hover over an element. Use to reveal hidden information or trigger hover effects.',
#     "select": r'`{"name": "select", "element": "<box>(x_1, y_1, x_2, y_2)</box>"}`: Select a specific element. Typically used for checkboxes, radio buttons, or list items.',
#     "swipe": r'`{"name": "swipe", "dual_point": {"from": "<point>(x_1, y_1)</point>", "to": "<point>(x_2, y_2)</point>"}}`: Swipe from one point to another. Use for scrolling or gesture-based interactions.',
#     "select_text": r'`{"name": "select_text", "dual_point": {"from": "<point>(x, y)</point>", "to": "<point>(x, y)</point>"}}`: Select text by dragging from start to end points. Useful for text selection operations.',
#     "scroll": r'`{"name": "scroll", "pixel": {"down": "...", "right": "..."}}`: Scroll the UI. Specify relative pixel values (from 0 to 1) for vertical (down) and horizontal (right) scrolling.',
#     "input": r'`{"name": "input", "text": "..."}`: Input text into a field. Use for filling out forms or search boxes.',
#     "answer": r'`{"name": "answer", "text": "..."}`: Indicate task completion (text = "task complete") or impossibility (text = "task is impossible").',
#     "response": r'`{"name": "response", "text": "..."}`: Respond to the user with a message. Use for providing information or asking questions.',
#     "copy": r'`{"name": "copy"}`: Record the currently selected content. Use after selecting text or elements to store information.',
#     "enter": r'`{"name": "enter"}`: Press the enter key. Useful for submitting forms or confirming actions.',
#     "home": r'`{"name": "home"}`: Press the home key. Use to return to the home screen on mobile devices.',
#     "back": r'`{"name": "back"}`: Press the back key. Use to navigate to the previous screen or state.',
#     "select_value": r'`{"name": "select_value", "element": "<box>(x_1, y_1, x_2, y_2)</box>", "value": "..."}`: Select an element with a specific value. The "value" parameter should match the element\'s displayed text.'
# }
# """


# PROMPT_PREFIX = """
# all_action = {
#     "tap": r'`{"name": "tap", "point": "<point>(x, y)</point>"}`: Tap a specific point. Use when needing to interact with a precise location.',
#     "click": r'`{"name": "click", "element": "<box>(x_1, y_1, x_2, y_2)</box>"}`: Click on a specific element. Suitable for buttons, links, or interactive areas.',
#     "input": r'`{"name": "input", "text": "..."}`: Input text into a field. Use for filling out forms or search boxes.',
#     "answer": r'`{"name": "answer", "text": "..."}`: Indicate task completion (text = "task complete") or impossibility (text = "task is impossible").',
#     "enter": r'`{"name": "enter"}`: Press the enter key. Useful for submitting forms or confirming actions.',
#     "home": r'`{"name": "home"}`: Press the home key. Use to return to the home screen on mobile devices.',
#     "back": r'`{"name": "back"}`: Press the back key. Use to navigate to the previous screen or state.',
# }
# """

MAX_ROUND = 3

SYSTEM_PROMPT = """You are an AI agent capable of interacting with a user interface through function calling. Your task is to assist users in completing UI-related tasks or answering related questions.

The current UI type is: Android smartphone

When referring to elements on the UI:
- Use `<box>(x_1, y_1, x_2, y_2)</box>` to indicate a box or element on the UI.
- Use `<point>(x, y)</point>` to indicate a specific point on the UI.
- The origin (0, 0) is at the top-left corner of the screen. The x-axis increases to the right, and the y-axis increases downward. Both x and y are relative values ranging from 0 to 100.

You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the UI-related tasks. Don't make assumptions about what values to plug into functions.
<tools>
[{"type": "function", "function": {"name": "select_text", "description": "Select text on UI from specified start to end points.", "parameters": {"properties": {"start_point": {"description": "Starting point of text selection, using format: \"<point>(x, y)</point>\"", "type": "string"}, "end_point": {"description": "Ending point of text selection, using format: \"<point>(x, y)</point>\"", "type": "string"}}, "required": ["start_point", "end_point"], "type": "object"}}}, {"type": "function", "function": {"name": "enter", "description": "Press the enter key to submit forms or confirm actions.", "parameters": {"properties": {}, "type": "object"}}}, {"type": "function", "function": {"name": "input", "description": "Input the specified text into the focused field on the UI.", "parameters": {"properties": {"text": {"description": "Text to be input into the focused field", "type": "string"}}, "required": ["text"], "type": "object"}}}, {"type": "function", "function": {"name": "tap", "description": "Tap the specified point on the UI.", "parameters": {"properties": {"point": {"description": "Position of the point to tap, using format: \"<point>(x, y)</point>\"", "type": "string"}}, "required": ["point"], "type": "object"}}}, {"type": "function", "function": {"name": "home", "description": "Press home key to return to home screen.", "parameters": {"properties": {}, "type": "object"}}}, {"type": "function", "function": {"name": "back", "description": "Press back key to navigate to previous screen or state.", "parameters": {"properties": {}, "type": "object"}}}, {"type": "function", "function": {"name": "swipe", "description": "Swipe from one specified UI point to another for scrolling or gesture-based interactions.", "parameters": {"properties": {"from_point": {"description": "Starting point of the swipe, using format: \"<point>(x, y)</point>\"", "type": "string"}, "to_point": {"description": "Ending point of the swipe, using format: \"<point>(x, y)</point>\"", "type": "string"}}, "required": ["from_point", "to_point"], "type": "object"}}}, {"type": "function", "function": {"name": "set_task_status", "description": "Set the current task status as completed or impossible.", "parameters": {"properties": {"status": {"enum": ["completed", "impossible"], "type": "string"}}, "required": ["status"], "type": "object"}}}]
</tools>
For each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:
<tool_call>
{"name": <function-name>, "arguments": <args-dict>}
</tool_call>"""


def encode_image(image: np.ndarray) -> str:
    return base64.b64encode(infer.array_to_jpeg_bytes(image)).decode('utf-8')


def build_message(role="user", text=None, image=None):
    if text is not None and image is not None:
        return {
            "role": role,
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(image)}"
                    },
                    "modalities": "multi-images" if MAX_ROUND > 1 else "image"
                },
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    else:
        return {
            "role": role,
            "content": text
        }

def parse_action_output(author_output):

    return ""


class M3A(base_agent.EnvironmentInteractingAgent):
  """M3A which stands for Multimodal Autonomous Agent for Android."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      name: str = 'M3A',
      wait_after_action_seconds: float = 2.0,
  ):
    """Initializes a M3A Agent.

    Args:
      env: The environment.
      llm: The multimodal LLM wrapper.
      name: The agent name.
      wait_after_action_seconds: Seconds to wait for the screen to stablize
        after executing an action
    """
    super().__init__(env, name)
    self.llm = llm
    self.history = []
    self.messages = [build_message("system", SYSTEM_PROMPT)]
    self.additional_guidelines = None
    self.wait_after_action_seconds = wait_after_action_seconds

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False):
    super().reset(go_home_on_reset)
    # Hide the coordinates on screen which might affect the vision model.
    self.env.hide_automation_ui()
    self.history = []
    self.messages = [build_message("system", SYSTEM_PROMPT)]

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    step_data = {
        'raw_screenshot': None,
        'before_screenshot_with_som': None,
        'after_screenshot_with_som': None,
        'action_prompt': None,
        'action_output': None,
        'action_output_json': None,
        'action_reason': None,
        'action_raw_response': None,
        'summary_prompt': None,
        'summary': None,
        'summary_raw_response': None,
    }
    print('----------step ' + str(len(self.history) + 1))

    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    # import pdb; pdb.set_trace()

    step_data['raw_screenshot'] = state.pixels.copy()
    
    if len(self.messages) >= MAX_ROUND * 2 + 1:
        # MAYBE WRONG
        self.messages.pop(1)
        self.messages.pop(1)

    print(goal)

    self.messages.append(build_message("user", goal, step_data['raw_screenshot']))

    action_output, is_safe, raw_response = self.llm.predict_custom(
        self.messages
    )
    print("action_output:", action_output)

    self.messages.append(build_message("assistant", action_output))

    if not raw_response:
      raise RuntimeError('Error calling LLM in action selection phase.')
    step_data['action_output'] = action_output
    step_data['action_raw_response'] = raw_response

    action = agent_utils.extract_json_from_action_v2(action_output)
    print("action:\n", action)

    # If the output is not in the right format, add it to step summary which
    # will be passed to next step and return.
    if not action:
      print('Action prompt output is not in the correct format.')
      step_data['summary'] = (
          'Output for action selection is not in the correct format, so no'
          ' action is performed.'
      )
      self.history.append(step_data)

      # quit this round
      self.messages.pop(1)
      self.messages.pop(1)

      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    try:
      converted_action = action
      step_data['action_output_json'] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to convert the output to a valid action.')
      print(str(e))
      step_data['summary'] = (
          'Can not parse the output to a valid action. Please make sure to pick'
          ' the action from the list with required parameters (if any) in the'
          ' correct JSON format!'
      )
      self.history.append(step_data)

      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    if converted_action['name'] == 'set_task_status':
      if converted_action["arguments"].get("status", "completed") == 'impossible':
        print('Agent stopped since it thinks mission impossible.')
      step_data['summary'] = 'Agent thinks the request has been completed.'
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(
          True,
          step_data,
      )

    try:
      self.env.execute_action_v3(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print('Failed to execute action.')
      print(str(e))
      step_data['summary'] = (
          'Can not execute the action, make sure to select the action with'
          ' the required parameters (if any) in the correct JSON format!'
      )
      return base_agent.AgentInteractionResult(
          False,
          step_data,
      )

    time.sleep(self.wait_after_action_seconds)

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(
        False,
        step_data,
    )
