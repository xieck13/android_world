import glob
import gzip
import io
import os
import pickle
import time
import json
from tqdm import tqdm
from PIL import Image
import glob
import json




def _unzip_and_read_pickle(file_path: str):
  """Reads a gzipped pickle file using 'with open', unzips, and unpickles it.

  Args:
      file_path: The path to the gzipped pickle file.

  Returns:
      The original Python object that was pickled and gzipped.
  """
  with open(file_path, 'rb') as f:
    compressed = f.read()

  with gzip.open(io.BytesIO(compressed), 'rb') as f_in:
    return pickle.load(f_in)


def process_ins(ins):
    ins_dataset = []
    goal = ins['goal']
    episode_length = ins['episode_length']
    episode_data = ins['episode_data']
    for ep_index in range(episode_length):
        # ep_index = 2
        action_output_json = episode_data["action_output_json"][ep_index]
        if action_output_json is None:
            continue
        action_type = action_output_json.action_type
        train_data = episode_data["train_data"][ep_index]
        reason = episode_data['action_reason'][ep_index]
        if reason is None:
            reason = ""
        
        summary = episode_data['summary'][ep_index]
        
        if summary is not None:
            tmp_idx = summary.find('}') + 3
            summary = summary[tmp_idx:]
        else:
            summary = ""
        
        tool_func = ""
        if action_type == "click" or action_type == "long_press":
            if train_data is None:
                continue
                # pass
            min_x, max_x, min_y, max_y = train_data[3:]
            cen_x, cen_y = int((min_x + max_x) / 2 / 1080 * 100) , int((min_y + max_y) / 2 / 2400 * 100)
            action = json.dumps({
                "name": "tap",
                "arguments": {
                    "point": {"x": cen_x, "y": cen_y}
                }})
            screenshot = train_data[1]
        elif action_type == "input_text":
            if train_data is None:
                continue
                # pass
            min_x, max_x, min_y, max_y = train_data[3:]
            cen_x, cen_y = int((min_x + max_x) / 2 / 1080 * 100) , int((min_y + max_y) / 2 / 2400 * 100)
            action = json.dumps({
                "name": "point_input",
                "arguments": {
                    "point": {"x": cen_x, "y": cen_y},
                    "text": action_output_json.text
                }})
            screenshot = train_data[1]
            
        elif action_type == "scroll":
            if train_data is not None:
                min_x, max_x, min_y, max_y = train_data[3:]
                start_x, start_y = int((x_min + x_max) / 2 / 1080 * 100), int((y_min + y_max) / 2 / 2400 * 100)
                screenshot = train_data[1]
            else:
                x_min, y_min, x_max, y_max = (0, 0, 100, 100)
                start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                screenshot = episode_data['raw_screenshot'][ep_index]
                
            direction = action_output_json.direction
            if direction == 'down':
              end_x, end_y = (x_min + x_max) // 2, y_min
            elif direction == 'up':
              end_x, end_y = (x_min + x_max) // 2, y_max
            elif direction == 'right':
              end_x, end_y = x_min, (y_min + y_max) // 2
            elif direction == 'left':
              end_x, end_y = x_max, (y_min + y_max) // 2
        
            action = json.dumps({
                "name": "swipe",
                "arguments": {
                    "from_point": {"x": start_x, "y": start_y},
                    "to_point": {"x": end_x, "y": end_y}
                }
            })
            
        elif action_type == "swipe":
            
            screen_width = screen_height = 100
            mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
            direction = action_output_json.direction
            if direction == 'down':
              start_x, start_y = mid_x, 0
              end_x, end_y = mid_x, screen_height
            elif direction == 'up':
              start_x, start_y = mid_x, screen_height
              end_x, end_y = mid_x, 0
            elif direction == 'left':
              start_x, start_y = 0, mid_y
              end_x, end_y = screen_width, mid_y
            elif direction == 'right':
              start_x, start_y = screen_width, mid_y
              end_x, end_y = 0, mid_y
        
            action = json.dumps({
                "name": "swipe",
                "arguments": {
                    "from_point": {"x": start_x, "y": start_y},
                    "to_point": {"x": end_x, "y": end_y}
                }
            })
            screenshot = episode_data['raw_screenshot'][ep_index]
        
        elif action_type == 'keyboard_enter':
            action = json.dumps({
                "name": "enter",
                "arguments": {}
            })
            screenshot = episode_data['raw_screenshot'][ep_index]
        
        elif action_type == 'navigate_home':
            action = json.dumps({
                "name": "home",
                "arguments": {}
            })
            screenshot = episode_data['raw_screenshot'][ep_index]
        
        elif action_type == 'navigate_back':
            action = json.dumps({
                "name": "back",
                "arguments": {}
            })
            screenshot = episode_data['raw_screenshot'][ep_index]
        
        elif action_type == 'open_app':
            action = json.dumps({
                "name": "open_app",
                "arguments": {
                    "app_name": action_output_json.app_name
                }
            })
            screenshot = episode_data['raw_screenshot'][ep_index]
        elif action_type == "status":
            if action_output_json.goal_status == "infeasible":
                status = "impossible"
            else:
                status = "completed"
            
            action = json.dumps({
            "name": "set_task_status",
            "arguments": {
                "status": status
                }
            })
        else:
            print(action_type)
            continue
        
        image_path = f"{time.time()}.png"
        data = {
            "goal": goal,
            "image_path": f"gpt4o/{image_path}",
            "action": f"<tool_call>\n{action}\n</tool_call>",
            "reason": reason,
            "summary": summary
        }
        ins_dataset.append(data)
        
        image = Image.fromarray(screenshot).save(f"dataset/gpt4o/{image_path}")

    return ins_dataset


path_list = glob.glob("/home/xieck13/workspace/android_workspace/runs/train_m3a_gpt4o/*/*.pkl.gz")

for path in path_list:
    ins = _unzip_and_read_pickle(path)[0]
    if ins['is_successful'] == 1.0:
        print(path)
        try:
            ins_dataset = process_ins(ins)
            all_dataset += ins_dataset
        except Exception as e:
            print(e)

json.dump(all_dataset, open(f"dataset/gpt4o_{int(time.time())}.json", mode='w'))


