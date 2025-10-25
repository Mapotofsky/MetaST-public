import os
import logging
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union, Optional
from itertools import product

from tqdm import tqdm
import gin

"""
This module is based on the deeptime project (https://github.com/deeptime-ml/deeptime).
We have added an additional _update_config_file method to improve the flexibility of configuration file handling.
"""


EXPERIMENTS_PATH = 'experiment_storage'
SearchSpace = List[Union[str, int, float]]


class Experiment(ABC):

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.root = Path(config_path).parent
        gin.parse_config_file(self.config_path)

    @gin.configurable()
    def build(self,
              experiment_name: str,
              module: str,
              repeat: int,
              variables_dict: Dict[str, SearchSpace],
              data_path: str,
              model_type: str):
        """
        Builds an experiment, which consists of a list of instances.
        Can be used for hyperparam optimization, or training an ensemble.
        :param experiment_name: Name of experiment.
        :param module: Name of the file to run.
        :param repeat: Number of repeated instances per hyperparam setting.
        :param variables_dict: Dictionary containing hyperparams to test.
        :param data_path: Dataset path for organizing experiments.
        :param model_type: Model type for organizing experiments.
        """
        # create experiment instance(s)
        logging.info('Creating experiment instances ...')

        # Handle seed logic
        seed_list = variables_dict.pop('seed', [])
        if seed_list:
            # Use seed list instead of repeat
            variables_dict['seed'] = seed_list
        else:
            # Use original repeat logic
            variables_dict['repeat'] = list(range(repeat))

        variable_names, variables = zip(*variables_dict.items())

        for instance_values in tqdm(product(*variables)):
            instance_variables = dict(zip(variable_names, instance_values))

            # Build parameter string for path (exclude seed/repeat)
            param_parts = []
            for name, value in instance_variables.items():
                if name not in ['seed', 'repeat']:
                    param_str = ('%s=%.4g' % (name.split('.')[-1], value)
                                 if isinstance(value, float)
                                 else '%s=%s' % (name.split('.')[-1], str(value).replace(' ', '_')))
                    param_parts.append(param_str)

            # Build path: data_path/model_type/params/seed_or_repeat
            params_str = ','.join(param_parts) if param_parts else 'default'
            if 'seed' in instance_variables:
                final_part = f"seed={instance_variables['seed']}"
            else:
                final_part = f"repeat={instance_variables['repeat']}"

            instance_path = os.path.join(EXPERIMENTS_PATH, data_path, model_type, params_str, final_part)
            Path(instance_path).mkdir(parents=True, exist_ok=False)

            # write parameters
            instance_config_path = os.path.join(instance_path, 'config.gin')
            copy(self.config_path, instance_config_path)

            # Separate config parameters from regular parameters
            config_updates = {}  # {config_name: {key: value}}
            regular_params = {}

            for name, value in instance_variables.items():
                if name == 'seed':
                    regular_params[name] = value
                elif name != 'repeat':
                    if 'config.' in name:
                        # Parse nested config parameter
                        parts = name.rsplit('config.', 1)
                        if len(parts) == 2:
                            config_name = parts[0] + 'config'
                            config_key = parts[1]
                            if config_name not in config_updates:
                                config_updates[config_name] = {}
                            config_updates[config_name][config_key] = value
                        else:
                            regular_params[name] = value
                    else:
                        regular_params[name] = value

            # Update config file with dictionary modifications
            self._update_config_file(instance_config_path, config_updates, regular_params, data_path, model_type)

            # write command file
            command_file = os.path.join(instance_path, 'command')
            with open(command_file, 'w') as cmd:
                cmd.write(f'python -m {module} '
                          f'--config_path={instance_config_path} '
                          f'run >> {instance_path}/instance.log 2>&1')

    def _update_config_file(self, config_path: str, config_updates: Dict, regular_params: Dict, data_path: str, model_type: str):
        """
        Update the config file with dictionary modifications and regular parameters.
        """
        import re

        # Read the current config file
        with open(config_path, 'r') as f:
            content = f.read()

        # Update dictionary configurations
        for config_name, updates in config_updates.items():
            # Find the dictionary definition with proper bracket matching
            pattern = rf"{re.escape(config_name)}\s*=\s*"
            start_match = re.search(pattern, content)

            if start_match:
                start_pos = start_match.end()
                # Find the opening brace
                brace_start = content.find('{', start_pos)
                if brace_start != -1:
                    # Find matching closing brace
                    brace_count = 0
                    brace_end = brace_start
                    for i, char in enumerate(content[brace_start:], brace_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i
                                break

                    # Extract the full dictionary definition
                    dict_start = start_match.start()
                    dict_end = brace_end + 1
                    dict_str = content[dict_start:dict_end]

                    # Extract the dictionary content (between braces)
                    dict_content = content[brace_start + 1:brace_end]

                    # Parse existing key-value pairs
                    existing_dict = {}

                    # Use a more sophisticated parser for nested dictionaries
                    dict_content = dict_content.strip()
                    i = 0
                    while i < len(dict_content):
                        # Skip whitespace and newlines
                        while i < len(dict_content) and dict_content[i] in ' \t\n':
                            i += 1
                        if i >= len(dict_content):
                            break

                        # Skip comments
                        if dict_content[i] == '#':
                            while i < len(dict_content) and dict_content[i] != '\n':
                                i += 1
                            continue

                        # Find key
                        key_match = re.match(r"['\"]([^'\"]+)['\"]:\s*", dict_content[i:])
                        if not key_match:
                            i += 1
                            continue

                        key = key_match.group(1)
                        i += key_match.end()

                        # Skip whitespace after colon
                        while i < len(dict_content) and dict_content[i] in ' \t':
                            i += 1

                        # Parse value
                        if i < len(dict_content) and dict_content[i] == '{':
                            # Nested dictionary - find matching closing brace
                            brace_count = 0
                            value_start = i
                            while i < len(dict_content):
                                if dict_content[i] == '{':
                                    brace_count += 1
                                elif dict_content[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        break
                                i += 1

                            if i < len(dict_content):
                                value = dict_content[value_start:i + 1]
                                i += 1
                            else:
                                value = dict_content[value_start:]
                                i = len(dict_content)
                        else:
                            # Simple value - find end (comma or closing brace)
                            value_start = i
                            paren_count = 0
                            bracket_count = 0
                            brace_count = 0
                            in_quotes = False
                            quote_char = None

                            while i < len(dict_content):
                                char = dict_content[i]
                                if not in_quotes:
                                    if char in '"\'':
                                        in_quotes = True
                                        quote_char = char
                                    elif char == '(':
                                        paren_count += 1
                                    elif char == ')':
                                        paren_count -= 1
                                    elif char == '[':
                                        bracket_count += 1
                                    elif char == ']':
                                        bracket_count -= 1
                                    elif char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        if brace_count > 0:
                                            brace_count -= 1
                                        elif paren_count == 0 and bracket_count == 0:
                                            break
                                    elif char == ',' and paren_count == 0 and bracket_count == 0 and brace_count == 0:
                                        break
                                else:
                                    if char == quote_char and (i == 0 or dict_content[i - 1] != '\\'):
                                        in_quotes = False
                                        quote_char = None
                                i += 1

                            value = dict_content[value_start:i].strip()

                        # Clean up value (remove trailing comma)
                        value = value.rstrip().rstrip(',').strip()
                        existing_dict[key] = value

                        # Skip comma if present
                        while i < len(dict_content) and dict_content[i] in ' \t\n':
                            i += 1
                        if i < len(dict_content) and dict_content[i] == ',':
                            i += 1

                    # Update with new values
                    for key, value in updates.items():
                        value_str = f"'{value}'" if isinstance(value, str) else str(value)
                        existing_dict[key] = value_str

                    # Reconstruct dictionary string
                    dict_items = []
                    for key, value in existing_dict.items():
                        dict_items.append(f"    '{key}': {value}")

                    new_dict_str = f"{config_name} = {{\n" + ",\n".join(dict_items) + "\n}"

                    # Replace in content
                    content = content[:dict_start] + new_dict_str + content[dict_end:]

        # Write updated content and append additional parameters
        with open(config_path, 'w') as f:
            f.write(content)
            f.write(f'\nColdStartForecastDataset.data_path = \'{data_path}\'\n')
            f.write(f'instance.model_type = \'{model_type}\'\n')

            # Write regular parameters
            for name, value in regular_params.items():
                if name == 'seed':
                    f.write(f'instance.seed = {value}\n')
                else:
                    value_str = f"'{value}'" if isinstance(value, str) else str(value)
                    f.write(f'{name} = {value_str}\n')

    @abstractmethod
    def instance(self):
        """
        Instance logic method must be implemented with @gin.configurable()
        """
        ...

    @gin.configurable()
    def run(self, timer: Optional[int] = 0):
        """
        Run instance logic.
        """
        time.sleep(random.uniform(0, timer))
        running_flag = os.path.join(self.root, '_RUNNING')
        success_flag = os.path.join(self.root, '_SUCCESS')
        if os.path.isfile(success_flag) or os.path.isfile(running_flag):
            return
        elif not os.path.isfile(running_flag):
            Path(running_flag).touch()

        try:
            self.instance()
        except Exception as e:
            Path(running_flag).unlink()
            raise e
        except KeyboardInterrupt:
            Path(running_flag).unlink()
            raise Exception('KeyboardInterrupt')

        # mark experiment as finished.
        Path(running_flag).unlink()
        Path(success_flag).touch()

    def build_experiment(self):
        if EXPERIMENTS_PATH in str(self.root):
            raise Exception('Cannot build ensemble from ensemble member configuration.')
        self.build()
