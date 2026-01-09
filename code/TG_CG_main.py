import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)
from peft import get_peft_model
import pdb
from tqdm import tqdm
from typing import List, Union, Dict
import json
import re
import codecs
import time
from pathlib import Path
from itertools import islice
import textwrap
import warnings
import inspect
import types
from typing import Any
from utils.Instruction import instruction_template
from utils.model_setting import bnb_config, lora_config
from utils.evaluation_test_case_pass_k import safe_eval, run_func_with_timeout
from utils.data_make_code_generation import (
    _get_param_order_from_code, 
    canonical_solution_load,
    safe_eval as safe_eval_dict
)


# Global counters copied from code_generation_main semantics (for logging/test-case filtering)
func_filtered_task_id = []
func_total_test_case = 0
func_filtered_test_case = 0
conc_total_test_case = 0
conc_filtered_test_case = 0

# if you want to add new model, you need to add it here
def simple_model_name(model_name):
    if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
        return "DeepSeek"
    elif model_name == "mistralai/Mistral-Nemo-Base-2407":
        return "Mistral"
    elif model_name == "Qwen/Qwen3-14B":
        return "Qwen"
    elif model_name == "microsoft/Phi-4-reasoning-plus":
        return "Phi"
    elif model_name == "microsoft/Phi-4-reasoning":
        return "Phi-reasoning"

def get_model_path(model_name, base_path="/mnt/shared/huggingface-cache/hub"):
    """
    Convert HuggingFace model name to local path.
    If model_name is already a local path, return it as is.
    Otherwise, try to find the model in the cache directory.
    HuggingFace cache structure: models--org--model-name/snapshots/<hash>/
    """
    # If it's already a local path (starts with /), return as is
    if model_name.startswith("/"):
        return model_name
    
    # Convert HuggingFace model name to local cache path format
    # e.g., "org/model-name" -> "models--org--model-name"
    local_name = model_name.replace("/", "--")
    cache_model_dir = os.path.join(base_path, f"models--{local_name}")
    
    # Check if the cache directory exists
    if os.path.exists(cache_model_dir):
        # Look for snapshots directory inside (HuggingFace cache structure)
        snapshots_dir = os.path.join(cache_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the first (or latest) snapshot
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                # Use the first snapshot (usually there's only one)
                model_path = os.path.join(snapshots_dir, snapshots[0])
                # Check if config.json exists in this path
                if os.path.exists(os.path.join(model_path, "config.json")):
                    return model_path
    
    # Return original model_name if local path doesn't exist
    # HuggingFace will use cache_dir to download/load from the specified cache directory
    return model_name

def get_cache_dir(base_path="/mnt/shared/huggingface-cache/hub"):
    """
    Return the cache directory path for HuggingFace models.
    This ensures models are downloaded to and loaded from the specified path.
    """
    return base_path
    
def load_jsonl_to_dict(jsonl_path, original_path=None, total_jsonl_path=None, use_instruction='None'):
    task_map = {}
    original_data = {}
    total_data_task_id = {}
    
    # for original data : contract, entry_point
    if original_path:
        with open(original_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                task_id = item['task_id']
                original_data[task_id] = {
                    'contract': item.get('contract'),
                    'entry_point': item.get('entry_point', 'solution')
                }

    # total task_id
    if total_jsonl_path:
        with open(total_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                task_id = item['name']
                total_data_task_id[task_id] = item
    
    # code promt data
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    i = json.loads(line)
                except:
                    if "\\zn" in line:
                        line = line.replace("\\zn", "\\n")
                    i = json.loads(line)
                
                
                if use_instruction.startswith("CODE_GENERATION") or use_instruction.startswith("NEGATIVE_EXAMPLE") or use_instruction == "BASE_PROMPT_WITH_CVT":
                    if total_data_task_id != {}:
                        if i['name'] not in total_data_task_id.keys():
                            continue
                        
                    task_id = i['name']
                    task_map[task_id] = {
                        'prompt': i['grammar'][0]['production'][0],
                        'contract': original_data[task_id]['contract'],
                        'entry_point': original_data[task_id]['entry_point'],
                    }
                else:
                    task_id = i['task_id']
                    task_map[task_id] = {
                        'prompt': i['prompt'],
                        'contract': i['contract'],
                        'canonical_solution': i['canonical_solution'],
                        'entry_point': i.get('entry_point', 'solution'),
                    }
        
    except Exception as e:
        print(e)
        pdb.set_trace()
    return task_map

def python_like_to_json(text):
    text = text.replace("True", "true").replace("False", "false").replace("None", "null")
    text = re.sub(r"\(([^()]*?,[^()]*?)\)", r"[\1]", text)
    return text

def extract_output_json(text):
    if isinstance(text, list):
        text = text[0]

    if text.strip().startswith("```json"):
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
            
    text = re.split(r"(</div>|<script|\n---|\n```|\n#)", text)[0].strip()
    text = python_like_to_json(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return None

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _build_contract_str(contracts: Union[str, List, Dict]) -> str:
    if not contracts:
        return ""

    if isinstance(contracts, list):
        flat = []
        for item in contracts:
            if isinstance(item, dict):
                flat.extend(item.values())
            else:
                flat.append(str(item))
        contracts = "\n".join(flat)

    if isinstance(contracts, dict):
        def key_order(k): 
            m = re.search(r"\d+", k)
            return int(m.group()) if m else k
        lines = [contracts[k] for k in sorted(contracts, key=key_order)]

    else:
        lines = []
        try:
            obj = json.loads(contracts)
            if isinstance(obj, dict):
                lines = [obj[k] for k in sorted(obj)]
        except json.JSONDecodeError:
            pass
        
        if not lines:
            lines = contracts.splitlines()

    cleaned = []
    for raw in lines:
        raw = str(raw).strip()
        if not raw:
            continue
        if re.match(r"^(contract list|canonical solution):?", raw, re.I):
            continue
        if raw.startswith("assert_") and ":" in raw:
            raw = raw.split(":", 1)[1].strip()
        raw = raw.replace("# $_CONTRACT_$", "").strip(" ,\"")
        if not raw.startswith("assert "):
            continue
        cleaned.append(raw)

    return "\n".join(f"assert_{i}: {c}" for i, c in enumerate(cleaned)) + ("\n" if cleaned else "")


def ensure_chat_template(tok):
    if getattr(tok, "chat_template", None):
        return

    name = tok.name_or_path.lower()
    
    if "mistral" in name or "mixtral" in name:
        tok.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}[INST] {{ m['content'] }} [/INST]\n"
            "{% elif m['role'] == 'assistant' %}{{ m['content'] }}\n"
            "{% elif m['role'] == 'system' %}{{ m['content'] }}\n"
            "{% else %}{{ m['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{% endif %}"
        )
        return

    tok.chat_template = (
        "{% for m in messages %}"
        "{{ m['role'] | capitalize }}: {{ m['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}Assistant: {% endif %}"
    )


def safe_extract_json(text):
    if isinstance(text, list):
        if not text:
            return None
        text = text[0]

    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if not m:
        return None

    try:
        json_blob = m.group(1) if isinstance(m, re.Match) else m.group(0)
        json_blob = python_like_to_json(json_blob)
        obj = json.loads(json_blob)
    except json.JSONDecodeError:
        return None

    if isinstance(obj, dict) and "code" in obj:
        code_str = obj["code"]
        code_str = codecs.decode(code_str, "unicode_escape")
        return code_str.strip("\n")

    return obj

def _inject_contracts_into_solution(sol, contracts, entry):
    if not contracts.strip():
        return sol
    pat = rf'^\s*def\s+{re.escape(entry)}\s*\([^)]*\)[^\n]*:'
    m = re.search(pat, sol, flags=re.MULTILINE)
    if not m:
        return sol
    header_end = m.end()
    indent = " " * 0
    indented = textwrap.indent(contracts.rstrip("\n"), indent)
    return sol[:header_end] + "\n" + indented + "\n" + sol[header_end:].lstrip("\n")


def load_contract_test_cases_from_json(dataset, task_id):
    """Load contract test cases from input_contract_test_cases_check_format.json file.
    
    Args:
        dataset: 'humaneval' or 'mbpp'
        task_id: Task ID (e.g., 'HumanEval/0')
        
    Returns:
        tuple: (input_contract_test_case_str, grammar_constraints)
            - input_contract_test_case_str: Combined test case instructions string
            - grammar_constraints: Dictionary of contract keys
    """
    # Determine path to input_contract_test_cases_check_format.json
    if dataset == "humaneval":
        json_path = "../../data/code_generation/humaneval/CODE_GENERATION_CT/o4-mini/input_contract_test_cases_check_format.json"
    elif dataset == "mbpp":
        json_path = "../../data/code_generation/mbpp/CODE_GENERATION_CT/o4-mini/input_contract_test_cases_check_format.json"
    else:
        print(f"[WARN] Unknown dataset: {dataset}, returning empty contract test cases")
        raise ValueError(f"Unknown dataset: {dataset}")
    
    input_contract_test_case_str = ""
    grammar_constraints = {}
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            input_contract_test_cases = json.load(f)
        
        if task_id in input_contract_test_cases:
            # Get grammar_constraints (contract keys)
            grammar_constraints = {key: "" for key in input_contract_test_cases[task_id].keys()}
            
            # Combine all test_case_instruction strings
            for contract_key, data in input_contract_test_cases[task_id].items():
                for test_case_instruction in data['test_case_instruction']:
                    input_contract_test_case_str += f"{test_case_instruction}\n"
    
    return input_contract_test_case_str, grammar_constraints


def load_contract_test_cases_from_grammar_smt_based_data(dataset, task_id, entry, example, contracts):
    if dataset == "humaneval":
        testcase_path = "../../code/evaluation_test_case_pass_k_for_grammar/humaneval/pre_filtering/grammar_assert_specification/o4-mini/o4-mini_contract_check_results.json"
        original_path = "../../data/evalplus-0.1.1/HumanEvalPlus.jsonl"
    elif dataset == "mbpp":
        testcase_path = "../../code/evaluation_test_case_pass_k_for_grammar/mbpp/pre_filtering/grammar_assert_specification/o4-mini/o4-mini_contract_check_results.json"
        original_path = "../../data/mbppplus-0.2.0/MbppPlus.jsonl"
    else:
        testcase_path = None
        original_path = None
    
    input_contract_test_case_str = ""
    grammar_constraints = {}
    if testcase_path and os.path.exists(testcase_path):
        # Load testcase_data
        with open(testcase_path, 'r') as f:
            full_testcase_data = json.load(f)
            testcase_data = full_testcase_data.get('contract_check_results', {})
        
        # Load original_data for canonical solution
        original_data = {}
        if original_path and os.path.exists(original_path):
            with open(original_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    original_data[item['task_id']] = item
        
        # Load code_generation_data to get grammar_constraints (same as data_make_code_generation.py)
        code_generation_data = {}
        if dataset == "humaneval":
            code_generation_data_path = "../../code/output_gpt/original/gpt-4o-mini/code_generation/humaneval_gpt-4o-mini_sft.jsonl"
        elif dataset == "mbpp":
            code_generation_data_path = "../../code/output_gpt/original/gpt-4o-mini/code_generation/mbpp_gpt-4o-mini_sft.jsonl"
        else:
            code_generation_data_path = None
        
        if code_generation_data_path and os.path.exists(code_generation_data_path):
            with open(code_generation_data_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        code_generation_data[item['name']] = item
                    except:
                        if "\\zn" in line:
                            line = line.replace("\\zn", "\\n")
                        item = json.loads(line)
                        code_generation_data[item['name']] = item
        
        # Generate contract test cases using the same logic as data_make_code_generation.py
        if task_id in testcase_data and task_id in original_data:
            # Get grammar constraints from code_generation_data (same as data_make_code_generation.py)
            if task_id in code_generation_data:
                try:
                    grammar_constraints = code_generation_data[task_id]['grammar'][0]['constraints'][0]
                except (KeyError, IndexError, TypeError):
                    grammar_constraints = {}
            else:
                # Fallback: try to get from example or contracts
                grammar_constraints = {}
                if 'grammar' in example:
                    try:
                        grammar_constraints = example['grammar'][0]['constraints'][0]
                    except (KeyError, IndexError, TypeError):
                        pass
                
                if not grammar_constraints and contracts:
                    if isinstance(contracts, dict):
                        grammar_constraints = contracts
                    elif isinstance(contracts, str):
                        # Parse contracts string to dict
                        for line in contracts.split('\n'):
                            line = line.strip()
                            if line.startswith('assert_') and ':' in line:
                                key = line.split(':', 1)[0].strip()
                                value = line.split(':', 1)[1].strip()
                                grammar_constraints[key] = value
            
            # Prepare for contract test cases
            input_contract_test_cases = {}
            for contract_key in grammar_constraints.keys():
                input_contract_test_cases[contract_key] = {'section':[], 'case_index':[], 'test_case':[], 'test_case_instruction':[]}
            
            # Load canonical solution
            canonical_solution = canonical_solution_load(original_data, task_id, dataset)
            
            # Get parameter order
            param_order = _get_param_order_from_code(canonical_solution, entry)
            used_case_indices = set()
            
            # Process contract test cases
            for contract_section in testcase_data.get(task_id, {}):
                if testcase_data[task_id][contract_section] != {}:
                    for task_id_2, contract_test_case_list in testcase_data[task_id][contract_section].items():
                        for contract_test_case in contract_test_case_list:
                            contract_in_key = contract_test_case['contract_in_key']
                            model_result = contract_test_case['model_result']
                            test_case = contract_test_case['input']
                            case_index = contract_test_case['case_index']
                            
                            if isinstance(model_result, str):
                                if model_result.startswith("AssertionError") and contract_in_key in input_contract_test_cases and input_contract_test_cases[contract_in_key]['section'] == []:
                                    # Skip if already selected by another contract key
                                    if f"{contract_section}:{case_index}" in used_case_indices:
                                        continue
                                    
                                    input_contract_test_cases[contract_in_key]['section'].append(contract_section)
                                    input_contract_test_cases[contract_in_key]['case_index'].append(case_index)
                                    input_contract_test_cases[contract_in_key]['test_case'].append(test_case)
                                    
                                    # Safely parse textual Python literals (e.g., "('a', 1)")
                                    test_case = safe_eval_dict(test_case)
                                    
                                    # Order the dict keys according to the function signature; append unknown keys afterwards
                                    ordered_keys = [k for k in param_order if k in test_case]
                                    args_str = ', '.join(f"{repr(test_case[k])}" for k in ordered_keys)
                                    input_contract_test_cases[contract_in_key]['test_case_instruction'].append(f'{contract_in_key}:\n>>> {entry}({args_str})\n "AssertionError: invalid input"\n')
                                    used_case_indices.add(f"{contract_section}:{case_index}")
            
            # Fallback: if a contract_key has no test case, add a None-based placeholder assertion
            has_any_case = any(len(_rec['test_case_instruction']) > 0 for _rec in input_contract_test_cases.values())
            if has_any_case:
                for _ck, _rec in input_contract_test_cases.items():
                    if _rec['test_case_instruction'] == []:
                        _args_str = ', '.join(['None' for _ in (param_order or [])]) if param_order is not None else ''
                        _rec['test_case_instruction'].append(f'{_ck}: no suitable test case\n')
            
            # Build contract test case string
            for contract_key in input_contract_test_cases.keys():
                for test_case_instruction in input_contract_test_cases[contract_key]['test_case_instruction']:
                    input_contract_test_case_str += f"{test_case_instruction}\n"
    return input_contract_test_case_str, grammar_constraints

def test_case_template_dataset(task_id, dataset, INSTRUCTION, entry, desc, sol, tests, OUTPUT_INSTRUCTION, tokenizer, use_instruction=None, contracts=None):
    system, user = "", ""
    contract_str = _build_contract_str(contracts) if contracts else ""
    system = f"{INSTRUCTION}\n\n"
    
    if dataset == "humaneval":
        sol_with_contracts = sol
    elif dataset == "mbpp":
        sol_with_contracts = _inject_contracts_into_solution(sol, contracts, entry)
    
    if dataset == "humaneval":
        i_user = (
            f"Method Name: {entry}\n"
            f"Problem Description:\n{desc}\n"
            f"{contracts}"
            f"{sol}\n\n"
            f"Contract List:\n{contract_str}\n"
        )
    elif dataset == "mbpp":
        i_user = (
                f"Method Name: {entry}\n"
                f"Problem Description:\n{desc}\n"
                f"{sol_with_contracts}\n\n"
                f"Contract List:\n{contract_str}\n"
        )        
        
    user = OUTPUT_INSTRUCTION + i_user
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)                      

def code_generation_template_dataset(task_id, dataset, example, INSTRUCTION, entry, desc, sol, tests, OUTPUT_INSTRUCTION, tokenizer, use_instruction=None, contracts=None, model_base_name=None, code_gen_mode='SFT'):
    global func_total_test_case, func_filtered_test_case, conc_total_test_case, conc_filtered_test_case
    
    messages = [{"role": "user", "content": ""}]
    
    system = ""
    user = ""
    
    data_path_base=f"../../code/evaluation_test_case_pass_k/{dataset}/pre_filtering"
    #combined_model_name = f"{code_gen_mode}-{simple_model_name(model_base_name[0])}"
    data_default_instruction = ["CODE_GENERATION_CS", "CODE_GENERATION_CT", 
                                "CODE_GENERATION_CS_CoT", "CODE_GENERATION_CT_CoT",
                                "BASE_PROMPT_WITH_CVT",
                                "NEGATIVE_EXAMPLE_WITH_BASE", "NEGATIVE_EXAMPLE_WITH_CS"]
    
    code_generation_instruction_append = ["BASE_PROMPT_WITH_CVT", 
                                          "NEGATIVE_EXAMPLE_WITH_BASE", "NEGATIVE_EXAMPLE_WITH_CS"]
    
    if "CODE_GENERATION" in use_instruction or use_instruction in code_generation_instruction_append:
        if use_instruction == "BASE_CODE_GENERATION":
            system = (
                f"{INSTRUCTION}\n"
                f"{OUTPUT_INSTRUCTION}\n"
            )
            
            user = (
                f"Method Name: {entry}\n"
                f"Problem Description:\n{desc}\n"
            )
        # data default
        elif use_instruction in data_default_instruction:
            system = example['system']
            
            if use_instruction.endswith("_CoT"):
                user = example['user'] + "Let's think step by step.\n"
            else:
                user = example['user']
        
        elif "Multi_turn" in use_instruction or "Two_turn" in use_instruction :
            
            # model name 
            if isinstance(model_base_name, list):
                raw_model_name = model_base_name[0] if model_base_name else None
            else:
                raw_model_name = model_base_name
                
            if raw_model_name is None:
                raise ValueError("model_base_name is required for Multi_turn mode")
            if '/' in raw_model_name:
                model_name_for_path = raw_model_name.split('/')[-1]
            else:
                model_name_for_path = raw_model_name

            model_name_for_parsing = model_name_for_path
            step1_dataset = load_generated_step_all(
                f"../../code/output_full/total/inference/{model_name_for_path}/base_code_generation/generated_step_all.json", 
                model_name_for_parsing
            )
            
            system = (
                f"{INSTRUCTION}\n"
                f"{OUTPUT_INSTRUCTION}"
            )
           
            data_type_name = 'humaneval' if task_id.lower().startswith("humaneval") else 'mbpp'
            # Load contract test cases from pre-generated JSON file instead of generating on-the-fly
            input_contract_test_case_str, grammar_constraints = load_contract_test_cases_from_json(data_type_name, task_id)
           
            if "CT_Multi_turn" in use_instruction:
                # Load testcase_data
                user = (
                    f"Method Name: {entry}\n"
                    f"Problem Description:\n{desc}\n"
                    f"Code: \n{step1_dataset[task_id]['parsed_code']}\n\n"
                    f"Contract List: {list(grammar_constraints.keys())}\n\n"
                    f"Contract Test Cases: \n{input_contract_test_case_str}"
                )
            elif "CS_Multi_turn" in use_instruction:
                user = (
                    f"Method Name: {entry}\n"
                    f"Problem Description:\n{desc}\n"
                    f"Code: \n{step1_dataset[task_id]['parsed_code']}"
                )
            
            elif "CT_Two_turn" in use_instruction:
                user = (
                    f"Method Name: {entry}\n"
                    f"Problem Description:\n{desc}\n"
                    f"Contract List: {list(grammar_constraints.keys())}\n\n"
                    f"Contract Test Cases: \n{input_contract_test_case_str}"
                )
            
            elif "CS_Two_turn" in use_instruction:
                user = (
                    f"Method Name: {entry}\n"
                    f"Problem Description:\n{desc}\n"
                    f"Code: \n{step1_dataset[task_id]['parsed_code']}"
                )
            else:
                raise ValueError(f"Invalid use_instruction: {use_instruction}")

    
    if user == 'None':
        encoded = 'None'
    else:
        ## Use system and user
        #messages[0]["content"] = system
        #messages[1]["content"] = user
        
        ## Use only user
        messages[0]["content"] = system + "\n" + user + "\n"
        
        ## if you want chat template
        encoded = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        ## not chat template
        # data not in cot
        if use_instruction not in ["CODE_GENERATION_CS_CoT", "CODE_GENERATION_CT_CoT", "CODE_GENERATION_CS_Multi_turn_no_limit", "CODE_GENERATION_CT_Multi_turn_no_limit"]:
            encoded = encoded + "I have done a short CoT reasoning and Aha! I have found the right answer.\n</think>\n\n```"
        
        # data in cot
        #encoded = encoded + example['reasoning'][0]
    return encoded

def build_prompt(dataset, task_id, example, tokenizer, use_instruction, model_base_name, code_gen_mode):
    entry = example.get('entry_point', '')
    desc  = example.get('prompt', '')
    sol   = example.get('canonical_solution', '')
    tests = example.get('test', [])
    contracts = example.get('contract', '')
    instruction, output_instruction = instruction_template(use_instruction)
    ensure_chat_template(tokenizer)     
    
    test_case_options = ["GRAMMAR_ASSERT_SPECIFICATION", "GRAMMAR_ASSERT_SPECIFICATION_humaneval", "GRAMMAR_ASSERT_SPECIFICATION_mbpp"]
    if use_instruction in test_case_options:
        return test_case_template_dataset(task_id, dataset, instruction, entry, 
                                          desc, sol, tests, output_instruction, 
                                          tokenizer, use_instruction, contracts)
    else:
        return code_generation_template_dataset(task_id, dataset, example, instruction, 
                                                entry, desc, sol, tests, 
                                                output_instruction, tokenizer, use_instruction, contracts, 
                                                model_base_name, code_gen_mode)
    
def count_output_tokens_only(tokenizer: PreTrainedTokenizerBase, input_prompt: str, outputs: List[str]) -> int:
    total = 0
    for out in outputs:
        out_ids = tokenizer(out, add_special_tokens=False)["input_ids"]
        total += len(out_ids)
    return total

def init_dict(model_key, args_settings):
    return {'Setting': args_settings, model_key: []}


def load_generated_step_all(path, model_name=None):
    from utils.evaluation_code_generation import EvaluationRewarder
    
    # Create a dummy rewarder instance to use parsing method
    # We only need the parsing method, so we can pass dummy paths
    dummy_humaneval_func = '../../data/evalplus-0.1.1/HumanEvalPlus.jsonl'
    dummy_mbpp_func = '../../data/mbppplus-0.2.0/MbppPlus.jsonl'
    # Use correct JSON files (not JSONL) for contracts
    dummy_humaneval_contract = "../../code/evaluation_test_case_pass_k/humaneval/pre_filtering/multi_assert_specification/o4-mini/o4-mini_all_results.json"
    dummy_mbpp_contract = "../../code/evaluation_test_case_pass_k/mbpp/pre_filtering/multi_assert_specification/o4-mini/o4-mini_all_results.json"
    
    try:
        rewarder = EvaluationRewarder(
            humaneval_functionality_dataset_path=dummy_humaneval_func,
            mbpp_functionality_dataset_path=dummy_mbpp_func,
            humaneval_contracts_dataset_path=dummy_humaneval_contract,
            mbpp_contracts_dataset_path=dummy_mbpp_contract,
            contract_test_case_type="direct"
        )
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize EvaluationRewarder: {e}")
        print("⚠️ Using simplified parsing method instead")
        rewarder = None
    
    dataset = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Find model name if not provided
    if model_name is None:
        for key in data.keys():
            if key != 'Setting':
                model_name = key
                break
    
    if model_name is None or model_name not in data:
        raise ValueError(f"Model name '{model_name}' not found in JSON file")
    
    # Get entry_point mapping from original data for better parsing
    entry_point_map = {}
    try:
        if 'humaneval' in path.lower():
            jsonl_path = '../../data/evalplus-0.1.1/HumanEvalPlus.jsonl'
        elif 'mbpp' in path.lower():
            jsonl_path = '../../data/mbppplus-0.2.0/MbppPlus.jsonl'
        else:
            jsonl_path = None
        
        if jsonl_path and os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    entry_point_map[item['task_id']] = item.get('entry_point')
    except Exception as e:
        print(f"⚠️ Warning: Could not load entry_point mapping: {e}")
    
    # Process each item in the model's data
    for item in data[model_name]:
        task_id = item.get('task_id')
        if not task_id:
            continue
        
        outputs = item.get('outputs', [])
        if not outputs:
            continue
        
        # Extract code from outputs[0] using parsing method
        model_inference = outputs[0]
        entry_point = entry_point_map.get(task_id)
        
        if rewarder:
            parsed_code = rewarder.parsing_code_generation_func(
                code_generation_model_path=path,
                model_inference=model_inference,
                entry_point=entry_point
            )
        else:
            # Fallback: simple extraction
            parsed_code = _simple_code_extraction(model_inference)
        
        dataset[task_id] = {
            'parsed_code': parsed_code,
            'outputs': outputs,
            'contracts': item.get('contracts', ''),
            'contracts_list': item.get('contracts_list', {}),
            'input': item.get('input', ''),
            'full_outputs': item.get('full_outputs', []),
        }
    
    return dataset

def _simple_code_extraction(model_inference):
    import re
    
    # Try to extract from JSON code block
    json_pattern = r'```json\s*\{[\s\S]*?"code"\s*:\s*[\'"]{3}([\s\S]*?)[\'"]{3}[\s\S]*?\}[\s\S]*?```'
    match = re.search(json_pattern, model_inference, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to extract from code block
    code_pattern = r'```(?:python|json)?\s*([\s\S]*?)```'
    matches = re.findall(code_pattern, model_inference, re.DOTALL)
    if matches:
        # Get the last code block (usually the answer)
        code_block = matches[-1].strip()
        # Try to extract from JSON if present
        if '"code"' in code_block:
            json_match = re.search(r'"code"\s*:\s*[\'"]{3}([\s\S]*?)[\'"]{3}', code_block, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
        return code_block
    
    # Fallback: return as is
    return model_inference.strip()

def load_dataset(args):
    if args.dataset == "total":
        train_dataset = {}

        data_instruction = "CODE_GENERATION_CS" if "CODE_GENERATION_CS" in args.use_instruction else "CODE_GENERATION_CT"
        
        if args.use_instruction == "BASE_CODE_GENERATION":
            humaneval_path = '../../data/evalplus-0.1.1/HumanEvalPlus.jsonl'
            mbpp_path = '../../data/mbppplus-0.2.0/MbppPlus.jsonl'
            
            for path in [humaneval_path, mbpp_path]:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        task_id = item['task_id']
                        train_dataset[task_id] = {
                            'prompt': item['prompt'],
                            'entry_point': item['entry_point'],
                            'contract': item['contract']
                        }
        
        elif "Multi_turn" in args.use_instruction or "Two_turn" in args.use_instruction:
            jsonl_path1 = '../../code/output_gpt/original/gpt-4o-mini/code_generation/humaneval_gpt-4o-mini_sft.jsonl'
            original_path1 = '../../data/evalplus-0.1.1/HumanEvalPlus.jsonl'
            jsonl_path2 = '../../code/output_gpt/original/gpt-4o-mini/code_generation/mbpp_gpt-4o-mini_sft.jsonl'
            original_path2 = '../../data/mbppplus-0.2.0/MbppPlus.jsonl'
            total_jsonl_path = f'../../data/code_generation/total/{data_instruction}/o4-mini/total.jsonl'
            
            train_dataset1 = load_jsonl_to_dict(jsonl_path1, original_path1, total_jsonl_path, args.use_instruction)
            train_dataset2 = load_jsonl_to_dict(jsonl_path2, original_path2, total_jsonl_path, args.use_instruction)
            train_dataset = {**train_dataset1, **train_dataset2}
            
            
        # load data from total jsonl path
        else:
            if args.use_instruction in ["CODE_GENERATION_CS_CoT", "CODE_GENERATION_CT_CoT"]:
                data_instruction = "CODE_GENERATION_CS" if "CODE_GENERATION_CS" in args.use_instruction else "CODE_GENERATION_CT"
            else:
                data_instruction = args.use_instruction
            
            with open(f'../../data/code_generation/total/{data_instruction}/o4-mini/total.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    task_id = item['name']
                    train_dataset[task_id] = {
                        'system': item['description'][0]['content'],
                        'user': item['description'][1]['content'],
                        'contract': item['grammar'][0]['constraints'],
                        'reasoning': item['reasoning'],
                    }
                
    elif args.dataset == 'humaneval':
        jsonl_path = '../../data/evalplus-0.1.1/HumanEvalPlus.jsonl'
        train_dataset = load_jsonl_to_dict(jsonl_path)
    
    elif args.dataset == 'mbpp':
        jsonl_path = '../../data/mbppplus-0.2.0/MbppPlus.jsonl'
        train_dataset = load_jsonl_to_dict(jsonl_path)
    
    return train_dataset

def _read_existing_single(save_path, model_key):
    '''
    This function reads the existing data from the save_path and returns a dictionary of task_ids that have been completed.
    '''
    if not os.path.exists(save_path):
        return {}, set()
    try:
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, set()

    done = set()
    for rec in data.get(model_key, []):
        tid = rec.get("task_id")
        if isinstance(tid, list):
            done.update(tid)
        elif tid is not None:
            done.add(tid)
    return data, done

def special_token_for_only_output(model_name):
    if model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
        return "<\uff5cAssistant\uff5c><think>\n"
    elif model_name == "mistralai/Mistral-Nemo-Base-2407":
        return "<|im_start|>assistant\n"
    elif model_name == "Qwen/Qwen3-14B":
        return "\nassistant\n<think>\n"
    elif model_name == "microsoft/Phi-4-reasoning-plus":
        return "assistant<think>"
    elif model_name == "microsoft/Phi-4-reasoning":
        return "assistant<think>"
    elif "gemma" in model_name.lower():
        return "<start_of_turn>model\n"
    # else model_name == "google/gemma-3-12b-it":
    #     return "assistant<think>"
    else:
        warnings.warn(f"Special token for only output is not defined for {model_name}")
        return "Not defined"

# ====== Model Classes ======
class Trainer:
    def __init__(self, model_names, max_length, max_new_tokens, use_qlora=False, model_cache_dir="/mnt/shared/huggingface-cache/hub"):
        # Convert model name to local path if available
        model_path = get_model_path(model_names[0], base_path=model_cache_dir)
        model_name = model_names[0]  # Keep original name for display purposes
        cache_dir = get_cache_dir(base_path=model_cache_dir)  # Get cache directory for downloads
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None or isinstance(self.tokenizer.pad_token_id, list):
            eos_id = self.tokenizer.eos_token_id
            if isinstance(eos_id, list):
                eos_id = eos_id[0]
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(eos_id)
            self.tokenizer.pad_token_id = eos_id
            
        if use_qlora:
            base = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            model = get_peft_model(base, lora_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

        eos_id = model.config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        pad_id = model.config.pad_token_id or eos_id
        if isinstance(pad_id, list):
            pad_id = eos_id

        model.config.pad_token_id = pad_id
        model.config.eos_token_id = eos_id
        model.config.use_cache = False

        self.models = {model_name: model}
        
        self.max_length = int(max_length)
        
        if max_new_tokens != 0:
            self.max_new_tokens = int(max_new_tokens)
        else:
            self.max_new_tokens = -1

    

    @torch.no_grad()
    def _generate_batch(self, model, prompts, n=1):
        dev = next(model.parameters()).device

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(dev)

        input_length = inputs.input_ids.shape[1]
        available_tokens = self.max_length - input_length
        actual_max_new_tokens = self.max_new_tokens if self.max_new_tokens != -1 else available_tokens

        outputs_list, full_outputs_list = [], []
        for _ in range(n):
            g = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
                do_sample=False, # True
                temperature=1.0, # 1.0
                top_k=None, # 50
                top_p=None, # 0.95
                max_new_tokens=actual_max_new_tokens,
                use_cache=True # default False
            )

            batch_texts = self.tokenizer.batch_decode(g, skip_special_tokens=True) # default False

            for idx, full_text in enumerate(batch_texts):
                special_token = special_token_for_only_output(model.config._name_or_path)
                if special_token in full_text and special_token != "Not defined":
                    gen_text = full_text.split(special_token)[-1].strip()
                else:
                # Get actual input length (excluding padding) for this specific sample
                    input_length = inputs.attention_mask[idx].sum().item()
                    # Decode only the generated tokens (skip input tokens)
                    gen_text = self.tokenizer.decode(g[idx][input_length:], skip_special_tokens=True)

                outputs_list.append(gen_text.strip())
                full_outputs_list.append(full_text)

        return outputs_list, full_outputs_list
    

    def generate_responses(self, prompts, idx, contracts, n=1, skip_map=None):
        results = {Path(name).name: [] for name in self.models.keys()}
        for model_name, model in self.models.items():
            short_name = Path(model_name).name
            skip_ids = skip_map.get(short_name, set()) if skip_map else set()
            outputs, full_outputs = self._generate_batch(model, prompts, n)

            for tid, prompt, c, out, full_out in zip(idx, prompts, contracts, outputs, full_outputs):
                if tid in skip_ids:
                    continue

                input_tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
                token_count = count_output_tokens_only(self.tokenizer, prompt, [out])
                contracts_str = _build_contract_str(c) if c else ""

                contract_dict = {}
                for line in contracts_str.splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)
                        contract_dict[key.strip()] = val.strip()

                results[short_name].append({
                    "task_id": tid,
                    "input": prompt,
                    "outputs": [out],
                    "full_outputs": [full_out],
                    "contracts": c,
                    "contracts_list": contract_dict,
                    "input_tokens": input_tokens,
                    "output_tokens": token_count
                })
        return results

def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="humaneval")
    parser.add_argument("--model_name", nargs='+', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts to process per training step (i.e., batch size of tasks)")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of outputs to generate per prompt per agent")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum number of input")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--use_instruction", type=str, help="Include instruction in prompt if set.")
    parser.add_argument("--use_qlora", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--skip_completed", type=lambda x: x.lower() == "true", default=True, help="Skip completed task_ids (True/False)")
    parser.add_argument("--code_gen_mode", type=str, default=None, help="Code generation mode")
    parser.add_argument("--sample_n", type=lambda x: x.lower() == "true", default=False, help="Sample n test cases (True/False)")
    parser.add_argument("--model_cache_dir", type=str, default="/mnt/shared/huggingface-cache/hub", help="Directory path for model cache (download and load)")
    return parser.parse_args()

def all_root(args):
    pwd = Path.cwd()
    model_name = Path(args.model_name[0]).name  
    base_dir   = Path("../../code/output_base") / args.dataset / "inference"
    
    if args.dataset == "total":
        output_dir = Path("../../code/output_full/total/inference") / model_name / args.use_instruction.lower()
    else:
        subfolder = special_map.get(args.use_instruction)
        output_dir = base_dir / subfolder / model_name
    
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = {"generate": "generated_step_all.json",}
    save_root = output_dir / filenames["generate"]
    return str(pwd), str(output_dir), save_root

def single_toggles(contracts_dict):
    single_toggles = []
    for key in contracts_dict.keys():
        single_toggles.append({"id": key, "on": f"({contracts_dict[key]})", "off": f"({contracts_dict[key]})"})
    return single_toggles


if __name__ == "__main__":
    args = input_parser()
    pwd, output_dir, save_root = all_root(args)
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = Trainer(args.model_name, args.max_length, args.max_new_tokens, use_qlora=args.use_qlora, model_cache_dir=args.model_cache_dir)
    train_dataset = load_dataset(args)
    task_ids       = list(train_dataset.keys())
    prompts    = {
        tid: build_prompt(
                args.dataset, 
                tid, 
                example, 
                trainer.tokenizer,
                args.use_instruction, 
                model_base_name=args.model_name,
                code_gen_mode=args.code_gen_mode
            )
        for tid, example in train_dataset.items()
    }
    
    contracts_map = {tid: example["contract"] for tid, example in train_dataset.items()}
    
    for tid in task_ids[:10]:
        print(tid)
        print(prompts[tid])
        print('--------------------------------')
        print('--------------------------------')

    # For test
    if args.sample_n:
        SAMPLE_N = 5
        prompts  = dict(islice(prompts.items(), SAMPLE_N))
        contracts_map = {tid: contracts_map[tid] for tid in prompts}
        task_ids = list(prompts.keys())  
    
    model_key = args.model_name[0].split("/")[1]

    if args.skip_completed:
        all_generated, completed_ids = _read_existing_single(save_root, model_key)
        if not all_generated:
            all_generated = init_dict(model_key, vars(args))
    else:
        all_generated = init_dict(model_key, vars(args))
        completed_ids = set()

    all_generated.setdefault(model_key, [])
    all_generated.setdefault("Setting", {})

    # Truncate the prompt if it is too long
    for tid in task_ids:
        original_prompt = prompts[tid]
        encoded = trainer.tokenizer(original_prompt, return_tensors="pt", truncation=False)
        length = encoded["input_ids"].shape[1]
        if length > args.max_length:
            print(f"[{tid}] Prompt too long ({length} > {args.max_length}), truncating...")
            truncated_ids = encoded["input_ids"][0][:args.max_length]
            prompts[tid] = trainer.tokenizer.decode(truncated_ids, skip_special_tokens=True)

##############################################################################################
## Start of main loop
    start_time = time.time()
    for start in tqdm(range(0, len(task_ids), args.batch_size),
                  desc="Pipeline Progress",
                  total=(len(task_ids) + args.batch_size - 1) // args.batch_size):  
        end = start + args.batch_size
        id_batch       = task_ids[start:end]
        
        if args.skip_completed:
            id_batch = [tid for tid in id_batch if tid not in completed_ids]
        
        if not id_batch:
            print(f"***No tasks to process in batch {start} to {end}***")
            continue
        
        prompt_batch   = [prompts[tid] for tid in id_batch]
        contract_batch = [contracts_map[tid] for tid in id_batch]
        
        responses = trainer.generate_responses(
            prompt_batch,          
            id_batch,              
            contract_batch,        
            n=args.num_samples,
            skip_map={model_key: completed_ids} if args.skip_completed else None
        )
        
        gen_list = responses.get(model_key, [])
        if gen_list:
            all_generated[model_key].extend(gen_list)
            for rec in gen_list:
                tid = rec.get("task_id")
                if tid is not None:
                    completed_ids.add(tid)

        # if args.use_instruction.startswith("GRAMMAR_ASSERT_SPECIFICATION"):
        #     single_toggle = single_toggles(gen_list[0]['contracts_list'])
        #     all_generated[model_key][start]["single_toggles"] = single_toggle
        
        
        save_json(all_generated, save_root)           
        
    end_time = time.time()
    all_generated['Setting']['time_min'] = (end_time - start_time) / 60

    save_json(all_generated, save_root)
    print(f"[Done] Mode `generate` execution completed. All outputs saved once.")
    
    
        