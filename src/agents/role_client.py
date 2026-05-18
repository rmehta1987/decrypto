# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from random import sample

import anthropic
import numpy as np
from dotenv import load_dotenv
from litellm import completion
from openai import AzureOpenAI, OpenAI

from src.agents.embedding_baseline import EmbeddingBaseline
from src.env import get_system_prompt
from src.types import AnthropicModel, APIModel, BaselineModel, LocalModel, OpenAIModel
from src.utils.json_utils import extract_json_answer

# Construct the correct path to baseline hint list
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_WORD_LIST = os.path.join(CURRENT_DIR, "baseline_word_list.txt")


class LiteClient:
    def __init__(
        self, api_key, api_version, api_base=None, provider_route="", num_retries=100
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.api_base = api_base
        self.provider_route = provider_route
        self.num_retries = num_retries

    def completion_create(self, model, messages, max_tokens, temperature, seed):
        response = completion(
            model=f"{self.provider_route}{model}",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            api_key=self.api_key,
            api_version=self.api_version,
            api_base=self.api_base,
            num_retries=self.num_retries,
        )
        return response


def initialize_client(model):
    """
    Initialize a client for a given model. Handles API models (Anthropic/OpenAI) and local models.
    :param model: Model dataclass specifying model to initialize client for
    :return: Client instance
    """
    if isinstance(model, BaselineModel):
        return None  # We'll handle baseline models in RoleClient
    elif isinstance(model, APIModel):
        load_dotenv()
        api_key = os.getenv(model.api_key_name)
        if model.use_litellm:
            client = LiteClient(
                api_key=api_key,
                api_version=model.api_version,
                api_base=f"https://{os.getenv(model.api_host_name)}",
                provider_route=model.provider_route,
                num_retries=model.num_retries,
            )
        elif isinstance(model, AnthropicModel):
            client = anthropic.Client(api_key=api_key)
        elif isinstance(model, OpenAIModel):
            # OpenAI models
            if model.use_azure:
                client = AzureOpenAI(
                    api_version=model.api_version,
                    api_key=api_key,
                    azure_endpoint=f"https://{os.getenv(model.api_host_name)}",
                )
            else:
                client = OpenAI(api_key=api_key)
    elif isinstance(model, LocalModel):
        url = sample(model.urls, k=1)[0]
        client = OpenAI(api_key="dummy_key", base_url=url)
    else:
        raise ValueError(
            f"Model type {type(model)} not recognized. Please use APIModel, or LocalModel."
        )
    return client


class RoleClient:
    def __init__(
        self,
        role,
        model,
        mode,
        role_mode=None,
        tom_mode=None,
        baseline_data_dir=None,
        no_error_history=True,
    ):
        self.role = role
        self.model_key = model.model_key
        self.max_tokens = model.max_tokens if hasattr(model, "max_tokens") else 750
        self.max_reasoning_tokens = (
            model.max_reasoning_tokens if hasattr(model, "max_reasoning_tokens") else 0
        )
        self.reasoning_effort = (
            model.reasoning_effort if hasattr(model, "reasoning_effort") else None
        )
        self.temperature = model.temperature if hasattr(model, "temperature") else 0.6
        self.mode = mode
        self.role_mode = role_mode
        self.tom_mode = tom_mode
        self.seed = model.model_seed
        self.no_error_history = no_error_history
        self.model_id = model.model_id
        self.pass_system_as_user = (
            model.pass_system_as_user
            if hasattr(model, "pass_system_as_user")
            else False
        )
        self.use_litellm = model.use_litellm if hasattr(model, "use_litellm") else False

        self.system_prompt = get_system_prompt(role, mode)
        if self.pass_system_as_user or isinstance(model, AnthropicModel):
            # Claude takes system prompt separately from message history
            # o1-preview and R1 do not take system prompts
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system_prompt}]

        if isinstance(model, BaselineModel):
            self.is_baseline = True
            self.baseline_data_dir = baseline_data_dir
            self.k = model.baseline_k if hasattr(model, "baseline_k") else None
            self.global_guess = model.global_guess
            if self.role == "encoder":
                with open(BASELINE_WORD_LIST, "r") as f:
                    hint_words = [line.strip() for line in f]
            else:
                hint_words = None
            if self.model_id.lower() == "glove":
                model_embeddings = np.load(
                    self.baseline_data_dir + "/glove.840B.300d.txt.npy", mmap_mode="r"
                )
                with open(
                    self.baseline_data_dir + "/glove.840B.300d.txt.pickle", "rb"
                ) as f:
                    vocab_to_idx = pickle.load(f)
            elif self.model_id.lower() == "word2vec":
                model_embeddings = np.load(
                    self.baseline_data_dir + "/word2vec_embeddings.npy", mmap_mode="r"
                )
                with open(
                    self.baseline_data_dir + "/word2vec_embeddings.pickle", "rb"
                ) as f:
                    vocab_to_idx = pickle.load(f)
            self.baseline = EmbeddingBaseline(
                role,
                smart_encoder=True,
                no_repeat_hints=True,
                allow_target_keyword=False,
                seed=self.seed,
                k=self.k,
                hint_words=hint_words,
                model=model,
                model_name=model.model_id,
                tokenizer=None,
                model_embeddings=model_embeddings,
                vocab_to_idx=vocab_to_idx,
                global_guess=model.global_guess,
            )

        elif isinstance(model, APIModel) or isinstance(model, LocalModel):
            self.is_baseline = False
            self.client = initialize_client(
                model
            )  # Initialize client for API/Local models

    def chat_completion(
        self,
        message,
        code=None,
        keywords=None,
        hints=None,
        hint_history=None,
        predict_code=False,
        predict_keywords=False,
    ):
        if self.is_baseline:
            return self._baseline_chat_completion(code, keywords, hints, hint_history)
        else:
            return self.get_valid_json_output(message, predict_code, predict_keywords)

    def _baseline_chat_completion(self, code, keywords, hints, hint_history):
        if self.role == "encoder":
            hints = self.baseline.encoder_hint(code, keywords)
            return hints, hints, {"attempts": 0, "fail": 0}
        elif self.role == "decoder":
            guess = self.baseline.decoder_guess(hints, keywords, hint_history)
            guess = [int(g) for g in guess]
            return guess, guess, {"attempts": 0, "fail": 0}
        elif self.role == "interceptor":
            guess = self.baseline.interceptor_guess(hints, hint_history)
            guess = [int(g) for g in guess]
            return guess, guess, {"attempts": 0, "fail": 0}

    def _vllm_chat_completion(self, message, append_to_history=True):
        message_dict = {"role": "user", "content": message}

        if append_to_history:
            self.history.append(message_dict)
            messages = self.history
        else:
            messages = self.history.copy()
            messages.append(message_dict)

        # try:
        import time
        import random
        import openai
        
        max_retries = 10
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.model_key.startswith("claude"):
                    # --- debug ---
                    print("DEBUG: Starting Claude chat completion")
                    # -----------------
                    if self.max_reasoning_tokens <= 0:
                        response = self.client.messages.create(
                            model=self.model_id,
                            messages=messages,
                            system=self.system_prompt,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                        )
                        content = response.content[0].text
                    else:
                        # Extended Thinking
                        # --- debug ---
                        import json
                        print(json.dumps({
                            "model": self.model_id,
                            "max_tokens": self.max_tokens,
                            "thinking": {"type": "enabled", "budget_tokens": self.max_reasoning_tokens},
                            "system": self.system_prompt,
                            "messages": messages,
                        }, indent=2))
                        # -----------------
                        response = self.client.messages.create(
                            model=self.model_id,
                            messages=messages,
                            system=self.system_prompt,
                            max_tokens=self.max_tokens,
                            thinking={
                                "type": "enabled",
                                "budget_tokens": self.max_reasoning_tokens,
                            },
                            # temperature=self.temperature, # No temp if thinking
                        )
                        # --- debug ---
                        print(f"response.content: {response.content}")
                        # -----------------
                        # Model outputs thinking and text blocks. Need to parse accordingly
                        # content = [c.text for c in response.content if c.type == "text"][0]
                        text_blocks = [c.text for c in response.content if c.type == "text"]
                        if not text_blocks:
                            print(f"DEBUG: No text block found! Raw response: {response.content}", flush=True)
                            raise ValueError("Model failed to return a text block (likely hit max_tokens during thinking).")
                        content = text_blocks[0]

                elif self.model_key.startswith("gpt"):
                    # --- debug ---
                    print("DEBUG: Starting gpt chat completion")
                    # -----------------
                    if self.use_litellm:
                        completion_method = self.client.completion_create
                    else:
                        completion_method = self.client.chat.completions.create

                    response = completion_method(
                        model=self.model_id,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        seed=self.seed,
                    )
                    content = response.choices[0].message.content

                elif self.model_key.startswith("o1") or self.model_key.startswith("o3"):
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        max_completion_tokens=self.max_tokens,
                        seed=self.seed,
                        reasoning_effort=self.reasoning_effort,
                    )
                    content = response.choices[0].message.content

                else:  # vLLM models
                    if self.max_reasoning_tokens > 0:
                        # Two-stage thinking-budget pattern for Qwen3-style
                        # reasoning models served via vLLM (which does not yet
                        # support thinking_budget natively, see vllm#17887).
                        #
                        # Stage 1: let the model think up to the budget, stopping
                        # when it emits </think> on its own.
                        stage1 = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            max_tokens=self.max_reasoning_tokens,
                            temperature=self.temperature,
                            seed=self.seed,
                            stop=["</think>"],
                        )
                        thinking_content = stage1.choices[0].message.content or ""
                        # Some servers echo the opening <think> tag; strip it so
                        # we can re-wrap cleanly below.
                        thinking_content = thinking_content.lstrip()
                        if thinking_content.startswith("<think>"):
                            thinking_content = thinking_content[len("<think>"):]
                        thinking_content = thinking_content.strip("\n")

                        # Stage 2: prefill the assistant turn with the (possibly
                        # truncated) reasoning, force-closed with </think>, and
                        # ask the model to continue with the remaining budget.
                        # This guarantees the model is in "answer mode" and must
                        # produce the ANSWER: block.
                        answer_budget = max(
                            self.max_tokens - self.max_reasoning_tokens, 64
                        )
                        prefill = f"<think>\n{thinking_content}\n</think>\n\n"
                        messages_with_prefill = messages + [
                            {"role": "assistant", "content": prefill}
                        ]
                        stage2 = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=messages_with_prefill,
                            max_tokens=answer_budget,
                            temperature=self.temperature,
                            seed=self.seed,
                            extra_body={
                                "continue_final_message": True,
                                "add_generation_prompt": False,
                            },
                        )
                        content = prefill + (stage2.choices[0].message.content or "")
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model_id,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            seed=self.seed,
                        )
                        content = response.choices[0].message.content
                
                # If successful, break the retry loop
                break
                
            except Exception as e:
                is_rate_limit = False
                if isinstance(e, anthropic.RateLimitError) or isinstance(e, openai.RateLimitError):
                    is_rate_limit = True
                elif "RateLimitError" in str(type(e)) or "429" in str(e):
                    is_rate_limit = True
                elif "OverloadedError" in str(type(e)) or "529" in str(e):
                    is_rate_limit = True
                elif "InternalServerError" in str(type(e)) or "500" in str(e) or "502" in str(e) or "503" in str(e):
                    is_rate_limit = True
                elif "APIConnectionError" in str(type(e)):
                    is_rate_limit = True
                    
                if is_rate_limit:
                    if attempt == max_retries - 1:
                        print("DEBUG: Max retries exceeded for RateLimitError")
                        raise e
                    sleep_time = (base_delay ** attempt) + random.uniform(0, 1)
                    print(f"DEBUG: Rate limited. Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    print(f"Error occurred with model {self.model_key} for {self.role}: {e}")
                    # raise e
                    import traceback
                    traceback.print_exc()
                    return None    

        if append_to_history:
            self.history.append({"role": "assistant", "content": content})
        return content

        # except Exception as e:
        #     print(f"Error occurred with model {self.model_key} for {self.role}: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return None

    def reset_used_hints(self):
        if self.is_baseline and self.role == "encoder":
            self.baseline.reset_used_hints()

    def reset_history(self):
        if self.model_key.lower().startswith("claude"):
            # Claude takes system prompt separately from message history
            self.system_prompt = get_system_prompt(self.role, self.mode)
            self.history = []
        else:
            self.history = [
                {"role": "system", "content": get_system_prompt(self.role, self.mode)}
            ]

    def get_valid_json_output(self, prompt, predict_code=False, predict_keywords=False):
        max_attempts = 10
        attempts = 0
        original_history_length = len(self.history)
        original_prompt = prompt
        info = {"attempts": 0, "fail": 0}

        append_to_history = not (predict_code or predict_keywords)

        while attempts < max_attempts:
            output = self._vllm_chat_completion(
                prompt, append_to_history=append_to_history
            )
            json_output = extract_json_answer(output)
            
            # handles invalid json output
            if json_output:
                # "hints" (encode keywords to hints)
                if self.role == "encoder" and not predict_code and "hints" not in json_output:
                    print(f"DEBUG: invalid json_output -- encoder: {json_output}")
                    json_output = None
                # "keywords" (infer the 4 actual keywords)
                elif predict_keywords and "keywords" not in json_output:
                    print(f"DEBUG: invalid json_output -- predict_keywords: {json_output}")
                    json_output = None
                # "guess" (decode the indices of 3 hinted keywords)
                elif (self.role == "decoder" or self.role == "interceptor" or predict_code) and not predict_keywords and "guess" not in json_output:
                    print(f"DEBUG: invalid json_output -- decoder/interceptor/predict_code: {json_output}")
                    json_output = None

            if json_output:
                if self.no_error_history and append_to_history:
                    # Remove all intermediate attempts from history
                    self.history = self.history[:original_history_length]
                    # Add only the successful attempt
                    self.history.append({"role": "user", "content": original_prompt})
                    self.history.append({"role": "assistant", "content": output})

                if self.role == "encoder" and not predict_code:
                    info["attempts"] = attempts
                    return json_output["hints"], output, info
                elif predict_keywords:
                    info["attempts"] = attempts
                    return json_output["keywords"], output, info
                elif (
                    self.role == "decoder" or self.role == "interceptor" or predict_code
                ):
                    info["attempts"] = attempts
                    guess = [int(x) for x in json_output["guess"].split("-")]
                    return guess, output, info
            else:
                attempts += 1
                print(
                    f"Error: Attempt {attempts}/{max_attempts}. Please provide a valid JSON output in the correct format."
                )
                if self.role == "encoder" and not predict_code:
                    error_message = """[GAME_MASTER] Error: Invalid or missing answer. Please provide a concise output formatted like "ANSWER: {"hints": ["hint_1", "hint_2", "hint_3"]}" where hint_1, hint_2, hint_3 are one word each."""
                elif predict_keywords:
                    error_message = """[GAME_MASTER] Error: Invalid or missing answer. Please provide a concise output formatted like "ANSWER: {"keywords": ["keyword_1", "keyword_2", "keyword_3", "keyword_4"]}" where keyword_1, keyword_2, keyword_3, keyword_4 are one word each."""
                elif (
                    self.role == "decoder" or self.role == "interceptor" or predict_code
                ):
                    error_message = """[GAME_MASTER] Error: Invalid or missing answer. Please provide a concise output formatted like "ANSWER: {"guess": "X-Y-Z"}" where X, Y, Z are unique digits from 1 to 4."""

                if append_to_history:
                    prompt = error_message
                else:
                    prompt = original_prompt

        # Handle the case when max attempts are reached
        if self.role == "encoder" and not predict_code:
            failure_prompt = """[GAME_MASTER] Error: Invalid or missing answer after maximum number of attempts. Answer will default to "ANSWER: {"hints": ["no_hint", "no_hint", "no_hint"]}"."""
            json_output = {"hints": ["no_hint", "no_hint", "no_hint"]}
            output = f"ANSWER: {json_output}"
        elif predict_keywords:
            failure_prompt = """[GAME_MASTER] Error: Invalid or missing answer after maximum number of attempts. Answer will default to "ANSWER: {"keywords": ["no_keyword", "no_keyword", "no_keyword", "no_keyword"]}"."""
            json_output = {
                "keywords": ["no_keyword", "no_keyword", "no_keyword", "no_keyword"]
            }
            output = f"ANSWER: {json_output}"
        elif self.role == "decoder" or self.role == "interceptor" or predict_code:
            failure_prompt = """[GAME_MASTER] Error: Invalid or missing answer after maximum number of attempts. Answer will default to "ANSWER: {"guess": "0-0-0"}"."""
            json_output = {"guess": "0-0-0"}
            output = f"ANSWER: {json_output}"

        if append_to_history:
            if self.no_error_history:
                self.history = self.history[: original_history_length + 2]

            self.history.append({"role": "user", "content": failure_prompt})
            self.history.append({"role": "assistant", "content": output})

        info["attempts"] = attempts
        info["fail"] = 1
        output = "Error: Maximum attempts reached. Unable to get valid JSON output."

        if self.role == "encoder" and not predict_code:
            return json_output["hints"], output, info
        elif predict_keywords:
            return json_output["keywords"], output, info
        elif self.role == "decoder" or self.role == "interceptor" or predict_code:
            return [int(x) for x in json_output["guess"].split("-")], output, info
