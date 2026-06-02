# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess


# for automatically get available servers from slurm job names
agent_paths = {
    "llama3.2_1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.1_8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1_70B": "/project/rcc/mehta5/vllm/models/Meta-Llama-3.1-70B-Instruct",
    "deepseek_r1_32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # DeepSeek R1 Distilled
    "qwen2_1.5B": "Qwen/Qwen2-1.5B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    # "qwen3_4b_hanabi": "/net/projects2/ycleong/sg/strategy-rl/MARSHAL/results/hf_models/selfplay/hanabi_selfplay",
    "qwen2.5_0.5B": "/project/rcc/mehta5/vllm/models/Qwen2.5-0.5B-Instruct",
    "qwen2.5_72B": "/project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct",
}


def _load_servers_from_file(path):
    """Read pre-built server list written by run_all.sbatch."""
    with open(path) as f:
        return json.load(f)


def _discover_servers_from_squeue():
    """Original discovery: parse squeue job names like 'model_key:port'."""
    result = subprocess.run(
        ["squeue", "--me", "-o", '"%j, %N, %T, %i"'], capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")

    # Initialize a list to store LocalModel instances
    local_models = []

    # Iterate over each line, skipping the header
    for line in lines[1:]:
        line = line.strip('"')
        # Get job name, nodelist, and status
        full_job_name, nodelist, status, job_id = line.split(", ")

        assert "[" not in nodelist, "Multi-node servers not currently supported."

        # Parse out the job name and port (if you added it)
        if ":" in full_job_name:
            job_name, port = full_job_name.split(":")
        else:
            job_name = full_job_name
            port = "8000" # Fallback

        # Keep only running jobs
        if status == "RUNNING" and job_name != "bash":
            try:
                model_path = agent_paths[job_name]
                server_address = f"http://{nodelist}:{port}/v1"

                # Check if a model with the same key already exists
                existing_model = next(
                    (m for m in local_models if m["model_key"] == job_name), None
                )

                if existing_model:
                    existing_model["urls"].append(server_address)
                    existing_model["job_ids"].append(job_id)
                else:
                    # Create a new LocalModel instance
                    local_model_info = {
                        "model_key": job_name,
                        "model_id": model_path,
                        "urls": [server_address],
                        "job_ids": [job_id],
                    }
                    local_models.append(local_model_info)
            except KeyError:
                continue

    return local_models


def get_available_servers():
    servers_file = os.environ.get("DECRYPTO_SERVERS_FILE")
    if servers_file and os.path.exists(servers_file):
        return _load_servers_from_file(servers_file)
    return _discover_servers_from_squeue()


if __name__ == "__main__":
    running_jobs = get_available_servers()
    print(running_jobs)
