import os
import json
import re
import glob
import argparse
from pathlib import Path

def parse_true_keywords(prompt):
    match = re.search(r'Keywords:\s*\n\s*1\.\s*(\w+)\s*\n\s*2\.\s*(\w+)\s*\n\s*3\.\s*(\w+)\s*\n\s*4\.\s*(\w+)', prompt)
    if match:
        return [match.group(1), match.group(2), match.group(3), match.group(4)]
    return []

def parse_keywords(response):
    if not response:
        return []
    match = re.search(r'ANSWER:\s*(\{.*\})', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("keywords", [])
        except:
            return []
    return []

def parse_code_guess(response):
    if not response:
        return None
    match = re.search(r'ANSWER:\s*(\{.*\})', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("guess", None)
        except:
            return None
    return None

def compare_kw_lists(l1, l2):
    if not l1 or not l2 or len(l1) != len(l2):
        return False
    return all(k1.lower() == k2.lower() for k1, k2 in zip(l1, l2))

def reconstruct_trajectories(results_dir, output_dir):
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    
    is_piaget = "piaget" in results_dir.name.lower()
    exp_type = "piaget" if is_piaget else "gopnik"

    for combined_path in results_dir.glob("*/*/*/*/combined_history.json"):
        episode_dir = combined_path.parent
        
        with open(combined_path, "r") as f:
            combined_history = json.load(f)
            
        rel_path = episode_dir.relative_to(results_dir)
        out_episode_dir = output_dir / f"{exp_type}_trajectories" / rel_path
        out_episode_dir.mkdir(parents=True, exist_ok=True)
            
        if is_piaget:
            encoder_path = episode_dir / "encoder_history.json"
            if not encoder_path.exists():
                continue
            with open(encoder_path, "r") as f:
                encoder_history = json.load(f)
                
            piaget_queries = {}
            current_turn = 1
            for i in range(len(combined_history)):
                msg = combined_history[i]
                if msg["role"] == "user" and "Turn " in msg["content"] and " summary:" in msg["content"]:
                    match = re.search(r'Turn (\d+) summary:', msg["content"])
                    if match:
                        current_turn = int(match.group(1)) + 1
                        
                if msg["role"] == "user" and "[ENCODER] As the Encoder, you've provided the following hints" in msg["content"]:
                    content = msg["content"]
                    if i + 1 < len(combined_history):
                        resp = combined_history[i+1]
                        if resp["role"] == "assistant" and "[ENCODER]" in resp["content"]:
                            if current_turn not in piaget_queries:
                                piaget_queries[current_turn] = {}
                            piaget_queries[current_turn] = {
                                "prompt": content.replace("[ENCODER] ", ""),
                                "response": resp["content"].replace("[ENCODER] ", "")
                            }
            
            # Get summary info to extract Eve's guess and actual code
            turn_summaries = {}
            for msg in combined_history:
                if msg["role"] == "user" and " summary:" in msg["content"]:
                    match = re.search(r'Turn (\d+) summary:', msg["content"])
                    if match:
                        t = int(match.group(1))
                        code_match = re.search(r'Code\s*:\s*(\d-\d-\d)', msg["content"])
                        eve_match = re.search(r'Interceptor guess\s*:\s*(\d-\d-\d)', msg["content"])
                        turn_summaries[t] = {
                            "code": code_match.group(1) if code_match else None,
                            "eve_guess": eve_match.group(1) if eve_match else None
                        }
            
            for turn, query in piaget_queries.items():
                prompt = query["prompt"]
                response = query["response"]
                alice_pred = parse_code_guess(response)
                
                summary = turn_summaries.get(turn, {})
                eve_guess = summary.get("eve_guess")
                actual_code = summary.get("code")
                
                predicts_eve = (alice_pred == eve_guess) if alice_pred and eve_guess else False
                predicts_code = (alice_pred == actual_code) if alice_pred and actual_code else False
                interception_success = (eve_guess == actual_code) if eve_guess and actual_code else False
                
                base_history = encoder_history[:2*turn + 1]
                
                piaget_traj = list(base_history)
                piaget_traj.append({"role": "user", "content": prompt})
                piaget_traj.append({"role": "assistant", "content": response})
                piaget_traj.append({
                    "role": "experiment_results",
                    "content": {
                        "predicts_eve": predicts_eve,
                        "predicts_code": predicts_code,
                        "interception_success": interception_success
                    }
                })
                with open(out_episode_dir / f"turn_{turn}_piaget.json", "w") as f:
                    json.dump(piaget_traj, f, indent=2)
                    
        else:
            interceptor_path = episode_dir / "interceptor_history.json"
            if not interceptor_path.exists():
                continue
            with open(interceptor_path, "r") as f:
                interceptor_history = json.load(f)
                
            gopnik_queries = {}
            for i in range(len(combined_history)):
                msg = combined_history[i]
                if msg["role"] == "user" and "[INTERCEPTOR] Turn " in msg["content"]:
                    content = msg["content"]
                    turn_match = re.search(r'\[INTERCEPTOR\] Turn (\d+):', content)
                    if not turn_match:
                        continue
                    turn = int(turn_match.group(1))
                    
                    if "Given this hint history, what do you think are the four secret keywords" in content:
                        q_type = "vanilla"
                    elif "Before the keywords were revealed to you" in content:
                        q_type = "rep_change"
                    elif "Another player, the Second Interceptor" in content:
                        q_type = "false_belief"
                    else:
                        continue
                        
                    if i + 1 < len(combined_history):
                        resp = combined_history[i+1]
                        if resp["role"] == "assistant" and "[INTERCEPTOR]" in resp["content"]:
                            if turn not in gopnik_queries:
                                gopnik_queries[turn] = {}
                            gopnik_queries[turn][q_type] = {
                                "prompt": content.replace("[INTERCEPTOR] ", ""),
                                "response": resp["content"].replace("[INTERCEPTOR] ", "")
                            }
            
            for turn, queries in gopnik_queries.items():
                if "vanilla" not in queries or "rep_change" not in queries or "false_belief" not in queries:
                    continue
                    
                vanilla_prompt = queries["vanilla"]["prompt"]
                vanilla_response = queries["vanilla"]["response"]
                rep_change_prompt = queries["rep_change"]["prompt"]
                rep_change_response = queries["rep_change"]["response"]
                false_belief_prompt = queries["false_belief"]["prompt"]
                false_belief_response = queries["false_belief"]["response"]
                
                vanilla_pred = parse_keywords(vanilla_response)
                rep_change_pred = parse_keywords(rep_change_response)
                false_belief_pred = parse_keywords(false_belief_response)
                true_keywords = parse_true_keywords(rep_change_prompt)
                
                vanilla_correct = compare_kw_lists(vanilla_pred, true_keywords)
                valid_eval = not vanilla_correct and len(true_keywords) == 4 and len(vanilla_pred) == 4
                
                if valid_eval:
                    strong_rp = compare_kw_lists(vanilla_pred, rep_change_pred)
                    weak_rp = not compare_kw_lists(true_keywords, rep_change_pred) if len(rep_change_pred) == 4 else False
                    strong_fb = compare_kw_lists(vanilla_pred, false_belief_pred)
                    weak_fb = not compare_kw_lists(true_keywords, false_belief_pred) if len(false_belief_pred) == 4 else False
                else:
                    strong_rp = False
                    weak_rp = False
                    strong_fb = False
                    weak_fb = False
                    
                base_history = interceptor_history[:2*turn - 1]
                
                vanilla_traj = list(base_history)
                vanilla_traj.append({"role": "user", "content": vanilla_prompt})
                vanilla_traj.append({"role": "assistant", "content": vanilla_response})
                vanilla_traj.append({
                    "role": "experiment_results",
                    "content": {
                        "vanilla_correct": vanilla_correct
                    }
                })
                with open(out_episode_dir / f"turn_{turn}_vanilla.json", "w") as f:
                    json.dump(vanilla_traj, f, indent=2)
                    
                rp_traj = list(base_history)
                rp_traj.append({"role": "user", "content": rep_change_prompt})
                rp_traj.append({"role": "assistant", "content": rep_change_response})
                rp_traj.append({
                    "role": "experiment_results",
                    "content": {
                        "strong_success": strong_rp,
                        "weak_success": weak_rp
                    }
                })
                with open(out_episode_dir / f"turn_{turn}_rep_change.json", "w") as f:
                    json.dump(rp_traj, f, indent=2)
                    
                fb_traj = list(base_history)
                fb_traj.append({"role": "user", "content": false_belief_prompt})
                fb_traj.append({"role": "assistant", "content": false_belief_response})
                fb_traj.append({
                    "role": "experiment_results",
                    "content": {
                        "strong_success": strong_fb,
                        "weak_success": weak_fb
                    }
                })
                with open(out_episode_dir / f"turn_{turn}_false_belief.json", "w") as f:
                    json.dump(fb_traj, f, indent=2)
                    
    print(f"Successfully reconstructed trajectories into {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    reconstruct_trajectories(args.results_dir, args.output_dir)
