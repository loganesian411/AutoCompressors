import os
import re
from collections import defaultdict
import json

_DEFAULT_OUTPUT_DIR = "/scratch1/loganesi/cs662/output_logs/"
_DEFAULT_ERROR_DIR = "/scratch1/loganesi/cs662/error_logs/"

def generate_filename(params):
    # Process each key and value in the dictionary to create a filename-friendly string
    components = []
    for key, value in params.items():
        # Clean up key to remove special characters like '--'
        key_clean = re.sub(r"[^a-zA-Z0-9]", "", key)
        
        if key == '--model_path':
            value = value.split('/')[-1]
        # Convert the value to a string, handling lists and booleans
        if isinstance(value, list):
            value_str = "_".join(map(str, value))
        elif isinstance(value, bool):
            value_str = "1" if value else "0"
        else:
            value_str = str(value)
        
        # Append the cleaned key and value
        components.append(f"{key_clean}-{value_str}")
    
    # Join all components with underscores to form the filename
    filename = "__".join(components)  
    return filename


def create_sbatch_script_for_seed(seed, runs, script_dir='sbatch_scripts',
            output_dir=_DEFAULT_OUTPUT_DIR, error_dir=_DEFAULT_ERROR_DIR):
    # Group all runs by the seed
    seed_runs = [run for run in runs if run["--seed"] == seed]

    if not seed_runs:
        return  # Skip if no runs for this seed

    # Create output and error directories if they don't exist
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)
    
    # Generate a file name for this seed-specific SBATCH script
    file_name = f"seed_{seed}"
    script_filename = os.path.join(script_dir, file_name + '.sbatch')

    # Define the log file paths for the overall Script
    sbatch_content = f"""#!/bin/bash
#SBATCH --account=jonmay_1455                # jonmay_231
#SBATCH --job-name=evaluate_{file_name}      # Job name for seed {seed}
#SBATCH --output={output_dir}/{file_name}.txt   # Main output log file for job
#SBATCH --error={error_dir}/{file_name}.txt      # Main error log file for job
#SBATCH --ntasks=1                             # Run a single task
#SBATCH --cpus-per-task=4                      # CPU cores per task
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=40G                              # Memory per node
#SBATCH --time=48:00:00                        # Time limit hrs:min:sec
#SBATCH --partition=gpu

module purge
eval "$(conda shell.bash hook)"
conda activate cs662_project
nvidia-smi
"""

    # Add all commands for the configurations for this seed
    for run in seed_runs:
        command = ['python3', '/home1/loganesi/git_repos/cs662/AutoCompressors/evaluate_icl.py']
        
        # Construct the command arguments for each run
        for arg_name, arg_val in run.items():
            if arg_name == '--use_calibration':
                if arg_val:
                    command += [arg_name]
            elif arg_name == '--num_softprompt_demonstrations':
                if len(arg_val) > 0:
                    command += [arg_name]
                    for x in arg_val: command += [str(x)]
            else:
                command += [arg_name,str(arg_val)]
    

        # Generate the unique output path for this command
        run_filename = generate_filename(run)
        command_output_file = os.path.join(output_dir, f"{run_filename}.txt")
        
        # Redirect stdout for this command to its own log file
        sbatch_content += f"{' '.join(command)} > {command_output_file} 2>&1\n"

    # Write the sbatch script to a file
    with open(script_filename, 'w') as f:
        f.write(sbatch_content)

    # Optionally log the file created
    print(f"Generated SBATCH script for seed {seed}: {script_filename}")


def create_runs_from_experiments(experiment_names, datasets, script_dir='./sbatch_scripts/',
                                output_dir=_DEFAULT_OUTPUT_DIR, seeds=[0]):
    ## From table 12. keys are the number of examples needed to get 750 tokens
    ## Following are values for OPT
    datasets_dict_opt = {"ag_news": int(750/65),
                         "sst2": int(750/22),
                         "boolq": int(750/165),
                         "wic": int(750/45),
                         "wsc": int(750/61),
                         "rte": int(750/75),
                         "cb": int(750/98),
                         "copa": int(750/21),
                         "multirc": int(750/350),
                         "mr": int(750/36),
                         "subj": int(750/40)}

    ## Following are values for Llama
    # datasets_dict_opt = {"ag_news": int(750/75),
    #                      "sst2": int(750/25),
    #                      "boolq": int(750/170),
    #                      "wic": int(750/45),
    #                      "wsc": int(750/50),
    #                      "rte": int(750/85),
    #                      "cb": int(750/95),
    #                      "copa": int(750/22),
    #                      "multirc": int(750/350),
    #                      "mr": int(750/40),
    #                      "subj": int(750/40)}

    calibration_datasets = ['ag_news', 'sst2', 'boolq', 'wic', 'multirc']

    # This step creates a dictionary to keep the path to the output file for each experiment
    result_path_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) 
    # result_path_dict[model_name][dataset][experiment_name][seed] = output_path

    runs = []
    for seed in seeds:
        for model_name in experiment_names.keys():
            for dataset in datasets:
                num_examples = datasets_dict_opt[dataset]
                for experiment_name in experiment_names[model_name]:
                    
                    # Clone the base config and add the dataset
                    experiment_config = dict()
                    experiment_config["--model_path"] = model_name
                    experiment_config["--seed"] = seed
                    experiment_config["--dataset"] = dataset
                    experiment_config["--use_calibration"] = dataset in calibration_datasets
                    
                    # Update num_softprompt_demonstrations based on the configuration
                    if experiment_name == 'zero_shot':
                        experiment_config['--num_plaintext_demonstrations'] = 0
                        experiment_config['--num_softprompt_demonstrations'] = []
                    elif experiment_name == 'summary_50':
                        experiment_config['--num_plaintext_demonstrations'] = 0
                        experiment_config['--num_softprompt_demonstrations'] = [num_examples]
                    elif experiment_name == 'summary_100':
                        experiment_config['--num_plaintext_demonstrations'] = 0
                        experiment_config['--num_softprompt_demonstrations'] = [num_examples] * 2
                    elif experiment_name == 'summary_150':
                        experiment_config['--num_plaintext_demonstrations'] = 0
                        experiment_config['--num_softprompt_demonstrations'] = [num_examples] * 3
                    elif experiment_name == 'icl_150':
                        experiment_config['--num_plaintext_demonstrations'] = int(num_examples / 3)
                        experiment_config['--num_softprompt_demonstrations'] = []
                    elif experiment_name == 'icl_750':
                        experiment_config['--num_plaintext_demonstrations'] = num_examples
                        experiment_config['--num_softprompt_demonstrations'] = []
                    else:
                        raise AssertionError("Error: Experiment name does not exist.")
                    
                    output_path = os.path.join(script_dir, output_dir, generate_filename(experiment_config) + '.txt')
                    result_path_dict[model_name][dataset][experiment_name][seed] = output_path
                    runs.append(experiment_config)
    
    # Create SBATCH scripts for each seed
    for seed in seeds:
        create_sbatch_script_for_seed(seed, runs, script_dir=script_dir, output_dir=output_dir)

    return runs, result_path_dict


if __name__ == '__main__':
    # Define experiment names for each model
    # OPT ICL
    # experiment_names = {'facebook/opt-2.7b': ['zero_shot', 'icl_150', 'icl_750'], 
    #                     'princeton-nlp/AutoCompressor-2.7b-6k': ['zero_shot', 'icl_150', 'icl_750', 'summary_50', 'summary_100', 'summary_150']}

    # RMT
    experiment_names = {'princeton-nlp/RMT-2.7b-8k': ['zero_shot', 'icl_150', 'icl_750', 'summary_50', 'summary_100', 'summary_150']}

    # Llama ICL 
    # experiment_names = {'meta-llama/Llama-2-7b-hf': [
    #                         'zero_shot', 'icl_150', 'icl_750'
    #                     ], 
    #                     'princeton-nlp/AutoCompressor-Llama-2-7b-6k': [
    #                         'zero_shot', 'icl_150', 'icl_750', 'summary_50', 'summary_100', 'summary_150'
    #                     ]}

    datasets = ['ag_news', 'sst2', 'boolq', 'wic', 'wsc', 'rte', 'cb', 'copa', 'multirc', 'mr', 'subj']
    seeds = list(range(7))  # or define specific seeds for debugging
    
    # Create runs and result path dict
    runs, result_path_dict = create_runs_from_experiments(experiment_names, datasets, seeds=seeds)

    # Optionally, save result_path_dict to a JSON file
    with open('result_path_dict.json', 'w') as f:
        json.dump(result_path_dict, f)

    print("SBATCH scripts created successfully.")
