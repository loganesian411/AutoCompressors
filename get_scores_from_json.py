import json
import argparse
import copy

# Initialize the parser
parser = argparse.ArgumentParser(description="argparser for reading result_path_dict.json files and getting accuracy scores")

# Positional argument
parser.add_argument("--input_json_filename", type=str, default='result_path_dict.json')
parser.add_argument("--output_json_filename", type=str, default='accuracy_dict.json')
args = parser.parse_args()

# read the dictionary that has the output txt paths
result_path_dict = json.load(open(args.input_json_filename,'r'))

# create a copy of the path dict. the path items will be replaced with accuracy scores
accuracy_dict = copy.deepcopy(result_path_dict)

for model, model_data in result_path_dict.items():
    for dataset, dataset_data in model_data.items():
        for experiment, experiment_data in dataset_data.items():
            for seed, result_path in experiment_data.items():
                # read the result path
                with open(result_path, 'r') as file:
                    lines = file.readlines()
                try:
                    acc = float(lines[-1].split()[-1])
                    accuracy_dict[model][dataset][experiment][seed] = acc
                except (ValueError, IndexError) as e:
                    print(f'skipping {result_path}')

json.dump(accuracy_dict,open(args.output_json_filename,'w'))
