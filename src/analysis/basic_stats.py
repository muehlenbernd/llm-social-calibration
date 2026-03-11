import json
from collections import defaultdict
from typing import Dict, Tuple, Any
import os


def get_max_sample_ids(json_file_path: str) -> Dict[Tuple[str, str, str, str, str], int]:
    """
    Reads a JSON file and returns the maximum sample_id for each combination
    of (model, scenario, context, utterance, attribute).
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        Dict: Dictionary with combination tuples as keys and max sample_id as values
    """
    # Dictionary to store max sample_id for each combination
    max_sample_ids = defaultdict(lambda: -1)
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Process each entry
    for entry in data:
        # Create combination tuple
        combination = (
            entry['model'],
            entry['scenario'], 
            entry['context'],
            entry['utterance'],
            entry['attribute']
        )
        
        # Update max sample_id for this combination
        current_sample_id = entry['sample_id']
        if current_sample_id > max_sample_ids[combination]:
            max_sample_ids[combination] = current_sample_id
    
    return dict(max_sample_ids)


def display_max_sample_ids(json_file_path: str, output_format: str = 'table') -> None:
    """
    Displays the maximum sample_id for each combination in a readable format.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_format (str): 'table' for formatted table, 'dict' for dictionary display
    """
    max_sample_ids = get_max_sample_ids(json_file_path)
    
    if output_format == 'table':
        # Create formatted table without pandas
        rows = []
        for (model, scenario, context, utterance, attribute), max_sample_id in max_sample_ids.items():
            rows.append([model, scenario, context, utterance, attribute, str(max_sample_id)])
        
        # Sort rows
        rows.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        
        # Calculate column widths
        headers = ['Model', 'Scenario', 'Context', 'Utterance', 'Attribute', 'Max Sample ID']
        col_widths = [len(header) for header in headers]
        
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        
        # Print table
        def print_row(row_data, widths):
            formatted_cells = [cell.ljust(width) for cell, width in zip(row_data, widths)]
            print(" | ".join(formatted_cells))
        
        def print_separator(widths):
            print("-+-".join(["-" * width for width in widths]))
        
        print_row(headers, col_widths)
        print_separator(col_widths)
        
        for row in rows:
            print_row(row, col_widths)
        
    elif output_format == 'dict':
        print("Maximum sample_id for each combination:")
        print("-" * 60)
        for combination, max_sample_id in sorted(max_sample_ids.items()):
            model, scenario, context, utterance, attribute = combination
            print(f"Model: {model}, Scenario: {scenario}, Context: {context}, "
                  f"Utterance: {utterance}, Attribute: {attribute} -> Max sample_id: {max_sample_id}")


def get_summary_stats(json_file_path: str) -> Dict[str, Any]:
    """
    Returns summary statistics about the data.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        Dict: Summary statistics
    """
    max_sample_ids = get_max_sample_ids(json_file_path)
    
    # Calculate statistics
    max_sample_values = list(max_sample_ids.values())
    
    stats = {
        'total_combinations': len(max_sample_ids),
        'overall_max_sample_id': max(max_sample_values) if max_sample_values else 0,
        'overall_min_sample_id': min(max_sample_values) if max_sample_values else 0,
        'average_max_sample_id': sum(max_sample_values) / len(max_sample_values) if max_sample_values else 0,
        'unique_models': len(set(combo[0] for combo in max_sample_ids.keys())),
        'unique_scenarios': len(set(combo[1] for combo in max_sample_ids.keys())),
        'unique_contexts': len(set(combo[2] for combo in max_sample_ids.keys())),
        'unique_utterances': len(set(combo[3] for combo in max_sample_ids.keys())),
        'unique_attributes': len(set(combo[4] for combo in max_sample_ids.keys()))
    }
    
    return stats


def analyze_file(file_path: str, file_description: str) -> None:
    """
    Analyze a single JSON file and display results.
    
    Args:
        file_path (str): Path to the JSON file
        file_description (str): Description of the file for display purposes
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print("="*80)
    print(f"ANALYSIS FOR: {file_description}")
    print("="*80)
    
    print(f"File: {file_path}")
    print("\nMAXIMUM SAMPLE IDS BY COMBINATION:")
    print("-" * 50)
    
    # Display results in table format
    display_max_sample_ids(file_path, output_format='table')
    
    print(f"\nSUMMARY STATISTICS FOR {file_description}:")
    print("-" * 50)
    
    # Show summary statistics
    stats = get_summary_stats(file_path)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n")


def main():
    """
    Analyze both JSON files with sample data.
    """
    # Define paths relative to the project root (two levels up from code/python)
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    
    json_files = [
        {
            'path': os.path.join(project_root, 'results_uncertainty_prompt.json'),
            'description': 'Uncertainty Prompt Results'
        },
        {
            'path': os.path.join(project_root, 'results_combined_prompt.json'),
            'description': 'Combined Prompt Results'
        }
    ]
    
    for file_info in json_files:
        analyze_file(file_info['path'], file_info['description'])


if __name__ == "__main__":
    main()
