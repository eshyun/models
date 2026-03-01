#!/usr/bin/env python3
"""Temporary script to inspect the raw data structure from fetch_data()."""

import json
from pprint import pprint
from models.main import ModelDataFetcher

def find_nested_keys(data, target_keys, path=None, results=None):
    """Recursively search for specific keys in a nested dictionary."""
    if path is None:
        path = []
    if results is None:
        results = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = path + [key]
            if any(target in str(key).lower() for target in target_keys):
                results['.'.join(current_path)] = value
            find_nested_keys(value, target_keys, current_path, results)
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            find_nested_keys(item, target_keys, path + [str(i)], results)
    
    return results

def main():
    print("Fetching data from models.dev API...")
    fetcher = ModelDataFetcher()
    data = fetcher.fetch_data()
    
    # Save the raw data to a file for inspection
    with open('raw_data_dump.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print("Raw data saved to raw_data_dump.json")
    
    # Print basic info about the data structure
    print("\nData structure type:", type(data))
    if isinstance(data, dict):
        print("Top-level keys (providers):", list(data.keys())[:5], "..." if len(data) > 5 else "")
    
    # Print structure of the first provider
    first_provider = next(iter(data.values())) if data else {}
    print("\nStructure of first provider:")
    if isinstance(first_provider, dict):
        print("Keys in provider:", list(first_provider.keys()))
        
        # Check models in the first provider
        if 'models' in first_provider and isinstance(first_provider['models'], dict):
            print("\nFound models in first provider. First model:")
            first_model_id = next(iter(first_provider['models'].keys()))
            first_model = first_provider['models'][first_model_id]
            print(f"Model ID: {first_model_id}")
            print("Model keys:", list(first_model.keys()))
            
            # Print model details
            print("\nModel structure:")
            pprint(first_model)
            
            # Check for cost data
            print("\nCost data in model:")
            if 'cost' in first_model:
                print("Cost data found:")
                pprint(first_model['cost'])
            else:
                print("No direct 'cost' key in model. Checking other locations...")
                
                # Check for cost in other locations
                for key, value in first_model.items():
                    if isinstance(value, dict) and any(k in str(key).lower() for k in ['cost', 'price']):
                        print(f"Found potential cost data in key '{key}':")
                        pprint(value)
    
    # Search for cost-related fields in the entire data structure
    print("\nSearching for cost-related fields in the entire data structure...")
    cost_keywords = ['cost', 'price', 'pricing', 'input', 'output', 'token', 'per_million']
    cost_fields = find_nested_keys(data, cost_keywords)
    
    print("\nFound cost-related fields in the data:")
    if cost_fields:
        print(f"Found {len(cost_fields)} cost-related fields. First few examples:")
        for i, (path, value) in enumerate(cost_fields.items()):
            if i < 10:  # Only show first 10 examples
                print(f"{path}: {value}")
    else:
        print("No cost-related fields found in the data.")
    
    # Try to find the actual models data
    models_data = {}
    for provider, provider_data in data.items():
        if isinstance(provider_data, dict) and 'models' in provider_data:
            models_data.update(provider_data['models'])
    
    if not models_data:
        print("\nCould not find any models data in the response. Check raw_data_dump.json")
        return
    
    print(f"\nFound {len(models_data)} models in the data.")
    
    # Look for cost information in each model
    models_with_costs = []
    
    for model_id, model_data in models_data.items():
        if not isinstance(model_data, dict):
            continue
            
        # Check for cost data in various possible locations
        cost_data = {}
        if 'cost' in model_data and isinstance(model_data['cost'], dict):
            cost_data = model_data['cost']
        elif 'pricing' in model_data and isinstance(model_data['pricing'], dict):
            cost_data = model_data['pricing']
        
        if cost_data:
            models_with_costs.append((model_id, cost_data))
    
    # Print examples of models with cost data
    print(f"\nFound {len(models_with_costs)} models with cost information. Examples:")
    for model_id, cost_data in models_with_costs[:5]:  # Show first 5 examples
        print(f"\nModel: {model_id}")
        pprint(cost_data)
    
    # Print the structure of the first few models for reference
    print("\nStructure of first few models:")
    for i, (model_id, model_data) in enumerate(models_data.items()):
        if i >= 3:  # Only show first 3 models
            break
        print(f"\nModel {i+1}: {model_id}")
        pprint(model_data)

if __name__ == "__main__":
    main()
