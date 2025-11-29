from kfp import compiler
from src.pipeline_components import extract_data, preprocess_data, train_model, evaluate_model

# Export each component to YAML using the compiler
compiler.Compiler().compile(extract_data, package_path='components/extract_data.yaml')
compiler.Compiler().compile(preprocess_data, package_path='components/preprocess_data.yaml')
compiler.Compiler().compile(train_model, package_path='components/train_model.yaml')
compiler.Compiler().compile(evaluate_model, package_path='components/evaluate_model.yaml')

print("All components compiled successfully into YAML files in the components/ directory.")
