#############
## Imports ##
#############

# import time
# import tracemalloc
import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    # pass
    train_data = pd.read_csv("Bayesian_Question/train_data.csv")
    validation_data = pd.read_csv("Bayesian_Question/validation_data.csv")
    return train_data, validation_data

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    # pass
    # tracemalloc.start()
    # start_time = time.time()
    features = df.columns
    edges = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            edges.append((features[i], features[j]))
    dag = bn.make_DAG(edges)
    # bn.plot(dag, pos=pos)
    model = bn.parameter_learning.fit(dag, df, n_jobs=-1)
    # end_time = time.time()
    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # time_taken = end_time - start_time
    # peak_memory = peak / 1024
    # print('Metrics for base model')
    # print(f"Time taken: {time_taken:} seconds")
    # print(f"Peak memory usage: {peak_memory:} KB")
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    # pass
    # tracemalloc.start()
    # start_time = time.time()
    features = df.columns
    edges = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            edges.append((features[i], features[j]))
    dag = bn.make_DAG(edges)
    dag_ind_test = bn.independence_test(dag, df, test='chi_square', prune=True)
    edges_pruned = dag_ind_test['model_edges']
    dag_pruned = bn.make_DAG(edges_pruned)
    # bn.plot(dag_pruned, pos=pos)
    model = bn.parameter_learning.fit(dag_pruned, df, n_jobs=-1)
    # end_time = time.time()
    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # time_taken = end_time - start_time
    # peak_memory = peak / 1024
    # print('Metrics for pruned model')
    # print(f"Time taken: {time_taken:} seconds")
    # print(f"Peak memory usage: {peak_memory:} KB")
    return model

def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    # pass
    # tracemalloc.start()
    # start_time = time.time()
    dag = bn.structure_learning.fit(df, methodtype='hc')
    # bn.plot(dag, pos=pos)
    model = bn.parameter_learning.fit(dag, df, n_jobs=-1)
    # end_time = time.time()
    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # time_taken = end_time - start_time
    # peak_memory = peak / 1024
    # print('Metrics for optimized model')
    # print(f"Time taken: {time_taken:} seconds")
    # print(f"Peak memory usage: {peak_memory:} KB")
    return model

def save_model(fname, model):
    """Save the model to a file using pickle."""
    # pass
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"Bayesian_Question/{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

pos = {'Start_Stop_ID': (140.0, 380.0), 'End_Stop_ID': (140.0, 340.0), 'Distance': (170.0, 360.0),
    'Zones_Crossed': (170.0, 320.0), 'Route_Type': (155.0, 340.0), 'Fare_Category': (155.0, 300.0)}

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()