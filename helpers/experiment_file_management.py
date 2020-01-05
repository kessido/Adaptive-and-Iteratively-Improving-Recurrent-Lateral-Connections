def get_log_path(trial_name):
    return f"/home/idokessler/feedback-loop/runs/{trial_name}.pk"

def is_log_exist(trial_name):
    import os
    return os.path.isfile(get_log_path(trial_name))

def save_trial_results(trial_name, res):
    import pickle
    import gzip
    with gzip.open(get_log_path(trial_name), "wb") as f:
        pickle.dump(res, f)

def load_trial_results(trial_name):
    import pickle
    import gzip
    with gzip.open(get_log_path(trial_name), "rb") as f:
        return pickle.load(f)

def delete_trial_results(trial_name):
    import os
    os.remove(get_log_path(trial_name))