import argparse
import os
import sys

def check_legal_fn(s):
    import string
    allowed = set(string.ascii_lowercase + string.digits + string.ascii_uppercase + '.()_ ')
    return set(s) <= allowed

def check_legal_cuda_devices(s):
    import string
    allowed = set(string.digits + ',')
    return set(s) <= allowed

parser = argparse.ArgumentParser(description='Trainer files create and run')
parser.add_argument('--trial_name', '-t', type=str, required=True)
parser.add_argument('--cuda_devices', '-c', type=str, required=True)


def query_yes_no(question):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [y/n] "

    while True:
        choice = input(question + prompt).lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n", flush=True)

            
def create_trial_file(file_path):
    global args
    os.system(f'jupyter nbconvert --to python --output "{file_path}" train_book_from_pretrained.ipynb')
    
def main():
    global args
    args = parser.parse_args()
    
    assert(sys.prefix == '/home/idokessler/dl-env'), 'Must be in dl-env venv to run this script'
    assert(check_legal_fn(args.trial_name)), 'trial_name contain unallowed characters'
    assert(check_legal_cuda_devices(args.cuda_devices)), 'cuda devices string is illegal'
    
    file_path = f'python_train_files/trainbook_trial_{args.trial_name}.py'
    if not os.path.exists(file_path):
        if not query_yes_no("trial name doesn't exist. Create it?"):
            return
        create_trial_file(file_path)
    else:
        if query_yes_no("RECREATE trial file?"):
            os.remove(file_path)
            create_trial_file(file_path)
    os.system(f'CUDA_VISIBLE_DEVICES={args.cuda_devices} ipython "{file_path}"')
    
if __name__ == '__main__':
    main()
