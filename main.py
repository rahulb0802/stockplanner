import subprocess
def run_script(script_name):
    """Run a Python script using subprocess."""
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode == 0:
        print(f"{script_name} ran successfully")
    else:
        print(f"Error running {script_name}:\n{result.stderr}")

if __name__ == "__main__":
    # Run the train.py script
    run_script('scripts/train.py')
    
    # Run the test.py script
    run_script('scripts/test.py')
    run_script('average.py')
    
