import subprocess
import os

def run_cv4cdd(model_path, input_dir, n_windows=200):
    """
    Activates the Poetry environment and runs CV4CDD using a single shell command on Windows.
    """
    venv_activate = r"C:\Users\Jan Schaffner\AppData\Local\pypoetry\Cache\virtualenvs\supervised-cd-IblWBCGH-py3.9\Scripts\activate.bat"
    cv4cdd_path = r"C:\Users\Jan Schaffner\Documents\Git Repo Clones\cv4cdd\approaches\object_detection"
    script_path = os.path.join(cv4cdd_path, "predict.py")
    output_dir = r"C:\master-thesis\context_drift\data\working_directory_cv4cdd\output_cv4cdd"

    # Create the full Windows shell command
    full_cmd = (
        f'call "{venv_activate}" && '
        f'cd /d "{cv4cdd_path}" && '
        f'python "{script_path}" '
        f'--model-path "{model_path}" '
        f'--log-dir "{input_dir}" '
        f'--encoding winsim '
        f'--n-windows {n_windows} '
        f'--output-dir "{output_dir}"'
    )

    result = subprocess.run(full_cmd, capture_output=True, text=True, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"CV4CDD failed:\n{result.stderr}")

    return output_dir