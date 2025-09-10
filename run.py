import subprocess
import os
import sys

def run_streamlit():
    """
    Constructs and executes the command to run the Streamlit application.
    """
    # Get the directory where run.py is located
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the main.py file
    app_path = os.path.join(project_root, "src", "fin_analyzer", "main.py")

    # Check if the app file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find the application file at {app_path}")
        sys.exit(1)

    # The command to execute
    command = ["streamlit", "run", app_path]

    print(f"Executing command: {' '.join(command)}")

    try:
        # Execute the command
        process = subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\nError: 'streamlit' command not found.")
        print("Please ensure Streamlit is installed and that its executable is in your system's PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred while running the Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    run_streamlit()
