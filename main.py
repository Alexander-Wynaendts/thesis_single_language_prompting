import subprocess
import time

start_time = time.time()

# List of script filenames
script_filenames = [
    "xnli_openai.py",
    "xnli_gemini.py",
    "xcopa_openai.py",
    "xcopa_gemini.py",
    "pawsx_openai.py",
    "pawsx_gemini.py",
    "mgsm_openai.py",
    "mgsm_gemini.py"
]

# Loop through each script and execute it
for script in script_filenames:
    script_path = script
    try:
        # Execute the script
        result = subprocess.run(["python", script_path], capture_output=True, text=True)

        # Print the output of the script
        print(f"Output of {script}:")
        print(result.stdout)

        # Print any errors from the script
        if result.stderr:
            print(f"Errors from {script}:")
            print(result.stderr)

    except Exception as e:
        print(f"Failed to execute {script}: {e}")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"All scripts have been executed in: {elapsed_time} seconds")
