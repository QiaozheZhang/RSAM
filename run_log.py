import subprocess
import argparse
import re

def get_args():
    parser = argparse.ArgumentParser(description="help")
    parser.add_argument('--command', nargs=argparse.REMAINDER,
                        help="The command and its arguments to run.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    command = args.command
    command = " ".join(args.command)
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # print(f"{command} is done, with {result}")

    bash_output = result.stdout
    file_path = None
    for line in bash_output.splitlines():
        if "log path is " in line:
            match = re.search(r'log path is (.+)', line)
            if match:
                file_path = match.group(1).strip()
                break

    if file_path:
        print(f"log path is: {file_path}")
    else:
        print("no log path!")

    with open(file_path+".txt", "w") as f:
        f.write(bash_output)