import subprocess

trials = 5

for idx in range(trials):
    program = [f'./main.py {str(idx + 1)}']

    subprocess.call(program, shell=True) # , capture_output=True)
    print("Finished:" + program[0])
