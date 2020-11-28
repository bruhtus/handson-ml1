import os
from fire import Fire

def main(file):
    os.system(f'column -s, -t < {file} | less -#2 -N -S')

if __name__ == '__main__':
    Fire(main)
