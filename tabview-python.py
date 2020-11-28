import os
from fire import Fire

def main(file_path):
    os.system(f'tabview {file_path}')

if __name__ == '__main__':
    Fire(main)
