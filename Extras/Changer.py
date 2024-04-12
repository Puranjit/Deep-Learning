# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:41:05 2024

@author: puran
"""

import os

def process_text_files(folder_path):
    try:
        # List all .txt files in the specified folder
        txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

        # Iterate through each text file
        for txt_file in txt_files:
            file_path = os.path.join(folder_path, txt_file)

            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Process each line (example condition: replace 'a' with 'x' if line starts with 'a')
            modified_lines = [line.replace('1', '0') if line.startswith('1') else line for line in lines]

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(modified_lines)

            print(f"Processed {txt_file} and saved changes.")

    except Exception as e:
        print(f"Error: {e}")

# Example usage:
folder_path = 'BBery 2022/train/labels/'  # Replace with the path to your folder
process_text_files(folder_path)
