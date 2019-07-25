#!/bin/bash

python tools/process.py --input_dir photos/original --operation resize --output_dir photos/original_resized &&
python tools/process.py --input_dir photos/landmarks --operation resize --output_dir photos/landmarks_resized &&
python tools/process.py --input_dir photos/landmarks_resized --b_dir photos/original_resized --operation combine --output_dir photos/combined &&
python tools/split.py --dir photos/combined                 




