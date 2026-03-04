#!/bin/bash
mkdir -p data
curl -L -o book-crossing-dataset.zip https://www.kaggle.com/api/v1/datasets/download/syedjaferk/book-crossing-dataset
unzip book-crossing-dataset.zip -d data