#!/bin/bash
for l in zh en xl
  do
    python extract_embeddings.py --model_language $l
  done