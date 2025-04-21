#!/bin/bash

torchrun --nproc_per_node=1 --master_port=29510 finetune.py