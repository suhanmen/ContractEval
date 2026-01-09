#!/bin/bash
# This is a script for generation test case or code generation.
##################################################################################################
### **Default parameters**
DATASETS=(
  #"humaneval"
  #"mbpp"
  "total"
)

# use hyper parameters
BATCH_SIZE=10
NUM_SAMPLES=1
MAX_LENGTH=4096
USE_QLORA=False
##################################################################################################
### **Custom parameters**
export CUDA_VISIBLE_DEVICES=${1:-1} # This means which GPU to use.
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

TASK='code_generation' # This parameter is used to determine the task to be executed.
# Default

MODEL_NAMES=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "microsoft/Phi-4-reasoning-plus"
    "Qwen/Qwen3-14B"    
    "google/gemma-3-12b-it"
    "mistralai/Mistral-Nemo-Base-2407"
    "microsoft/Phi-4-reasoning"
    
)
 
CODE_GEN_MODE="SFT" 
# Default

SKIP_COMPLETED=True # This parameter is used to determine whether to skip the completed test cases.
# Components: "True", "False"

SAMPLE_N="False" # This parameter is used to determine whether to sample the test cases.
# Components: "True", "False"

MODEL_CACHE_DIR=None
##################################################################################################
if [ "$TASK" = "code_generation" ]; then
  USE_INSTRUCTIONS=(
    "BASE_CODE_GENERATION"
    "NEGATIVE_EXAMPLE_WITH_BASE"
    "CODE_GENERATION_CS"
    "CODE_GENERATION_CT"
    "CODE_GENERATION_CS_CoT"
    "CODE_GENERATION_CT_CoT"
    "CODE_GENERATION_CS_Multi_turn"
    "CODE_GENERATION_CT_Multi_turn"
    "CODE_GENERATION_CS_Two_turn"
    "CODE_GENERATION_CT_Two_turn"
  )
fi

[ -d log ] || mkdir log

for DATASET in "${DATASETS[@]}"; do
  echo "==============================="
  echo "Using dataset: $DATASET"
  for USE_INSTRUCTION in "${USE_INSTRUCTIONS[@]}"; do
    echo "Using instruction: $USE_INSTRUCTION"
    if [ "$USE_INSTRUCTION" = "MULTI_ASSERT_SPECIFICATION" ] || [ "$USE_INSTRUCTION" = "GRAMMAR_ASSERT_SPECIFICATION" ]; then
      MAX_NEW_TOKENS=0
    else
      MAX_NEW_TOKENS=2048
    fi

    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
      MODEL_SHORT_NAME=$(basename "$MODEL_NAME")
      echo "Running model: $MODEL_NAME"
      echo "==============================="
      mkdir -p log/${DATASET}_${MODEL_SHORT_NAME}
      python ../../code/TG_CG_main.py \
        --dataset "$DATASET" \
        --model_name "$MODEL_NAME" \
        --batch_size $BATCH_SIZE \
        --num_samples $NUM_SAMPLES \
        --max_length $MAX_LENGTH \
        --max_new_tokens $MAX_NEW_TOKENS \
        --use_instruction $USE_INSTRUCTION \
        --use_qlora $USE_QLORA \
        --skip_completed $SKIP_COMPLETED \
        --code_gen_mode $CODE_GEN_MODE \
        --sample_n $SAMPLE_N \
        --model_cache_dir "$MODEL_CACHE_DIR" \
        > log/${DATASET}_${MODEL_SHORT_NAME}/${USE_INSTRUCTION}.log 
    done
  done
done