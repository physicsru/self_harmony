set -x
module load cuda/12.6
module load cudnn/9.4.0 
module load nccl/2.24.3 

source ~/.bashrc
conda activate MI_release
export NCCL_DEBUG=INFO


unset VLLM_ATTENTION_BACKEND
export VLLM_ENABLE_CUDA_GRAPH=true
export VLLM_ENFORCE_EAGER=false
export WANDB_API_KEY="YOUR WANDB"
# Default model if LLM_PATH is not provided externally
export LLM_PATH=${LLM_PATH:-"Qwen/Qwen3-4B-Base"}

# Control parameter for dynamic vs static generation
USE_DYNAMIC_GENERATION=${USE_DYNAMIC_GENERATION:-true}

# Warmup parameters for dynamic generation format learning
WARMUP_STEPS=${WARMUP_STEPS:-20}
WARMUP_VOTING_STRATEGY=${WARMUP_VOTING_STRATEGY:-"harmonic-mean"}
MAIN_VOTING_STRATEGY=${MAIN_VOTING_STRATEGY:-"harmonic-mean"}

# Regularization parameters
FORMAT_PENALTY_COEF=${FORMAT_PENALTY_COEF:-0.1}
SHARPNESS_OBJECTIVE_COEF=${SHARPNESS_OBJECTIVE_COEF:-0.1}
ENABLE_AUG_GRPO_REWARD=${ENABLE_AUG_GRPO_REWARD:-true}
PENALTY_TYPE=${PENALTY_TYPE:-"consistency"}

# Set train files based on generation mode
if [ "$USE_DYNAMIC_GENERATION" = "true" ]; then
    TRAIN_FILES="data/train_aux_math.parquet"
    EXP_SUFFIX="dynamic_warmup"
else
    TRAIN_FILES="data/train_aux_math.parquet"
    EXP_SUFFIX="static_warmup"
fi

# Harmonic mean reward parameters
USE_HARMONIC_MEAN_REWARD=${USE_HARMONIC_MEAN_REWARD:-false}

# Soft/hard reward control for aug branch (only effective when specific conditions are met)
USE_SOFT_REWARD=${USE_SOFT_REWARD:-false}

# Mathematical equivalence control (use math_verify for equivalence checking)
USE_MATH_EQUIVALENCE=${USE_MATH_EQUIVALENCE:-true}

# Choice handling parameters for mathematical problems
CHOICE_TYPE=${CHOICE_TYPE:-"math"}  # Use "math" for mathematical semantic equivalence
CHOICE_SIZE=${CHOICE_SIZE:-4}       # Irrelevant for math type, but required parameter

# Random seed control
RANDOM_SEED=${RANDOM_SEED:-42}

echo "=== EXPERIMENT CONFIGURATION ==="
echo "Training files: $TRAIN_FILES"
echo "Experiment suffix: $EXP_SUFFIX"
echo "Dynamic generation: $USE_DYNAMIC_GENERATION"
echo "Warmup steps: $WARMUP_STEPS"
echo "Warmup voting strategy: $WARMUP_VOTING_STRATEGY"
echo "Main voting strategy: $MAIN_VOTING_STRATEGY"
echo "Format penalty coefficient: $FORMAT_PENALTY_COEF"
echo "Sharpness objective coefficient: $SHARPNESS_OBJECTIVE_COEF"
echo "Enable aug GRPO reward: $ENABLE_AUG_GRPO_REWARD"
echo "Penalty type: $PENALTY_TYPE"
echo "LLM Path: $LLM_PATH"
echo "Use harmonic mean reward: $USE_HARMONIC_MEAN_REWARD"
echo "Use soft reward: $USE_SOFT_REWARD"
echo "Use mathematical equivalence: $USE_MATH_EQUIVALENCE"
echo "Choice type: $CHOICE_TYPE"
echo "Choice size: $CHOICE_SIZE"
echo "Random seed: $RANDOM_SEED"
echo "================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTHONUNBUFFERED=1

cd [PATH OF SELF_PLAY REPO]
# Spin the Ray server on the last node

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=self_harmony \
    data.train_files=data/train_math.parquet \
    data.train_aug_files=$TRAIN_FILES \
    data.val_files=data/train_math.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.record_entropy=False \
    actor_rollout_ref.model.path=$LLM_PATH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=self_harmony \
    +reward_model.reward_kwargs.voting_strategy=$MAIN_VOTING_STRATEGY \
    +reward_model.reward_kwargs.warmup_steps=$WARMUP_STEPS \
    +reward_model.reward_kwargs.warmup_voting_strategy=$WARMUP_VOTING_STRATEGY \
    +reward_model.reward_kwargs.format_penalty_coef=$FORMAT_PENALTY_COEF \
    +reward_model.reward_kwargs.sharpness_objective_coef=$SHARPNESS_OBJECTIVE_COEF \
    +reward_model.reward_kwargs.enable_aug_grpo_reward=$ENABLE_AUG_GRPO_REWARD \
    +reward_model.reward_kwargs.penalty_type=$PENALTY_TYPE \
    +reward_model.reward_kwargs.use_harmonic_mean_reward=$USE_HARMONIC_MEAN_REWARD \
    +reward_model.reward_kwargs.use_soft_reward=$USE_SOFT_REWARD \
    +reward_model.reward_kwargs.use_math_equivalence=$USE_MATH_EQUIVALENCE \
    +reward_model.reward_kwargs.choice_type=$CHOICE_TYPE \
    +reward_model.reward_kwargs.choice_size=$CHOICE_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=rollout_log/$EXP_SUFFIX-$WARMUP_VOTING_STRATEGY-$EXP_SUFFIX-1024-qwen3-4b-math500-1-0912 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl_ttrl_0907 \
    trainer.experiment_name=$EXP_SUFFIX-$WARMUP_VOTING_STRATEGY-1024-qwen3-4b-math500-1-0912 \
    trainer.default_local_dir=./checkpoint/$EXP_SUFFIX-$WARMUP_VOTING_STRATEGY-1024-qwen3-4b-math500-1-0912 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    +reward_model.reward_kwargs.bootstrap_samples_per_turn=0 \
    +reward_model.reward_kwargs.bootstrap_turns=1 \
    >& output_math500_trainer_0926_qwen3_4b_test.out
