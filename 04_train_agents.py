"""
Step 4 — Train PPO and/or SAC agents on AORVAEnv.

Prerequisites: run steps 00, 01, 02, 02b, 03 first.

Outputs:
    models/ppo_aorva_final.zip
    models/sac_aorva_final.zip
    models/ppo_best/best_model.zip
    models/sac_best/best_model.zip
    logs/ppo/  (TensorBoard)
    logs/sac/  (TensorBoard)

Usage:
    python scripts/04_train_agents.py ppo
    python scripts/04_train_agents.py sac
    python scripts/04_train_agents.py both              (default)
    python scripts/04_train_agents.py both --ppo-steps 500000 --sac-steps 300000

Monitor training:
    tensorboard --logdir logs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_agents import main

if __name__ == "__main__":
    main()
