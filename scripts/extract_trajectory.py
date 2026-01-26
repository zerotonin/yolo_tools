#!/usr/bin/env python3
"""
extract_trajectory.py

Efficiently extract and plot the Nth trajectory of a given experimenter
from a large SQLite DB without loading everything into memory.
"""

import os, sys, sqlite3, argparse
import pandas as pd
import matplotlib.pyplot as plt

# === adjust these three ===
DB_PATH      = '/home/geuba03p/PyProjects/yolo_tools/data/fly_choice.db'
OUT_DIR      = '/home/geuba03p/PyProjects/fly_decision_analysis/figures'
EXPERIMENTER = 'Lyall Shannon'
NTH_TRAJ     = 2798
# ===========================

# make sure we can import your plotting helpers
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
from yolo_tools.plotting.two_choice_plots import plot_trajectory, save_figure


def main():
    # check DB
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # 1) Pull exactly the Nth trajectory_id & trial_id
    sql_nth = """
    SELECT t.id   AS trajectory_id
         , t.trial_id
    FROM trajectories AS t
    JOIN trial      AS tr  ON t.trial_id       = tr.id
    JOIN experiment AS ex  ON tr.experiment_id = ex.id
    JOIN experimenter AS exp ON ex.experimenter_id = exp.id
    WHERE exp.name = ?
    ORDER BY t.id
    LIMIT 1 OFFSET ?
    """
    cur = conn.execute(sql_nth, (EXPERIMENTER, NTH_TRAJ-1))
    row = cur.fetchone()
    if row is None:
        print(f"No trajectory #{NTH_TRAJ} for experimenter '{EXPERIMENTER}'")
        conn.close()
        return

    trajectory_id, trial_id = row
    print(f"Selected trajectory_id={trajectory_id}, trial_id={trial_id}")

    # 2) Load trial metadata from view
    df_trial = pd.read_sql(
        "SELECT * FROM two_choice_results WHERE trial_id = ?",
        conn,
        params=(trial_id,)
    )

    # 3) Fetch only the frames for that trial
    df_frames = pd.read_sql(
        """
        SELECT pos_x_mm_arena_centered AS pos_x_mm_arena_centered,
               pos_y_mm_arena_centered AS pos_y_mm_arena_centered
        FROM trajectories
        WHERE trial_id = ?
        ORDER BY id
        """,
        conn,
        params=(trial_id,)
    )
    # assign frame index
    df_frames.insert(0, 'frame', df_frames.index + 1)

    conn.close()

    # 4) Print summaries
    print("\n=== Trial metadata ===")
    print(df_trial.to_string(index=False))
    print("\n=== First 5 frames ===")
    print(df_frames.head().to_string(index=False))

    if df_trial.fly_is_female.iloc[0] == 1:
        fly_sex ="fly_female"
    else:
        fly_sex ="fly_male"

    # 5) Plot & save
    stim_01 = f"{df_trial.stimulus_01_name.iloc[0]} {df_trial.stimulus_01_amplitude.iloc[0]} {df_trial.stimulus_01_amplitude_unit.iloc[0]}"
    stim_02 = f"{df_trial.stimulus_02_name.iloc[0]} {df_trial.stimulus_02_amplitude.iloc[0]} {df_trial.stimulus_02_amplitude_unit.iloc[0]}"
  
    fig = plot_trajectory(df_frames, 10,stim_01,stim_02)

    # ensure figures folder
   
    os.makedirs(OUT_DIR, exist_ok=True)
    file_name = f"traj_{NTH_TRAJ}_{EXPERIMENTER.replace(' ','_')}_{fly_sex}"
    save_figure(fig, OUT_DIR, file_name)

    print(f"Saved trajectory plot to {OUT_DIR}/{file_name}.png/svg")


if __name__ == '__main__':
    main()
