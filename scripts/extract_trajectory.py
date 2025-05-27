#!/usr/bin/env python3
"""
extract_trajectory.py

Extract the Nth trajectory for a given experimenter from the SQLite DB,
and print both the trial info and the frame-by-frame path.
"""

import os
import sqlite3
import pandas as pd

# ===== User parameters: edit these three =====
DB_PATH      = 'data/fly_choice.db'
EXPERIMENTER = 'Lyall Shannon'
NTH_TRAJ     = 234
# =============================================

def main():
    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # 1) Fetch all trajectories for this experimenter, ordered by trajectory ID
    sql_all = """
    SELECT
      t.id   AS trajectory_id,
      t.trial_id,
      t.pos_x_mm_arena_centered AS x,
      t.pos_y_mm_arena_centered AS y
    FROM trajectories AS t
    JOIN trial      AS tr  ON t.trial_id       = tr.id
    JOIN experiment AS ex  ON tr.experiment_id = ex.id
    JOIN experimenter AS exp ON ex.experimenter_id = exp.id
    WHERE exp.name = ?
    ORDER BY t.id
    """
    df_all = pd.read_sql(sql_all, conn, params=(EXPERIMENTER,))

    total = len(df_all)
    if total == 0:
        print(f"No trajectories found for experimenter “{EXPERIMENTER}”.")
        conn.close()
        return
    if total < NTH_TRAJ:
        print(f"Only {total} trajectories (need {NTH_TRAJ}).")
        conn.close()
        return

    # 2) Select the Nth trajectory (1-based)
    sel = df_all.iloc[NTH_TRAJ - 1]
    trajectory_id = int(sel.trajectory_id)
    trial_id      = int(sel.trial_id)
    print(f"→ Trajectory #{NTH_TRAJ}: trajectory_id={trajectory_id}, trial_id={trial_id}\n")

    # 3) Load the trial metadata from the view two_choice_results
    df_trial = pd.read_sql(
        "SELECT * FROM two_choice_results WHERE trial_id = ?",
        conn,
        params=(trial_id,)
    )

    # 4) Extract the full frame‐by‐frame path for that trial
    df_frames = (
        df_all[df_all.trial_id == trial_id]
        .reset_index(drop=True)
        .copy()
    )
    df_frames.insert(0, 'frame', df_frames.index + 1)
    df_frames = df_frames[['frame', 'x', 'y']]

    conn.close()

    # 5) Print results
    print("=== Trial metadata (two_choice_results) ===")
    print(df_trial.to_string(index=False))

    print("\n=== Trajectory frames (frame, x, y) ===")
    print(df_frames.to_string(index=False))


if __name__ == '__main__':
    main()


# Usage example:
# python extract_trajectory.py
# This script will extract the Nth trajectory for the specified experimenter
# and print the trial metadata along with the frame-by-frame path.
#
# python extract_trajectory.py > traj_234_Doe.txt