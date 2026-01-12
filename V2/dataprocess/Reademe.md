
# Data Split Strategies

This directory contains two different data splitting strategies used for constructing samples in experiments.

## 1. sessionSplit

 `sessionSplit` folder uses a **session-based data splitting strategy**:

- User interactions within a **continuous time period (a session)** are treated as one data sample;
- Each session corresponds to a single data instance;

## 2. userSplit

 `userSplit` folder uses a **user-based data splitting strategy**:

- All interaction records of a **single user** are treated as one data sample;
- Each user corresponds to a single data instance;

## Historical Data Concatenation

Both data splitting strategies apply historical data concatenation:

- The historical sequence length is set to **50** in the experiments;
- Each data sample consists of the current interactions concatenated with the **most recent 50 historical records**.

## Reported Results and Comparison

The results reported in the paper are based on the **user-based data split**.

Our experiments indicate that **user-based splitting is more challenging** than session-based splitting, while **session-based splitting is relatively easier**.

Taking the **NYC dataset** as an example:

- Under the **user-based split**, the accuracy ranges from **0.35 to 0.37**, where **V2 shows a consistent improvement over V1**;
- Under the **session-based split**, the accuracy can reach **0.43 to 0.44**
