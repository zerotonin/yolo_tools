# Detection Pipeline 

## Flow

Inputs                        Movie Preprocessing           Detection               trajec. analysis     Database Management

file with meta data        -> multiple meta data files   --------------------------------------------->    -------------
                                                                                                           | SQLite DB |
movie with multiple arenas -> multiple movies of 1 arena -> mult trajectories -> mult. decision data ->    -------------

Status:                     meta: open | data: done          data: done           data: work             db: done | handler: open


## Meta Data needed to run the pipeline

- Experimenter
    - name = Column(String), user input cross checked with database

- Experiment (check for duplicates)
    - date_time = Column(DateTime), automated from file I/O input
    - fps = Column(Float), automated from file I/O input
    - video_file_path = Column(String), automated from file I/O input
    - experiment_type = Column(String), user input cross checked with database
    - number_of_arenas = Column(Integer)
    - number_of_arena_rows = Column(Integer)
    - number_of_arena_columns = Column(Integer)

- Arena (per arena, check for duplicates)
    - name = Column(String), user input cross checked with database
    - size_width_mm = Column(Float), user input new if not in db
    - size_height_mm = Column(Float), user input new if not in db
    - size_radius_mm = Column(Float), user input new if not in db
    - arena_attribute_1 = Column(String), user input new if not in db
    - arena_attribute_2 = Column(String), user input new if not in db
    - arena_attribute_3 = Column(String), user input new if not in db
    - arena_attribute_4 = Column(String), user input new if not in db
    - arena_attribute_5 = Column(String), user input new if not in db

- Fly (per arena)     
    - is_female = Column(Boolean), user input
    - genotype_id = Column(String), user input cross checked with database
    - age_day_after_eclosion = Column(Float), user input
    - fly_attribute_2 = Column(String), user input cross checked with database
    - fly_attribute_3 = Column(String), user input cross checked with database
    - fly_attribute_4 = Column(String), user input cross checked with database
    - fly_attribute_5 = Column(String), user input cross checked with database

- Trial
    - arena_number = Column(Integer), user input
    - stimuli_01 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_02 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_03 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_04 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_05 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_06 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_07 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_08 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_09 = Column(Integer Foreign Key Stimuli table), automated
    - stimuli_10 = Column(Integer Foreign Key Stimuli table), automated

- Stimuli
    - name = Column(String), user input cross checked with database
    - type = Column(String), user input cross checked with database
    - amplitude = Column(Float), user input cross checked with database
    - amplitude_unit = Column(String), user input cross checked with database
    - stimuli_attribute_1 = Column(String), user input cross checked with database
    - stimuli_attribute_2 = Column(String), user input cross checked with database
    - stimuli_attribute_3 = Column(String), user input cross checked with database
    - stimuli_attribute_4 = Column(String), user input cross checked with database
    - stimuli_attribute_5 = Column(String), user input cross checked with database

## Input Stimuli

ENTER NEW STIMULUS
    Step 1: Show all Stimuli with IDs
    Step 2: Ask for novel Stimuli
    Step 3: User enters values into form

ENTER STIMULUS LIST FOR ARENA:
    Repeat until break or length list is 10:
        Step 1: Show all Stimuli with IDs
        Step 2: If stimulus not present -> ENTER NEW STIMULUS

PATTERN UNIFORM:
    Step 1: ENTER STIMULUS LIST FOR ARENA
    Step 2: Assign Stimulus list to all arenas

PATTERN CHECKERBOARD:
    Step 1: ENTER STIMULUS LIST FOR ARENA
    For each arena_row in range(number_of_arena_rows)
        For each arena_col in range(number_of_arena_columns)
            if arena_row is even:
                if arena_col is even:
                    Assign stimulus list to arena
                else:
                    Assign inverse stimulus list to arena
            else:
                if arena_col is odd:
                    Assign stimulus list to arena
                else:
                    Assign inverse stimulus list to arena


PATTERN INDIVIDUAL:
    Iterate through arena_list:
        Step 1: ENTER STIMULUS LIST FOR ARENA
        Step 2: Assign stimulus list to current arena


ENTER STIMULI FOR EXPERIMENT:
    Step1: User chooses pattern from (horizontal_split, vertical_split, checkerboard, uniform, individual)
    Step2: 
