# Detection Pipeline 

## Flow

Inputs                        Movie Preprocessing           Detection               trajec. analysis     Database Management

file with meta data        -> multiple meta data files   --------------------------------------------->    -------------
                                                                                                           | SQLite DB |
movie with multiple arenas -> multiple movies of 1 arena -> mult trajectories -> mult. decision data ->    -------------

Status:                     meta: open | data: done          data: done           data: work             db: done | handler: open

## Arena Numbering assignement:

Arenas are counted from the top left starting with zero. Hoprizontal first, than vertical 
e.g.

0 1 2 3
4 5 6 7

## Meta Data needed to run the pipeline


### ToDo
- Experimenter
    - name = Column(String), user input cross checked with database
    
- Experiment Type
    - name

- Experiment (check for duplicates)
    - date_time = Column(DateTime), automated from file I/O input
    - fps = Column(Float), automated from file I/O input
    - video_file_path = Column(String), automated from file I/O input
    - experiment_type = Column(String), user input cross checked with database
    - number_of_arenas = Column(Integer), user input
    - number_of_arena_rows = Column(Integer), user input
    - number_of_arena_columns = Column(Integer), user input




### Done

- Fly (per arena)     
    - is_female = Column(Boolean), user input
    - genotype_id = Column(String), user input cross checked with database
    - age_day_after_eclosion = Column(Float), user input
    - fly_attribute_2 = Column(String), user input cross checked with database
    - fly_attribute_3 = Column(String), user input cross checked with database
    - fly_attribute_4 = Column(String), user input cross checked with database
    - fly_attribute_5 = Column(String), user input cross checked with database

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

- Fly Assignment to Arena
    - check the number of available flies
    - do sex by church divide (left female right male)
    - try to reserve rows for one genotype
    - allow to delete or correct flies by hand


