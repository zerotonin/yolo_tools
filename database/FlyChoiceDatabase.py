from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Table,Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


trial_stimuli_association = Table('trial_stimuli_association', Base.metadata,
    Column('trial_id', Integer, ForeignKey('trial.id'), primary_key=True),
    Column('stimulus_id', Integer, ForeignKey('stimulus.id'), primary_key=True)
)
stimulus_attributes_association = Table('stimulus_attributes_association', Base.metadata,
    Column('stimulus_id', Integer, ForeignKey('stimulus.id'), primary_key=True),
    Column('attribute_id', Integer, ForeignKey('stimuli_attribute.id'), primary_key=True)
)
arena_attributes_association = Table('arena_attributes_association', Base.metadata,
    Column('arena_id', Integer, ForeignKey('arena.id'), primary_key=True),
    Column('attribute_id', Integer, ForeignKey('arena_attribute.id'), primary_key=True)
)
fly_attributes_association = Table('fly_attributes_association', Base.metadata,
    Column('fly_id', Integer, ForeignKey('fly.id'), primary_key=True),
    Column('attribute_id', Integer, ForeignKey('fly_attribute.id'), primary_key=True)
)



class ArenaAttribute(Base):
    """Represents an attribute that can be assigned to arenas, describing their characteristics.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the attribute.
        arenas (relationship): Many-to-many relationship to `Arena`.
    """
    __tablename__ = 'arena_attribute'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    # Relationship back to Arena
    arenas = relationship("Arena", secondary=arena_attributes_association, back_populates="attributes")


class Arena(Base):
    """Represents an arena where experiments are conducted.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the arena.
        size_width_mm (Float): The width of the arena in millimeters.
        size_height_mm (Float): The height of the arena in millimeters.
        size_radius_mm (Float): The radius of the arena in millimeters, if applicable.
        attributes (relationship): Many-to-many relationship to `ArenaAttribute`.
    """
    __tablename__ = 'arena'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    size_width_mm = Column(Float, nullable=True)
    size_height_mm = Column(Float, nullable=True)
    size_radius_mm = Column(Float, nullable=True)
    arena_attribute_1 = Column(Integer, ForeignKey('arena_attribute.id'), nullable=True)
    arena_attribute_2 = Column(Integer, ForeignKey('arena_attribute.id'), nullable=True)
    arena_attribute_3 = Column(Integer, ForeignKey('arena_attribute.id'), nullable=True)
    arena_attribute_4 = Column(Integer, ForeignKey('arena_attribute.id'), nullable=True)
    arena_attribute_5 = Column(Integer, ForeignKey('arena_attribute.id'), nullable=True)
    # New relationship to ArenaAttribute


class StimuliAttribute(Base):
    """Represents an attribute that can be assigned to stimuli, describing their characteristics.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the attribute.
        stimuli (relationship): Many-to-many relationship to `Stimulus`.
    """
    __tablename__ = 'stimuli_attribute'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    # Relationship back to Stimulus
    stimuli = relationship("Stimulus", secondary=stimulus_attributes_association, back_populates="attributes")


class Stimulus(Base):
    """Represents a stimulus used in trials of an experiment.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the stimulus.
        type (String): The type of the stimulus (e.g., "chemical", "light").
        amplitude (Float): The intensity of the stimulus.
        amplitude_unit (String): The unit of the amplitude measurement.
        attributes (relationship): Many-to-many relationship to `StimuliAttribute`.
        trials (relationship): Many-to-many relationship to `Trial` through `trial_stimuli_association`.
    """
    __tablename__ = 'stimulus'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    type = Column(String)
    amplitude = Column(Float)
    amplitude_unit = Column(String)
    stimuli_attribute_1 = Column(Integer, ForeignKey('stimuli_attribute.id'), nullable=True)
    stimuli_attribute_2 = Column(Integer, ForeignKey('stimuli_attribute.id'), nullable=True)
    stimuli_attribute_3 = Column(Integer, ForeignKey('stimuli_attribute.id'), nullable=True)
    stimuli_attribute_4 = Column(Integer, ForeignKey('stimuli_attribute.id'), nullable=True)
    stimuli_attribute_5 = Column(Integer, ForeignKey('stimuli_attribute.id'), nullable=True)
    # Relationship to Experiment
    experiments = relationship("Experiment", secondary=trial_stimuli_association, back_populates="stimuli")
    attributes = relationship("StimuliAttribute", secondary=stimulus_attributes_association, back_populates="stimuli")



class Experimenter(Base):
    """Represents an experimenter who conducts experiments.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the experimenter.
        experiments (relationship): One-to-many relationship to `Experiment`.
    """
    __tablename__ = 'experimenter'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    experiments = relationship("Experiment", back_populates="experimenter")

class Experiment(Base):
    """Represents an experiment conducted by an experimenter.
    
    Attributes:
        id (Integer): The primary key.
        date_time (DateTime): The date and time when the experiment was conducted.
        fps (Float): Frames per second of the video recording of the experiment.
        video_file_path (String): File path to the video recording of the experiment.
        experiment_type_id (Integer): Foreign key to `ChoiceExperimentType`.
        experimenter_id (Integer): Foreign key to `Experimenter`.
        number_of_arenas (Integer): The total number of arenas used in the experiment.
        number_of_arena_rows (Integer): The number of rows of arenas in the setup.
        number_of_arena_columns (Integer): The number of columns of arenas in the setup.
        experimenter (relationship): Many-to-one relationship to `Experimenter`, linking to the experimenter managing the experiment.
        choice_experiment_type (relationship): Many-to-one relationship to `ChoiceExperimentType`, indicating the type of experiment conducted.
        trials (relationship): One-to-many relationship to `Trial`, linking to the trials conducted as part of this experiment.
    """
    __tablename__ = 'experiment'
    id = Column(Integer, primary_key=True)
    date_time = Column(DateTime)
    fps = Column(Float)
    video_file_path = Column(String)
    experiment_type = Column(Integer, ForeignKey('choice_experiment_type.id'))
    number_of_arenas = Column(Integer)
    number_of_arena_rows = Column(Integer)
    number_of_arena_columns = Column(Integer)
    # Relationships
    experimenter = relationship("Experimenter", back_populates="experiments")
    choice_experiment_type = relationship("ChoiceExperimentType", back_populates="experiments")
    trials = relationship("Trial", back_populates="experiment")

class ChoiceExperimentType(Base):
    """Represents the type of a choice experiment.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the experiment type.
        experiments (relationship): One-to-many relationship to `Experiment`.
    """
    __tablename__ = 'choice_experiment_type'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    experiments = relationship("Experiment", order_by=Experiment.id, back_populates="choice_experiment_type")

class FlyAttribute(Base):
    """Represents an attribute that can be assigned to flies, describing their characteristics.
    
    Attributes:
        id (Integer): The primary key.
        name (String): The name of the attribute.
        flies (relationship): Many-to-many relationship to `Fly`.
    """
    __tablename__ = 'fly_attribute'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    flies = relationship("Fly", secondary=fly_attributes_association, back_populates="attributes")


class Genotype(Base):
    """Represents a genotype of a fly.
    
    Attributes:
        id (Integer): The primary key.
        shortname (String): A short name or abbreviation of the genotype.
        genotype (String): The detailed genotype description.
        flies (relationship): One-to-many relationship to `Fly`.
    """
    __tablename__ = 'genotype'
    id = Column(Integer, primary_key=True)
    shortname = Column(String)
    genotype = Column(String)
    flies = relationship("Fly", back_populates="genotype")

class Fly(Base):
    """Represents a fly used in experiments.
    
    Attributes:
        id (Integer): The primary key.
        is_female (Boolean): Whether the fly is female.
        genotype_id (Integer): Foreign key to `Genotype`.
        age_day_after_eclosion (Float): Age of the fly in days after eclosion.
        attributes (relationship): Many-to-many relationship to `FlyAttribute`.
        genotype (relationship): Many-to-one relationship to `Genotype`.
    """
    __tablename__ = 'fly'
    id = Column(Integer, primary_key=True)
    is_female = Column(Boolean)
    genotype_id = Column(Integer, ForeignKey('genotype.id'))
    age_day_after_eclosion = Column(Float)
    fly_attribute_1 = Column(Integer, ForeignKey('fly_attribute.id'), nullable=True)
    fly_attribute_2 = Column(Integer, ForeignKey('fly_attribute.id'), nullable=True)
    fly_attribute_3 = Column(Integer, ForeignKey('fly_attribute.id'), nullable=True)
    fly_attribute_4 = Column(Integer, ForeignKey('fly_attribute.id'), nullable=True)
    fly_attribute_5 = Column(Integer, ForeignKey('fly_attribute.id'), nullable=True)
    # Establishing the relationship to the Genotype table
    genotype = relationship("Genotype", back_populates="flies")
    attributes = relationship("FlyAttribute", secondary=fly_attributes_association, back_populates="flies")

class Trial(Base):
    """Represents a trial within an experiment, including associated stimuli.
    
    Attributes:
        id (Integer): The primary key.
        arena_number (Integer): The identifier for the arena used in the trial.
        experiment_id (Integer): Foreign key to `Experiment`.
        fly_id (Integer): Foreign key to `Fly`.
        arena_id (Integer): Foreign key to `Arena`.
        stimuli (relationship): Many-to-many relationship to `Stimulus` through `trial_stimuli_association`.
        experiment (relationship): Many-to-one relationship to `Experiment`.
        fly (relationship): Many-to-one relationship to `Fly`.
        arena (relationship): Many-to-one relationship to `Arena`.
    """
    __tablename__ = 'trial'
    id = Column(Integer, primary_key=True)
    arena_number = Column(Integer)
    experiment_id = Column(Integer, ForeignKey('experiment.id'))
    fly_id = Column(Integer, ForeignKey('fly.id'))
    arena_id = Column(Integer, ForeignKey('arena.id'))
    stimuli_01 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_02 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_03 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_04 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_05 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_06 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_07 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_08 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_09 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    stimuli_10 = Column(Integer, ForeignKey('stimulus.id'), nullable=True)
    
    # Relationship to Experiment
    experiment = relationship("Experiment", back_populates="trials")
    # Relationship to Fly
    fly = relationship("Fly")
    # Relationship to Arena
    arena = relationship("Arena")
    # New relationship to Stimulus
    stimuli = relationship("Stimulus", secondary=trial_stimuli_association)
    locomotor_data = relationship("Locomotor", back_populates="trial", uselist=False)
    two_choice_decision = relationship("TwoChoiceDecision", back_populates="trial", uselist=False)
    trajectories = relationship("Trajectories", back_populates="trial")
    
class Locomotor(Base):
    """
    Represents locomotor data associated with a trial in an experiment.

    Attributes:
        id (Integer): The primary key.
        trial_id (Integer): Foreign key to `Trial`.
        distance_walked_mm (Float): The total distance walked by the subject in millimeters.
        max_speed_mmPs (Float): The maximum speed achieved by the subject in millimeters per second.
        avg_speed_mmPs (Float): The average speed of the subject in millimeters per second.
        trial (relationship): Many-to-One relationship to `Trial`.
    """
    __tablename__ = 'locomotor'
    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trial.id'))
    distance_walked_mm = Column(Float)
    max_speed_mmPs = Column(Float)
    avg_speed_mmPs = Column(Float)

    # Establishing the relationship to the Trial table
    trial = relationship("Trial", back_populates="locomotor_data")

class TwoChoiceDecision(Base):
    """
    Represents decision-making data for a two-choice trial.

    Attributes:
        id (Integer): The primary key.
        trial_id (Integer): Foreign key to `Trial`.
        fraction_left (Float): Proportion of time spent favoring the left choice.
        fraction_right (Float): Proportion of time spent favoring the right choice.
        fraction_middle (Float): Proportion of time spent that were non-committal.
        fraction_positive (Float): Proportion of time spent on the positive side.
        fraction_negative (Float): Proportion of time spent on the negative side.
        preference_index (Float): A measure of overall preference Michelson contrast.
        decision_to_positive_num (Float): Number of decisions to positive outcomes.
        decision_from_positive_num (Float): Number of decisions from positive outcomes.
        decision_to_negative_num (Float): Number of decisions to negative outcomes.
        decision_from_negative_num (Float): Number of decisions from negative outcomes.
        duration_after_positive (Float): Time spent after positive decisions.
        duration_after_negative (Float): Time spent after negative decisions.
        trial (relationship): Many-to-One relationship to `Trial`.
    """
    __tablename__ = 'two_choice_decision'
    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trial.id'))
    fraction_left = Column(Float)
    fraction_right = Column(Float)
    fraction_middle = Column(Float)
    fraction_positive = Column(Float)
    fraction_negative = Column(Float)
    preference_index = Column(Float)
    decision_to_positive_num = Column(Float)
    decision_from_positive_num = Column(Float)
    decision_to_negative_num = Column(Float)
    decision_from_negative_num = Column(Float)
    duration_after_positive = Column(Float)
    duration_after_negative = Column(Float)

    # Relationship to Trial
    trial = relationship("Trial", back_populates="two_choice_decision")

class Trajectories(Base):
    """
    Represents the trajectory data of a subject within a trial, centered around the arena's coordinates.

    Attributes:
        id (Integer): The primary key for the trajectory record.
        trial_id (Integer): Foreign key linking back to the associated `Trial`.
        pos_x_mm_arena_centered (Float): The X position (in millimeters) of the subject, centered to the arena.
        pos_y_mm_arena_centered (Float): The Y position (in millimeters) of the subject, centered to the arena.
        trial (relationship): Many-to-One relationship to `Trial`, indicating which trial this trajectory data belongs to.
    """
    __tablename__ = 'trajectories'
    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trial.id'))
    pos_x_mm_arena_centered = Column(Float)
    pos_y_mm_arena_centered = Column(Float)

    # Relationship to Trial
    trial = relationship("Trial", back_populates="trajectories")

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

class DatabaseHandler:
    def __init__(self, connection_string):
        """
        Initializes the database handler with a connection string.
        
        Args:
            connection_string (str): The database connection string.
        """
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        if connection_string.startswith('sqlite:///'):
            db_path = connection_string.replace('sqlite:///', '')
            if not os.path.exists(db_path):
                self.create_database()

    def create_database(self):
        """
        Creates the database tables and prints an ASCII art message indicating creation.
        """
        Base.metadata.create_all(self.engine)
        print(r"""
        *********************************************
        *                                           *
        *     New Database Created Successfully!    *
        *                                           *
        *********************************************
        """)

    def __enter__(self):
        self.session = self.Session()
        return self

    def __enter__(self):
        """
        Enters a runtime context related to this object. The with statement will bind this method’s return
        value to the target specified in the as clause of the statement.
        """
        self.session = self.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the runtime context and optionally handles an exception.
        
        Args:
            exc_type: The type of the exception.
            exc_val: The value of the exception.
            exc_tb: The traceback of the exception.
        """
        self.session.close()

    def execute_query(self, query, params=None):
        """
        Executes a SQL query directly.

        Args:
            query (str): The SQL query to execute.
            params (dict, optional): Parameters to pass to the SQL query.

        Returns:
            The result of the query execution.
        """
        if params is None:
            result = self.session.execute(query)
        else:
            result = self.session.execute(query, params)
        return result

    def add_record(self, record):
        """
        Adds a new record to the session.

        Args:
            record (Base): The record (instance of a mapped class) to add.
        """
        self.session.add(record)
        self.session.commit()

    def get_records(self, model, filters=None):
        """
        Retrieves records from the database based on the model and filters provided.

        Args:
            model (Base): The model class to query.
            filters (dict, optional): Conditions to filter the query.

        Returns:
            Query result as a list of model instances.
        """
        query = self.session.query(model)
        if filters:
            query = query.filter_by(**filters)
        return query.all()

    def update_records(self, model, filters, updates):
        """
        Updates records based on the model, filters, and updates provided.

        Args:
            model (Base): The model class to update.
            filters (dict): Conditions to filter the records to update.
            updates (dict): Dictionary of fields to update.
        """
        records = self.session.query(model).filter_by(**filters).update(updates)
        self.session.commit()
        return records

    def delete_records(self, model, filters):
        """
        Deletes records based on the model and filters provided.

        Args:
            model (Base): The model class from which to delete records.
            filters (dict): Conditions to filter the records to delete.
        """
        records = self.session.query(model).filter_by(**filters).delete()
        self.session.commit()
        return records

'''
# Assuming you have a model defined as `MyModel` and SQLAlchemy setup done.
db_url = 'sqlite:///your_database.db'
with DatabaseHandler(db_url) as db:
    # Adding a new record
    new_record = MyModel(name="New Record")
    db.add_record(new_record)

    # Querying records
    records = db.get_records(MyModel, filters={'name': 'New Record'})

    # Updating records
    db.update_records(MyModel, filters={'name': 'New Record'}, updates={'name': 'Updated Record'})

    # Deleting records
    db.delete_records(MyModel, filters={'name': 'Updated Record'})

'''