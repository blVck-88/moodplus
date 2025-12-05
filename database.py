from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# 1. Configuration: Use SQLite (a file-based DB) for simplicity
DATABASE_URL = "sqlite:///wellness_companion.db"

# Create a base class for the ORM models
Base = declarative_base()

# --- TABLE 1: USER ---
class User(Base):
    """
    Stores user authentication and profile information.
    """
    __tablename__ = 'users'

    # Primary Key
    id = Column(Integer, primary_key=True)
    
    # Login Credentials
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # Relationship to Journal Entries (allows accessing user.entries)
    entries = relationship("JournalEntry", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(username='{self.username}')>"

# --- TABLE 2: JOURNAL ENTRY ---
class JournalEntry(Base):
    """
    Stores the daily log, user ratings, and analysis results.
    """
    __tablename__ = 'journal_entries'

    # Primary Key
    id = Column(Integer, primary_key=True)
    
    # Foreign Key linking to the User
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Core Data
    entry_date = Column(Date, nullable=False)
    raw_text = Column(Text)
    mood_rating = Column(Integer) # User's self-reported rating (e.g., 1-10)

    # Analysis Results (to be populated by NLTK/Scikit-learn)
    vader_compound_score = Column(Float)
    ml_stress_level = Column(String) # E.g., 'Low', 'Medium', 'High'

    # Relationship back to the User (allows accessing entry.user)
    user = relationship("User", back_populates="entries")

    def __repr__(self):
        return f"<JournalEntry(date='{self.entry_date}', mood='{self.mood_rating}')>"


# --- Database Initialization Functions ---

def init_db():
    """Initializes the database engine and creates all tables."""
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Returns a new database session."""
    engine = init_db()
    Session = sessionmaker(bind=engine)
    return Session()

# Initialize the database file when the script runs
if __name__ == '__main__':
    engine = init_db()
    print(f"Database initialized at: {DATABASE_URL}")
    print("Tables created: users, journal_entries")