import streamlit as st
import streamlit_authenticator as stauth
from datetime import date
import pandas as pd
import bcrypt
# app.py (top of file)
# ... existing imports ...
from datetime import date # Already there, but confirm it's imported
# --- NEW IMPORT ---
from analysis import analyze_sentiment, classify_stress_level 
# ... rest of the file ...

# --- Import Database Logic ---
# Assuming 'database.py' contains the SQLAlchemy models and get_session() function
from database import User, JournalEntry, get_session 

# --- FUNCTIONS FOR DB INTERACTION ---

def hash_password(password):
    """Hashes the password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def get_all_users(session):
    """Retrieves all usernames and their hashed passwords from the database."""
    users = session.query(User).all()
    usernames = {user.username: user.hashed_password for user in users}
    return usernames

def add_new_user(session, username, password):
    """Adds a new user to the database after hashing the password."""
    try:
        hashed_pwd = hash_password(password)
        new_user = User(username=username, hashed_password=hashed_pwd)
        session.add(new_user)
        session.commit()
        st.success(f"Account created successfully for **{username}**! Please log in.")
        return True
    except Exception as e:
        session.rollback()
        st.error(f"Error creating user: Username '{username}' already exists.")
        return False

# app.py (add this function near get_all_users/add_new_user)
def save_journal_entry(session, user_id, entry_date, raw_text, mood_rating, compound_score, stress_level):
    """Saves a new JournalEntry object to the database."""
    try:
        new_entry = JournalEntry(
            user_id=user_id,
            entry_date=entry_date,
            raw_text=raw_text,
            mood_rating=mood_rating,
            vader_compound_score=compound_score,
            ml_stress_level=stress_level
        )
        session.add(new_entry)
        session.commit()
        st.success(f"Entry saved successfully for {entry_date}!")
        return True
    except Exception as e:
        session.rollback()
        st.error(f"Error saving entry. Did you already log an entry for today? ({e})")
        return False
    
# --- STREAMLIT APP LOGIC ---

def run_app():
    st.set_page_config(page_title="Wellness Companion", layout="centered")
    st.title("🧠 Wellness Companion")

    # 1. Initialize DB Session
    session = get_session()

    # 2. Retrieve existing users for the Authenticator
    usernames_passwords = get_all_users(session)
    
    # Format for stauth: keys are usernames, values are hashed passwords
    credentials = {"usernames": usernames_passwords}
    
    # Map usernames to a more display-friendly 'names' dictionary if needed, 
    # but for simplicity, we'll use usernames as the names for now.
    names = {user: user for user in usernames_passwords.keys()}

    # 3. Initialize Streamlit Authenticator
    authenticator = stauth.Authenticate(
        credentials,
        'wellness_cookie',  # Cookie name
        'secure_key_123',   # Secret key (CHANGE THIS IN A REAL APP!)
        30                 # Cookie expiry days
    )

    # 4. Handle Authentication Status
    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status == False:
        st.error('Username/password is incorrect')

    elif authentication_status == None:
        st.warning('Please enter your username and password')
        
        # Display the Signup Form below the login fields
        st.subheader("New User? Sign Up!")
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type='password')
            new_password_confirm = st.text_input("Confirm Password", type='password')
            submitted = st.form_submit_button("Create Account")

            if submitted:
                if not new_username or not new_password:
                    st.error("Username and Password are required.")
                elif new_password != new_password_confirm:
                    st.error("Passwords do not match.")
                elif new_username in usernames_passwords:
                    st.error(f"Username '{new_username}' already taken.")
                else:
                    add_new_user(session, new_username, new_password)
                    # Force a refresh to show the login success message / new user list
                    st.experimental_rerun()


    # app.py (inside the 'elif authentication_status:' block)

    elif authentication_status:
        # --- USER IS LOGGED IN ---
        st.sidebar.caption(f'Logged in as **{username}**')
        authenticator.logout('Logout', 'sidebar')

        # Get the current user's database ID
        current_user_db = session.query(User).filter_by(username=username).first()
        current_user_id = current_user_db.id
        
        st.header(f"Hello, {current_user_db.username}! 👋")
        
        # --- DAILY JOURNAL INPUT FORM ---
        st.subheader("How was your day?")
        with st.form("daily_entry_form", clear_on_submit=True):
            
            entry_date = st.date_input("Date of Entry", date.today())
            
            # Mood Rating Slider
            mood_rating = st.slider(
                "Self-Reported Mood (1=Awful, 10=Fantastic)", 
                min_value=1, max_value=10, value=5, step=1
            )
            
            # Main Journal Text Area
            journal_text = st.text_area(
                "Journal Entry (Write about your day, feelings, and events):",
                height=200
            )
            
            submitted = st.form_submit_button("Analyze & Save Entry")

            if submitted:
                if not journal_text:
                    st.warning("Please write something in your journal entry before saving.")
                else:
                    # 1. Perform Sentiment Analysis
                    compound_score = analyze_sentiment(journal_text)
                    
                    # 2. Get Simple Stress Classification
                    stress_level = classify_stress_level(compound_score, mood_rating)

                    # 3. Save to DB
                    save_journal_entry(
                        session=session,
                        user_id=current_user_id,
                        entry_date=entry_date,
                        raw_text=journal_text,
                        mood_rating=mood_rating,
                        compound_score=compound_score,
                        stress_level=stress_level
                    )
                    
                    # Provide immediate feedback on the analysis
                    st.info(f"Sentiment Analysis Score (VADER): **{compound_score:.3f}** (Range: -1 to +1)")
                    st.info(f"Estimated Stress Level: **{stress_level}**")
                    
        # Placeholder for the Dashboard (Next Step!)
        st.markdown("---")
        st.header("📊 Your Wellness Dashboard")
        st.write("Historical data and trends will appear here.")
        
    session.close()
        
    


if __name__ == '__main__':
    run_app()