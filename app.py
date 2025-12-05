import streamlit as st
import streamlit_authenticator as stauth
from datetime import date
from analysis import analyze_sentiment, classify_stress_level, generate_suggestions_with_model
import pandas as pd
import bcrypt
import plotly.express as px # NEW IMPORT

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

# --- STREAMLIT APP LOGIC ---
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

# app.py (Add these functions below your data saving function)

def get_user_entries_df(session, user_id):
    """Retrieves all journal entries for a user and returns them as a Pandas DataFrame."""
    
    # Query the database
    entries = session.query(JournalEntry).filter_by(user_id=user_id).order_by(JournalEntry.entry_date).all()
    
    # Convert list of SQLAlchemy objects to a list of dictionaries
    data = []
    for entry in entries:
        data.append({
            'Date': entry.entry_date,
            'Mood Rating (1-10)': entry.mood_rating,
            'VADER Sentiment Score': entry.vader_compound_score,
            'Stress Level': entry.ml_stress_level,
        })
        
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    # Ensure Date is the index and data is sorted
    if not df.empty:
        df = df.set_index('Date')
        df.sort_index(inplace=True)
        
    return df

def display_dashboard(df):
    """Creates and displays interactive Plotly charts."""
    
    if df.empty:
        st.info("Log your first journal entry to see your dashboard!")
        return

    st.subheader("Mood and Sentiment Trends Over Time")

    # 1. Line Chart for Mood Rating and VADER Score
    
    # Prepare data for plotting (VADER and Mood)
    plot_df = df[['Mood Rating (1-10)', 'VADER Sentiment Score']]
    
    fig_line = px.line(
        plot_df, 
        y=plot_df.columns, 
        title='Daily Mood vs. Sentiment Score',
        height=400
    )
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Score")
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. Bar Chart for Stress Level Distribution
    
    # Count the occurrences of each stress level
    stress_counts = df['Stress Level'].value_counts().reset_index()
    stress_counts.columns = ['Stress Level', 'Count']
    
    # Define order and colors for better visualization
    stress_order = ['Low', 'Medium', 'High']
    stress_counts['Stress Level'] = pd.Categorical(stress_counts['Stress Level'], categories=stress_order, ordered=True)
    stress_counts = stress_counts.sort_values('Stress Level')
    
    color_map = {'Low': 'lightgreen', 'Medium': 'gold', 'High': 'salmon'}

    fig_bar = px.bar(
        stress_counts, 
        x='Stress Level', 
        y='Count', 
        color='Stress Level',
        color_discrete_map=color_map,
        title='Distribution of Calculated Stress Levels',
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- STREAMLIT APP LOGIC ---
# (Keep all your existing function definitions: hash_password, get_all_users, add_new_user, 
# save_journal_entry, get_user_entries_df, display_dashboard)
# ...

def run_app():
    st.set_page_config(page_title="Wellness Companion", layout="wide") # Use wide for better dashboard view
    st.title("🧠 Wellness Companion")

    # --- SESSION STATE INITIALIZATION ---
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    # 1. Initialize DB Session
    session = get_session()

    # 2. Retrieve existing users for the Authenticator
    usernames_passwords = get_all_users(session)

    # --- MAIN APPLICATION LOGIC ---

    if st.session_state['authenticated']:
        # --- USER IS LOGGED IN ---
        username = st.session_state['username']
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header(f"Hello, {username}! 👋")
        with col2:
            if st.button("Logout", key="logout_btn"):
                st.session_state['authenticated'] = False
                st.session_state['username'] = None
                st.rerun()
        
        # Get the current user's database ID
        current_user_db = session.query(User).filter_by(username=username).first()
        current_user_id = current_user_db.id
        
        # --- DAILY JOURNAL INPUT FORM & DASHBOARD LAYOUT ---
        st.markdown("---")
        input_col, dashboard_col = st.columns([1, 2], gap="large")

        with input_col:
            st.subheader("✍️ Log Your Day")
            with st.form("daily_entry_form", clear_on_submit=True):
                
                entry_date = st.date_input("Date of Entry", date.today())
                
                mood_rating = st.slider(
                    "Self-Reported Mood (1=Awful, 10=Fantastic)", 
                    min_value=1, max_value=10, value=5, step=1
                )
                
                journal_text = st.text_area(
                    "Journal Entry (Write about your day, feelings, and events):",
                    height=200
                )
                
                submitted = st.form_submit_button("Analyze & Save Entry")

                # app.py (inside the 'elif st.session_state["authenticated"]:' block, inside the 'if submitted:' block)

                if submitted:
                    if not journal_text:
                        st.warning("Please write something in your journal entry before saving.")
                    else:
                        # 1. Perform Sentiment Analysis
                        compound_score = analyze_sentiment(journal_text)
                        
                        # 2. Get Simple Stress Classification
                        stress_level = classify_stress_level(compound_score, mood_rating)

                        # --- NEW: Get Historical Data before saving to include it in the analysis ---
                        history_df_before_save = get_user_entries_df(session, current_user_id)
                        
                        # 3. Generate Suggestions using AI model
                        suggestions = generate_suggestions_with_model(compound_score, stress_level, journal_text, history_df_before_save)

                        # 4. Save to DB
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
                        st.success(f"✅ Entry saved successfully!")
                        st.info(f"VADER Score: **{compound_score:.3f}** | Stress Level: **{stress_level}**")
                        
                        st.subheader("🎯 Actionable Wellness Suggestions:")
                        with st.container():
                            for suggestion in suggestions:
                                st.markdown(suggestion)
                        
        with dashboard_col:
            st.header("📊 Your Wellness Dashboard")
            
            # 1. Get the data
            history_df = get_user_entries_df(session, current_user_id)
            
            # 2. Display the data/charts
            display_dashboard(history_df)
            
    else:
        # --- LOGIN / SIGNUP VIEW ---
        st.subheader("Login to Your Account")
        
        # Manual Login Form
        with st.form("login_form"):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
            
            if login_submitted:
                if login_username in usernames_passwords:
                    # Check password with bcrypt
                    if bcrypt.checkpw(login_password.encode('utf-8'), usernames_passwords[login_username].encode('utf-8')):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = login_username
                        st.success(f"Welcome back, {login_username}!")
                        st.rerun()
                    else:
                        st.error('❌ Username/password is incorrect')
                else:
                    st.error('❌ Username/password is incorrect')

        # Signup Form
        st.markdown("---")
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
                # The add_new_user function handles the username already taken check
                else:
                    if add_new_user(session, new_username, new_password):
                        st.rerun() # Rerun to update the user list for login

    session.close()

if __name__ == '__main__':
    run_app()
