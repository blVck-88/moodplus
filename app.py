import streamlit as st
from datetime import date
import pandas as pd
import bcrypt
import plotly.express as px

from analysis import (
    analyze_sentiment,
    classify_stress_level,
    generate_suggestions_with_model,
)

# --- Import Database Logic ---
from database import User, JournalEntry, get_session


# -------------------------------------------------------------------
# PASSWORD & USER MANAGEMENT
# -------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hashes the password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def get_all_users(session) -> dict:
    """
    Retrieves all usernames and their hashed passwords from the database.
    Returns dict: {username: hashed_password}
    """
    users = session.query(User).all()
    return {user.username: user.hashed_password for user in users}


def add_new_user(session, username: str, password: str) -> bool:
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
        st.error(f"Error creating user: Username '{username}' may already exist. ({e})")
        return False


# -------------------------------------------------------------------
# JOURNAL ENTRY PERSISTENCE
# -------------------------------------------------------------------

def save_journal_entry(
    session,
    user_id: int,
    entry_date,
    raw_text: str,
    mood_rating: int,
    compound_score: float,
    stress_level: str,
) -> bool:
    """Saves a new JournalEntry object to the database."""
    try:
        new_entry = JournalEntry(
            user_id=user_id,
            entry_date=entry_date,
            raw_text=raw_text,
            mood_rating=mood_rating,
            vader_compound_score=compound_score,
            ml_stress_level=stress_level,
        )
        session.add(new_entry)
        session.commit()
        st.success(f"Entry saved successfully for {entry_date}!")
        return True
    except Exception as e:
        session.rollback()
        st.error(
            f"Error saving entry. Did you already log an entry for this date? ({e})"
        )
        return False


# -------------------------------------------------------------------
# DATA EXTRACTION FOR DASHBOARD
# -------------------------------------------------------------------

def get_user_entries_df(session, user_id: int) -> pd.DataFrame:
    """
    Retrieves all journal entries for a user and returns them as a Pandas DataFrame.
    Columns: Date, Mood Rating (1-10), VADER Sentiment Score, Stress Level
    """
    entries = (
        session.query(JournalEntry)
        .filter_by(user_id=user_id)
        .order_by(JournalEntry.entry_date)
        .all()
    )

    data = []
    for entry in entries:
        data.append(
            {
                "Date": entry.entry_date,
                "Mood Rating (1-10)": entry.mood_rating,
                "VADER Sentiment Score": entry.vader_compound_score,
                "Stress Level": entry.ml_stress_level,
            }
        )

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.set_index("Date")
        df.sort_index(inplace=True)

    return df


# -------------------------------------------------------------------
# DASHBOARD RENDERING
# -------------------------------------------------------------------

def display_dashboard(df: pd.DataFrame) -> None:
    """Creates and displays interactive Plotly charts."""
    if df.empty:
        st.info("Log your first journal entry to see your dashboard!")
        return

    st.subheader("Mood and Sentiment Trends Over Time")

    # 1. Line Chart for Mood Rating and VADER Score
    plot_df = df[["Mood Rating (1-10)", "VADER Sentiment Score"]]

    fig_line = px.line(
        plot_df,
        y=plot_df.columns,
        title="Daily Mood vs. Sentiment Score",
        height=400,
    )
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Score")
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. Bar Chart for Stress Level Distribution
    stress_counts = df["Stress Level"].value_counts().reset_index()
    stress_counts.columns = ["Stress Level", "Count"]

    stress_order = ["Low", "Medium", "High"]
    stress_counts["Stress Level"] = pd.Categorical(
        stress_counts["Stress Level"], categories=stress_order, ordered=True
    )
    stress_counts = stress_counts.sort_values("Stress Level")

    color_map = {"Low": "lightgreen", "Medium": "gold", "High": "salmon"}

    fig_bar = px.bar(
        stress_counts,
        x="Stress Level",
        y="Count",
        color="Stress Level",
        color_discrete_map=color_map,
        title="Distribution of Calculated Stress Levels",
        height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

def run_app():
    st.set_page_config(page_title="Wellness Companion", layout="wide")
    st.title("🧠 Wellness Companion")

    # --- SESSION STATE INITIALIZATION ---
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None

    # Use try/finally to ensure DB session is always closed
    session = get_session()
    try:
        # Fetch all users once per run
        usernames_passwords = get_all_users(session)

        # ---------------------------
        # AUTHENTICATED VIEW
        # ---------------------------
        if st.session_state["authenticated"]:
            username = st.session_state["username"]

            col1, col2 = st.columns([4, 1])
            with col1:
                st.header(f"Hello, {username}! 👋")
            with col2:
                if st.button("Logout", key="logout_btn"):
                    st.session_state["authenticated"] = False
                    st.session_state["username"] = None
                    st.success("You have been logged out.")
                    st.rerun()

            current_user_db = session.query(User).filter_by(username=username).first()
            if current_user_db is None:
                st.error("User not found in database. Please log in again.")
                st.session_state["authenticated"] = False
                st.session_state["username"] = None
                st.rerun()
                return

            current_user_id = current_user_db.id

            st.markdown("---")
            input_col, dashboard_col = st.columns([1, 2], gap="large")

            # ---------------------------
            # JOURNAL FORM
            # ---------------------------
            with input_col:
                st.subheader("✍️ Log Your Day")
                with st.form("daily_entry_form", clear_on_submit=True):
                    entry_date = st.date_input("Date of Entry", date.today())

                    mood_rating = st.slider(
                        "Self-Reported Mood (1=Awful, 10=Fantastic)",
                        min_value=1,
                        max_value=10,
                        value=5,
                        step=1,
                    )

                    journal_text = st.text_area(
                        "Journal Entry (Write about your day, feelings, and events):",
                        height=200,
                    )

                    submitted = st.form_submit_button("Analyze & Save Entry")

                    if submitted:
                        if not journal_text.strip():
                            st.warning(
                                "Please write something in your journal entry before saving."
                            )
                        else:
                            # 1. Sentiment Analysis
                            compound_score = analyze_sentiment(journal_text)

                            # 2. Stress Classification
                            stress_level = classify_stress_level(
                                compound_score, mood_rating
                            )

                            # 3. Historical Data BEFORE saving this entry
                            history_df_before_save = get_user_entries_df(
                                session, current_user_id
                            )

                            # 4. AI Suggestions
                            suggestions = generate_suggestions_with_model(
                                compound_score,
                                stress_level,
                                journal_text,
                                history_df_before_save,
                            )

                            # 5. Save Entry
                            save_success = save_journal_entry(
                                session=session,
                                user_id=current_user_id,
                                entry_date=entry_date,
                                raw_text=journal_text,
                                mood_rating=mood_rating,
                                compound_score=compound_score,
                                stress_level=stress_level,
                            )

                            if save_success:
                                st.info(
                                    f"VADER Score: **{compound_score:.3f}** | Stress Level: **{stress_level}**"
                                )
                                st.subheader("🎯 Actionable Wellness Suggestions:")
                                for suggestion in suggestions:
                                    st.markdown(suggestion)

            # ---------------------------
            # DASHBOARD
            # ---------------------------
            with dashboard_col:
                st.header("📊 Your Wellness Dashboard")
                history_df = get_user_entries_df(session, current_user_id)
                display_dashboard(history_df)

        # ---------------------------
        # LOGIN / SIGNUP VIEW
        # ---------------------------
        else:
            st.subheader("Login to Your Account")

            # Login form
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login")

                if login_submitted:
                    if login_username in usernames_passwords:
                        stored_hash = usernames_passwords[login_username]
                        if bcrypt.checkpw(
                            login_password.encode("utf-8"),
                            stored_hash.encode("utf-8"),
                        ):
                            st.session_state["authenticated"] = True
                            st.session_state["username"] = login_username
                            st.success(f"Welcome back, {login_username}!")
                            st.rerun()
                        else:
                            st.error("❌ Username/password is incorrect.")
                    else:
                        st.error("❌ Username/password is incorrect.")

            # Signup form
            st.markdown("---")
            st.subheader("New User? Sign Up!")
            with st.form("signup_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                new_password_confirm = st.text_input(
                    "Confirm Password", type="password"
                )
                signup_submitted = st.form_submit_button("Create Account")

                if signup_submitted:
                    if not new_username or not new_password:
                        st.error("Username and Password are required.")
                    elif new_password != new_password_confirm:
                        st.error("Passwords do not match.")
                    else:
                        if add_new_user(session, new_username, new_password):
                            st.rerun()

    finally:
        # Ensure DB session is always closed
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_app()
