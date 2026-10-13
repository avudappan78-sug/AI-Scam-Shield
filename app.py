import re
import sqlite3
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

APP_DIR = Path(__file__).resolve().parent
DATASET_PATH = APP_DIR / "SMSSpamCollection"
DB_PATH = APP_DIR / "scam_history.db"

st.set_page_config(
    page_title="AI Scam Shield",
    page_icon="🛡️",
    layout="wide"
)


# =====================================================
# 2. PROFESSIONAL UI
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: #ffffff;
}

h1, h2, h3, h4, p, label {
    color: #ffffff !important;
}

[data-testid="stMetric"] {
    background-color: #151e33;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #263451;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: bold;
}

[data-testid="stTextArea"] textarea {
    background-color: #151e33;
    color: white;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #263451;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. DATABASE
# =====================================================

def get_connection():
    return sqlite3.connect(DB_PATH, timeout=10)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                matched_phrases TEXT,
                url_warnings TEXT
            )
        """)


def save_history(
    message,
    prediction,
    confidence,
    risk_level,
    matched_phrases,
    url_warnings
):
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO history (
                timestamp,
                message,
                prediction,
                confidence,
                risk_level,
                matched_phrases,
                url_warnings
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            message,
            prediction,
            confidence,
            risk_level,
            ", ".join(matched_phrases),
            " | ".join(url_warnings)
        ))


def read_history():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT
                id,
                timestamp,
                message,
                prediction,
                confidence,
                risk_level,
                matched_phrases,
                url_warnings
            FROM history
            ORDER BY id DESC
        """, conn)


initialize_database()


# =====================================================
# 4. DATASET AND MACHINE LEARNING
# =====================================================

@st.cache_resource
def load_model(dataset_path, modified_time):
    data = pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        encoding_errors="replace"
    )

    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["message"] = (
        data["message"]
        .fillna("")
        .astype(str)
    )

    data = data[
        data["label"].isin(["ham", "spam"])
        & data["message"].str.strip().ne("")
    ].copy()

    if data.empty:
        raise ValueError("No valid SMS records found.")

    if data["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain ham and spam labels."
        )

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    def create_pipeline():
        return Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ])

    # Evaluate on a held-out test set
    evaluation_model = create_pipeline()
    evaluation_model.fit(X_train, y_train)

    test_predictions = evaluation_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test, test_predictions
        ),
        "precision": precision_score(
            y_test,
            test_predictions,
            pos_label="spam",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            test_predictions,
            pos_label="spam",
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            test_predictions,
            pos_label="spam",
            zero_division=0
        ),
        "test_count": len(y_test)
    }

    # Train the final prediction model on the full dataset
    final_model = create_pipeline()
    final_model.fit(X, y)

    return final_model, len(data), metrics


# =====================================================
# 5. LOAD MODEL SAFELY
# =====================================================

if not DATASET_PATH.is_file():
    st.error(
        "Dataset not found. Keep SMSSpamCollection "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model, dataset_count, metrics = load_model(
        str(DATASET_PATH),
        DATASET_PATH.stat().st_mtime
    )

except Exception as error:
    st.error("Unable to load the dataset or train the model.")
    st.code(f"{type(error)._name_}: {error}")
    st.info(
        "Check the dataset file and ensure it contains "
        "tab-separated ham/spam labels and SMS messages."
    )
    st.stop()


# =====================================================
# 6. SCAM PHRASES AND URL CHECKER
# =====================================================

SCAM_PHRASES = [
    "won a prize",
    "claim your reward",
    "click this link",
    "verify your account",
    "send money",
    "share your otp",
    "bank details",
    "urgent payment",
    "lottery",
    "account suspended",
    "act immediately",
    "you have won",
    "limited time offer",
    "confirm your password",
    "claim your gift",
    "update your bank details"
]

SUSPICIOUS_URL_WORDS = [
    "login",
    "verify",
    "claim",
    "prize",
    "bonus",
    "free",
    "gift"
]

URL_PATTERN = re.compile(
    r"""https?://[^\s<>"']+|www\.[^\s<>"']+""",
    re.IGNORECASE
)


def find_urls(message):
    return URL_PATTERN.findall(message)


def inspect_url(url):
    """Basic URL checks; these do not verify site reputation."""

    url = url.rstrip(".,!?;:)")

    candidate = url

    if candidate.lower().startswith("www."):
        candidate = "https://" + candidate

    reasons = []

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname or ""
    except ValueError:
        return ["URL format appears invalid."]

    if not hostname or "." not in hostname:
        reasons.append("The hostname appears unusual.")

    if parsed.scheme.lower() == "http":
        reasons.append("The URL uses unencrypted HTTP.")

    if "@" in parsed.netloc:
        reasons.append(
            "The URL contains an @ symbol in its address."
        )

    if any(
        word in url.lower()
        for word in SUSPICIOUS_URL_WORDS
    ):
        reasons.append(
            "The URL contains a potentially suspicious keyword."
        )

    return reasons


# =====================================================
# 7. HEADER
# =====================================================

st.title("🛡️ AI Scam Shield")
st.subheader("AI-Powered SMS Scam Detection System")

st.write(
    "Analyze SMS messages using Machine Learning, "
    "warning phrases and basic URL indicators."
)

st.caption(
    "Educational prototype. Results can be incorrect."
)

st.divider()


# =====================================================
# 8. READ STORED HISTORY
# =====================================================

history_df = read_history()

total_checked = len(history_df)

scam_count = int(
    (history_df["risk_level"] != "Low").sum()
) if not history_df.empty else 0

safe_count = total_checked - scam_count


# =====================================================
# 9. DASHBOARD
# =====================================================

st.header("📊 Detection Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Dataset Messages", f"{dataset_count:,}")
col2.metric("Messages Checked", total_checked)
col3.metric("Suspicious Results", scam_count)
col4.metric("Low-Risk Results", safe_count)

st.caption(
    "History and dashboard counts are stored locally "
    "in scam_history.db."
)

st.divider()


# =====================================================
# 10. MODEL PERFORMANCE
# =====================================================

st.header("📈 Machine Learning Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    f"{metrics['accuracy'] * 100:.1f}%"
)

m2.metric(
    "Precision",
    f"{metrics['precision'] * 100:.1f}%"
)

m3.metric(
    "Recall",
    f"{metrics['recall'] * 100:.1f}%"
)

m4.metric(
    "F1-Score",
    f"{metrics['f1'] * 100:.1f}%"
)

st.caption(
    f"Metrics are calculated on a held-out test set of "
    f"{metrics['test_count']:,} messages from the dataset. "
    "They do not guarantee the same performance on real-world scams."
)

st.divider()


# =====================================================
# 11. ANALYTICS CHART
# =====================================================

st.header("📊 SMS Analytics")

if history_df.empty:
    st.info(
        "No messages analyzed yet. Analyze a message "
        "to start building your chart."
    )

else:
    chart_df = (
        history_df["risk_level"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )

    st.bar_chart(
        chart_df,
        x_label="Risk Level",
        y_label="Number of Messages"
    )


st.divider()


# =====================================================
# 12. SMS DETECTOR
# =====================================================

st.header("🔍 SMS Scam Detector")

message = st.text_area(
    "📩 Paste your SMS message here",
    placeholder=(
        "Example: Congratulations! You won a prize. "
        "Click this link to claim your reward."
    ),
    height=150,
    max_chars=10000
)

analyze = st.button(
    "🔍 Analyze Message",
    use_container_width=True
)


# =====================================================
# 13. ANALYSIS
# =====================================================

if analyze:

    if not message.strip():
        st.warning("Please enter an SMS message first.")

    else:
        try:
            prediction = model.predict([message])[0]

            probabilities = model.predict_proba([message])[0]
            classes = list(model.classes_)

            confidence = float(
                probabilities[classes.index(prediction)]
                * 100
            )

            normalized_message = message.casefold()

            matched_phrases = [
                phrase
                for phrase in SCAM_PHRASES
                if phrase.casefold() in normalized_message
            ]

            url_results = []

            for url in find_urls(message):
                reasons = inspect_url(url)

                if reasons:
                    url_results.append((url, reasons))

            # Determine risk level using configured indicators
            indicator_count = (
                int(prediction == "spam")
                + int(bool(matched_phrases))
                + int(bool(url_results))
            )

            if indicator_count >= 2:
                risk_level = "High"

            elif indicator_count == 1:
                risk_level = "Medium"

            else:
                risk_level = "Low"

            # Save the result
            all_url_warnings = [
                f"{url}: {', '.join(reasons)}"
                for url, reasons in url_results
            ]

            save_history(
                message=message,
                prediction=prediction,
                confidence=confidence,
                risk_level=risk_level,
                matched_phrases=matched_phrases,
                url_warnings=all_url_warnings
            )

            st.divider()
            st.header("📋 Analysis Result")

            result_col, confidence_col = st.columns(2)

            with result_col:
                if risk_level == "High":
                    st.error("🚨 HIGH RISK: Suspicious Message")

                elif risk_level == "Medium":
                    st.warning("⚠️ MEDIUM RISK: Review Carefully")

                else:
                    st.success("✅ LOW RISK: No Configured Indicators")

            with confidence_col:
                st.metric(
                    "Model Prediction Confidence",
                    f"{confidence:.1f}%"
                )

            st.write(
                "*Machine Learning Prediction:* "
                + (
                    "Scam-like (Spam)"
                    if prediction == "spam"
                    else "Normal-like (Ham)"
                )
            )

            st.write(f"*Risk Level:* {risk_level}")

            st.caption(
                "Model confidence is not the probability that "
                "a message is genuinely fraudulent."
            )

            if matched_phrases:
                st.subheader("⚠️ Warning Phrases Found")

                for phrase in matched_phrases:
                    st.write(f"- {phrase}")

            if url_results:
                st.subheader("🔗 URL Warning Check")

                for url, reasons in url_results:
                    st.write(f"*URL:* {url}")

                    for reason in reasons:
                        st.warning(reason)

            urls_found = find_urls(message)

            if urls_found and not url_results:
                st.info(
                    "No configured URL warning indicators were "
                    "found. The URLs have not been verified as safe."
                )

            if not matched_phrases and not url_results:
                st.write(
                    "No configured warning phrases or URL "
                    "indicators were found."
                )

            st.info(
                "🛡️ Safety Tip: Never share your OTP, password "
                "or bank details with unknown people. Verify "
                "messages through official channels."
            )

            st.caption(
                "Risk levels are heuristic indicators, not a "
                "guarantee. Some scams may be missed, and normal "
                "messages may be flagged."
            )

            # Refresh the page data after saving
            st.rerun()

        except Exception as error:
            st.error("Unable to analyze this message.")
            st.code(f"{type(error)._name_}: {error}")


# =====================================================
# 14. SMS HISTORY
# =====================================================

st.divider()
st.header("📱 SMS Analysis History")

history_df = read_history()

if history_df.empty:
    st.info("Your analyzed messages will appear here.")

else:
    st.dataframe(
        history_df[
            [
                "timestamp",
                "message",
                "prediction",
                "confidence",
                "risk_level"
            ]
        ].head(20),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing the latest 20 records. The database stores "
        "the analysis history locally."
    )

    # Download report
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Full SMS History (CSV)",
        data=csv_data,
        file_name="ai_scam_shield_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# =====================================================
# 15. SAFETY GUIDE
# =====================================================

st.divider()
st.header("🛡️ Scam Prevention Guide")

with st.expander("How can I stay safe from SMS scams?"):

    st.markdown("""
    - Never share your OTP or banking password.
    - Do not trust unexpected prize or lottery messages.
    - Avoid clicking unknown links.
    - Verify bank messages using official contact details.
    - Never transfer money because of threats or urgency.
    - Report suspicious messages through appropriate official channels.
    """)


# =====================================================
# 16. FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Scam Shield | SMS Machine Learning + Analytics | "
    "Educational Project 🛡️"
)

st.caption(
    "Privacy note: Analyzed messages are stored in the "
    "local scam_history.db file. Use demo messages during "
    "presentations and avoid entering private information."
)