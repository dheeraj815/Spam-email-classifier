"""
AI-Powered Spam Message Detection System
========================================
Advanced machine learning application for detecting spam messages using
Natural Language Processing and Naive Bayes classification.

Features:
 - Real-time spam detection with confidence scoring
 - TF-IDF vectorization with custom text preprocessing
 - Interactive message analysis and visualization
 - Comprehensive performance metrics
 - Message history tracking and export
 - Professional UI with modern design

Created by: Dheeraj Muley
Year: 2025
"""

import streamlit as st
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ================================================
# 🎨 PAGE CONFIGURATION
# ================================================
st.set_page_config(
    page_title="🛡️ AI Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# 🎨 CUSTOM STYLING
# ================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        
        /* Main container */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Space Grotesk', sans-serif;
        }
        
        /* Hide defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Title styling */
        h1 {
            color: #ffffff;
            font-weight: 700;
            text-align: center;
            font-size: 3.5em !important;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            margin-bottom: 0px;
            letter-spacing: -1px;
        }
        
        h2, h3 {
            color: #ffffff;
            font-weight: 600;
        }
        
        /* Card containers */
        .detection-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 700;
            padding: 15px 30px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div.stButton > button:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        /* Text area */
        .stTextArea textarea {
            border-radius: 12px;
            border: 2px solid #667eea;
            font-size: 16px;
            padding: 15px;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2.2em;
            font-weight: 800;
            color: #667eea;
        }
        
        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #764ba2;
        }
        
        /* Result boxes */
        .spam-result {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
            margin: 20px 0;
        }
        
        .ham-result {
            background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(81, 207, 102, 0.4);
            margin: 20px 0;
        }
        
        .result-box h2 {
            font-size: 2.5em;
            margin: 10px 0;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Section headers */
        .section-header {
            color: #667eea;
            font-weight: 800;
            font-size: 1.8em;
            margin-bottom: 15px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        /* Caption text */
        .caption-text {
            text-align: center;
            color: #ffffff;
            font-size: 1.2em;
            font-weight: 600;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
            margin-top: -10px;
            margin-bottom: 30px;
        }
        
        /* Info boxes */
        .info-box {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: white;
            font-size: 1.1em;
            font-weight: 600;
            padding: 20px;
            margin-top: 40px;
            background: rgba(0,0,0,0.2);
            border-radius: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ================================================
# 🔧 NLTK SETUP
# ================================================


@st.cache_resource
def setup_nltk():
    """Initialize NLTK resources"""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    return ps, stop_words


ps, stop_words = setup_nltk()

# ================================================
# 📊 ENHANCED DATASET
# ================================================


@st.cache_data
def load_dataset():
    """Load and prepare spam detection dataset"""
    data = [
        # Spam messages (expanded dataset)
        ["spam", "Win a free iPhone now! Click here immediately"],
        ["spam", "Claim your lottery prize today! You won $10000"],
        ["spam", "Urgent account verification required immediately"],
        ["spam", "You have won 1000 dollars cash! Collect now"],
        ["spam", "Limited time offer click now! Don't miss out"],
        ["spam", "Get instant loan approval! No credit check needed"],
        ["spam", "Congratulations you are selected for special offer"],
        ["spam", "Free recharge available now! Limited slots"],
        ["spam", "Your package is waiting! Confirm delivery now"],
        ["spam", "Make money fast! Work from home opportunity"],
        ["spam", "Free gift card! Click to claim your prize"],
        ["spam", "WINNER! You've been selected for cash prize"],
        ["spam", "Exclusive deal just for you! Act now"],
        ["spam", "Your loan is approved! Get cash today"],
        ["spam", "Free vacation booking! All expenses paid"],
        ["spam", "Hot singles in your area! Meet now"],
        ["spam", "Congratulations! You qualify for special discount"],
        ["spam", "Urgent: Update your payment information now"],
        ["spam", "You're pre-approved for credit card! Apply now"],
        ["spam", "Final notice: Claim your refund immediately"],

        # Ham messages (legitimate messages)
        ["ham", "Are we still meeting today for the project discussion"],
        ["ham", "Please review the attached document when you get a chance"],
        ["ham", "Lunch at 1 pm at the usual place"],
        ["ham", "Project deadline is next week, need to finalize"],
        ["ham", "Can you send the files we discussed yesterday"],
        ["ham", "Let me know your availability for the meeting"],
        ["ham", "Thanks for your help with the presentation"],
        ["ham", "Call me when you are free to discuss"],
        ["ham", "Meeting rescheduled to tomorrow at 3pm"],
        ["ham", "Great job on the presentation today"],
        ["ham", "Can you pick up groceries on your way home"],
        ["ham", "Mom's birthday is next Sunday, don't forget"],
        ["ham", "The report looks good, just minor edits needed"],
        ["ham", "Traffic is bad, will be 15 minutes late"],
        ["ham", "Happy birthday! Hope you have a wonderful day"],
        ["ham", "Reminder about dentist appointment tomorrow"],
        ["ham", "Please confirm receipt of the package"],
        ["ham", "Team lunch is scheduled for Friday"],
        ["ham", "Your order has been shipped and will arrive Monday"],
        ["ham", "Monthly team meeting agenda is ready for review"]
    ]

    df = pd.DataFrame(data, columns=["label", "message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


df = load_dataset()

# ================================================
# 🧹 TEXT PREPROCESSING
# ================================================


def clean_text(text):
    """Advanced text cleaning and preprocessing"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # Tokenize
    words = text.split()

    # Remove stopwords and apply stemming
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# ================================================
# 🤖 MODEL TRAINING
# ================================================


@st.cache_resource
def train_spam_model():
    """Train the spam detection model with comprehensive metrics"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label_num"],
        test_size=0.25,
        random_state=42,
        stratify=df["label_num"]
    )

    # Create pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            max_features=1000,
            ngram_range=(1, 2)
        )),
        ("nb", MultinomialNB(alpha=0.1))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Get feature importance (top words)
    vectorizer = model.named_steps['tfidf']
    nb_model = model.named_steps['nb']

    feature_names = vectorizer.get_feature_names_out()
    spam_log_probs = nb_model.feature_log_prob_[1]

    # Get top spam indicators
    top_spam_indices = spam_log_probs.argsort()[-10:][::-1]
    top_spam_words = [(feature_names[i], spam_log_probs[i])
                      for i in top_spam_indices]

    return model, accuracy, conf_matrix, class_report, top_spam_words, X_test, y_test


model, accuracy, conf_matrix, class_report, top_spam_words, X_test, y_test = train_spam_model()

# ================================================
# 💾 SESSION STATE FOR HISTORY
# ================================================
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# ================================================
# 🎯 MAIN APPLICATION
# ================================================

# Header
st.markdown("<h1>🛡️ AI SPAM DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='caption-text'>Advanced Machine Learning-Based Spam Message Detection | Created by <b>Dheeraj Muley</b> — 2025</p>",
            unsafe_allow_html=True)

# ================================================
# 📱 MAIN DETECTION INTERFACE
# ================================================
st.markdown("---")
st.markdown("<div class='detection-card'>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>📧 Message Analysis Tool</p>",
            unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Enter message to analyze:",
        height=150,
        placeholder="Type or paste your message here...",
        help="Enter any text message to check if it's spam or legitimate"
    )

with col2:
    st.markdown("### Quick Examples")
    if st.button("📩 Spam Example", use_container_width=True):
        user_input = "Congratulations! You won $1000. Click here to claim now!"
        st.rerun()

    if st.button("✅ Ham Example", use_container_width=True):
        user_input = "Can we reschedule our meeting to tomorrow at 3pm?"
        st.rerun()

    if st.button("🔄 Clear", use_container_width=True):
        user_input = ""
        st.rerun()

# Detection button
detect_button = st.button("🔍 ANALYZE MESSAGE", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================================================
# 🎯 DETECTION RESULTS
# ================================================
if detect_button:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Get prediction
        prediction = model.predict([user_input])[0]
        probability = model.predict_proba([user_input])[0]
        confidence = max(probability) * 100

        # Determine result
        is_spam = prediction == 1
        spam_prob = probability[1] * 100
        ham_prob = probability[0] * 100

        # Display result
        st.markdown("---")

        if is_spam:
            st.markdown(f"""
                <div class='spam-result'>
                    <h2>🚫 SPAM DETECTED</h2>
                    <p style='font-size: 1.3em; margin: 10px 0;'>
                        This message appears to be spam!
                    </p>
                    <p style='font-size: 1.1em; opacity: 0.9;'>
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='ham-result'>
                    <h2>✅ LEGITIMATE MESSAGE</h2>
                    <p style='font-size: 1.3em; margin: 10px 0;'>
                        This message appears to be legitimate!
                    </p>
                    <p style='font-size: 1.1em; opacity: 0.9;'>
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Detailed analysis
        st.markdown("<div class='detection-card'>", unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>📊 Detailed Analysis</p>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Spam Probability", f"{spam_prob:.1f}%")
        with col2:
            st.metric("Ham Probability", f"{ham_prob:.1f}%")
        with col3:
            st.metric("Confidence Level", f"{confidence:.1f}%")
        with col4:
            word_count = len(user_input.split())
            st.metric("Word Count", word_count)

        # Probability chart
        st.markdown("### 📈 Probability Distribution")

        fig = go.Figure(data=[
            go.Bar(
                x=['Spam', 'Ham'],
                y=[spam_prob, ham_prob],
                marker_color=['#ff6b6b', '#51cf66'],
                text=[f'{spam_prob:.1f}%', f'{ham_prob:.1f}%'],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Classification Probabilities',
            yaxis_title='Probability (%)',
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#667eea')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Text analysis
        st.markdown("### 🔍 Message Breakdown")
        cleaned = clean_text(user_input)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Message:**")
            st.info(user_input)
        with col2:
            st.markdown("**Processed Text:**")
            st.info(
                cleaned if cleaned else "(All words removed during preprocessing)")

        st.markdown("</div>", unsafe_allow_html=True)

        # Save to history
        st.session_state.detection_history.append({
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Message': user_input[:50] + "..." if len(user_input) > 50 else user_input,
            'Prediction': 'SPAM' if is_spam else 'HAM',
            'Confidence': f"{confidence:.1f}%",
            'Spam_Prob': f"{spam_prob:.1f}%",
            'Ham_Prob': f"{ham_prob:.1f}%"
        })

# ================================================
# 📜 DETECTION HISTORY
# ================================================
if st.session_state.detection_history:
    st.markdown("---")
    st.markdown("<div class='detection-card'>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>📜 Detection History</p>",
                unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.detection_history)

    # Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Checks", len(history_df))
    with col2:
        spam_count = len(history_df[history_df['Prediction'] == 'SPAM'])
        st.metric("Spam Detected", spam_count)
    with col3:
        ham_count = len(history_df[history_df['Prediction'] == 'HAM'])
        st.metric("Ham Detected", ham_count)

    # Display history
    st.dataframe(history_df.tail(
        10), use_container_width=True, hide_index=True)

    # Download button
    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download History (CSV)",
        data=csv_data,
        file_name=f'spam_detection_history_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
        use_container_width=True
    )

    # Clear history button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.detection_history = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ================================================
# 📊 MODEL PERFORMANCE & INSIGHTS
# ================================================
st.markdown("---")
st.markdown("<div class='detection-card'>", unsafe_allow_html=True)
st.markdown("<p class='section-header'>📊 Model Performance & Insights</p>",
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(
    ["📈 Performance Metrics", "🎯 Confusion Matrix", "🔤 Top Spam Indicators"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Accuracy", f"{accuracy*100:.1f}%")
    with col2:
        st.metric("Spam Precision",
                  f"{class_report['1']['precision']*100:.1f}%")
    with col3:
        st.metric("Spam Recall", f"{class_report['1']['recall']*100:.1f}%")
    with col4:
        st.metric("F1-Score", f"{class_report['1']['f1-score']*100:.1f}%")

    st.markdown("---")

    # Performance breakdown
    metrics_df = pd.DataFrame({
        'Class': ['Ham (Legitimate)', 'Spam'],
        'Precision': [class_report['0']['precision'], class_report['1']['precision']],
        'Recall': [class_report['0']['recall'], class_report['1']['recall']],
        'F1-Score': [class_report['0']['f1-score'], class_report['1']['f1-score']],
        'Support': [class_report['0']['support'], class_report['1']['support']]
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

with tab2:
    # Confusion matrix visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'},
                ax=ax)

    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("""
        <div class='info-box'>
            <b>Understanding the Confusion Matrix:</b><br>
            • <b>True Negatives (Top-Left):</b> Correctly identified Ham messages<br>
            • <b>False Positives (Top-Right):</b> Ham messages incorrectly marked as Spam<br>
            • <b>False Negatives (Bottom-Left):</b> Spam messages incorrectly marked as Ham<br>
            • <b>True Positives (Bottom-Right):</b> Correctly identified Spam messages
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🚨 Top Words Associated with Spam")

    # Create dataframe of top spam words
    spam_words_df = pd.DataFrame(top_spam_words, columns=[
                                 'Word', 'Log Probability'])
    spam_words_df['Importance'] = spam_words_df['Log Probability'].apply(
        lambda x: abs(x))

    # Bar chart
    fig = px.bar(spam_words_df,
                 x='Importance',
                 y='Word',
                 orientation='h',
                 title='Most Indicative Spam Words',
                 color='Importance',
                 color_continuous_scale='Reds')

    fig.update_layout(
        showlegend=False,
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Tip:** Messages containing these words are more likely to be classified as spam.")

st.markdown("</div>", unsafe_allow_html=True)

# ================================================
# 📱 SIDEBAR
# ================================================
st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.markdown("---")

st.sidebar.metric("Algorithm", "Naive Bayes")
st.sidebar.metric("Vectorization", "TF-IDF")
st.sidebar.metric("Accuracy", f"{accuracy*100:.1f}%")
st.sidebar.metric("Training Samples", len(df))

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Distribution")

spam_count = len(df[df['label'] == 'spam'])
ham_count = len(df[df['label'] == 'ham'])

fig_dist = go.Figure(data=[go.Pie(
    labels=['Spam', 'Ham'],
    values=[spam_count, ham_count],
    marker_colors=['#ff6b6b', '#51cf66'],
    hole=0.4
)])

fig_dist.update_layout(
    showlegend=True,
    height=300,
    margin=dict(l=20, r=20, t=20, b=20)
)

st.sidebar.plotly_chart(fig_dist, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
This AI-powered spam detector uses:
- **Natural Language Processing**
- **TF-IDF Vectorization**
- **Multinomial Naive Bayes**
- **Advanced Text Preprocessing**

The model analyzes message patterns to accurately classify spam vs legitimate messages.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛡️ Features")
st.sidebar.markdown("""
✅ Real-time detection  
✅ Confidence scoring  
✅ Message history  
✅ Performance analytics  
✅ Export capabilities  
""")

# ================================================
# 📱 FOOTER
# ================================================
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <h3 style='margin: 0; color: white;'>🛡️ AI-Powered Spam Detection System</h3>
        <p style='margin: 10px 0 0 0; font-size: 1.1em;'>
            Powered by Machine Learning & NLP | Created with ❤️ by <b>Dheeraj Muley</b> | © 2025
        </p>
        <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;'>
            Built with Streamlit, Scikit-Learn & NLTK
        </p>
    </div>
""", unsafe_allow_html=True)
