import streamlit as st
import logging
from components.logistic_reg import LogisticRegressionModel
from components.xgboost import XGBoostModel
import xgboost

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize models
lr_model = LogisticRegressionModel()
xgb_model = XGBoostModel()

# Model to instance mapping
MODELS = {
    "Model_A": lr_model,
    "Model_B": xgb_model
}

def load_model(model_name, version):
    model_instance = MODELS.get(model_name)
    if model_instance is None:
        st.error(f"Model {model_name} not found.")
        logging.error(f"Model {model_name} not found.")
        return None

    try:
        model_instance.load_model(version=f"v{version}")
        logging.info(f"Loaded {model_name} version {version} successfully.")
        return model_instance
    except Exception as e:
        st.error(f"Error loading model {model_name} version {version}: {e}")
        logging.error(f"Error loading model {model_name} version {version}: {e}")
        return None

# Initialize session state for storing dummy comments and model
if 'dummy_comments' not in st.session_state:
    st.session_state.dummy_comments = [
        {"user": "User1", "comment": "This is a sample comment with some toxic content!", "profile": "👤"},
        {"user": "User2", "comment": "Great post! I completely agree with you.", "profile": "👤"}
    ]

if 'model' not in st.session_state:
    st.session_state.model = load_model("Model_A", "1")  # Load default model and version
    st.session_state.model_name = "Model_A"
    st.session_state.model_version = "1"

# Top bar with name and version
st.sidebar.title('Comment Toxicity Detection')
st.sidebar.text('Model Version: 1.0')

# Model selection with default values
st.sidebar.header('Select Model')
model_name = st.sidebar.selectbox('Choose a model:', ['Model_A', 'Model_B'], index=0)
model_version = st.sidebar.selectbox('Choose a version:', ['1'], index=0)

# Load the selected model if it has changed
if st.session_state.model_name != model_name or st.session_state.model_version != model_version:
    st.session_state.model_name = model_name
    st.session_state.model_version = model_version
    st.session_state.model = load_model(model_name, model_version)
    logging.info(f"Model updated: {st.session_state.model_name}, version: {st.session_state.model_version}")

# Log model instance
logging.info(f"Current model: {st.session_state.model}, version: {st.session_state.model_version}")

# Custom CSS for horizontal radio buttons
st.markdown(
    """
    <style>
    .horizontal-radio .stRadio > div {
        display: flex;
        flex-direction: row;
    }
    .horizontal-radio .stRadio > div > label {
        margin-right: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Section for toggling between "Raw Model" and "Practical"
st.header('Comment Toxicity Detection')

# Use custom CSS class for horizontal layout
section = st.radio(
    "Select an option",
    ["Raw Model", "Practical"],
    index=0,
    key="horizontal-radio"
)

def predict_toxicity(comment, model):
    try:
        toxicity_scores = model.prediction(comment)[0]
        logging.info(f"Predicted toxicity scores: {toxicity_scores}")
        return {
            "Toxic": toxicity_scores[0],
            "Severe Toxic": toxicity_scores[1],
            "Obscene": toxicity_scores[2],
            "Threat": toxicity_scores[3],
            "Insult": toxicity_scores[4]
        }
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        logging.error(f"Error during prediction: {e}")
        return None

if section == "Raw Model":
    st.subheader('Comments Heading')
    comment = st.text_area('Enter your comment here:')

    if st.button('Show Prediction'):
        if comment and st.session_state.model:
            # Log the comment and model details
            logging.info(f"Comment: {comment}")
            logging.info(f"Model instance: {st.session_state.model}")

            # Get prediction from the loaded model
            toxicity_scores = predict_toxicity(comment, st.session_state.model)
            if toxicity_scores:
                st.write('Prediction Results:')
                st.write(toxicity_scores)
        else:
            st.write('Please enter a comment and select a model to analyze.')

elif section == "Practical":
    st.subheader('Comment Section')

    # Display dummy comments
    for entry in st.session_state.dummy_comments:
        st.write(f"{entry['profile']} **{entry['user']}**: {entry['comment']}")

    # Input field for new comment and send button
    new_comment = st.text_area('Enter your comment to post:')
    if st.button('Post Comment'):
        if new_comment and st.session_state.model:
            # Append new comment to the dummy comments list
            toxicity_scores = predict_toxicity(new_comment, st.session_state.model)
            isValidComment = True
            if toxicity_scores:
                for key, value in toxicity_scores.items():
                    if value > 0.5:
                        isValidComment = False
                        break
            st.session_state.dummy_comments.append({
                "user": "You",
                "comment": new_comment if isValidComment else "This comment has been removed due to toxicity.",
                "profile": "😎" if isValidComment else "🛑"
            })
            # Clear the input field
            st.session_state.new_comment = ""
            # Redraw the comments section
            st.rerun()
        else:
            st.write('Please enter a comment to post.')
