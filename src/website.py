import streamlit as st
import requests
import base64
import pandas as pd

# Initialize session state for processing state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "results" not in st.session_state:
    st.session_state.results = None

# Set up the page
st.set_page_config(page_title="Image Analysis Tool", layout="wide")


# Function to convert image to base64
def get_image_base64(file):
    bytes_data = file.getvalue()
    base64_str = base64.b64encode(bytes_data).decode("utf-8")
    return base64_str


# Function to analyze image
def analyze_image():
    st.session_state.processing = True
    st.session_state.results = None

    # Convert image to base64
    base64_image = get_image_base64(st.session_state.uploaded_file)

    try:
        # Send API request
        response = requests.post(
            "http://localhost:8000/predict",
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"image": base64_image, "url": st.session_state.url_input},
        )

        if response.status_code == 200:
            st.session_state.results = response.json()
        else:
            st.session_state.error = (
                f"Error: API returned status code {response.status_code}"
            )
            st.session_state.error_details = response.text
    except Exception as e:
        st.session_state.error = f"An error occurred: {e}"

    st.session_state.processing = False
    st.rerun()


# Header
st.title("Image Analysis Tool")

# Use columns for input controls at the top
col1, col2 = st.columns([2, 1])

with col1:
    # URL input
    st.session_state.url_input = st.text_input(
        "Target URL:", placeholder="https://example.com"
    )

with col2:
    # Upload Section
    st.session_state.uploaded_file = st.file_uploader(
        "Upload image:", type=["jpg", "jpeg", "png"]
    )

# Analyze button - centered and more prominent
if st.session_state.uploaded_file and st.session_state.url_input:
    if st.session_state.processing:
        st.button(
            "Processing...", disabled=True, use_container_width=True, type="primary"
        )
    else:
        st.button(
            "Analyze Image",
            on_click=analyze_image,
            use_container_width=True,
            type="primary",
        )
    can_analyze = True
else:
    missing = []
    if not st.session_state.get("uploaded_file"):
        missing.append("image")
    if not st.session_state.get("url_input"):
        missing.append("URL")

    st.info(f"Please provide {' and '.join(missing)} to analyze.")
    st.button("Analyze Image", disabled=True, use_container_width=True)
    can_analyze = False

# Display uploaded image
if st.session_state.get("uploaded_file"):
    st.image(st.session_state.uploaded_file, caption="Uploaded Image", width=600)

# Display error if any
if hasattr(st.session_state, "error"):
    st.error(st.session_state.error)
    if hasattr(st.session_state, "error_details"):
        with st.expander("Error Details"):
            st.code(st.session_state.error_details)

    # Clear errors on next run
    delattr(st.session_state, "error")
    if hasattr(st.session_state, "error_details"):
        delattr(st.session_state, "error_details")

# Process and display results
if st.session_state.results:
    # Results Section with cleaner layout
    st.write("## Analysis Results")

    # Handle the array of results from the API
    if isinstance(st.session_state.results, list):
        models = ["VisualPhish", "Phishpedia"]

        # Create tabs for different results
        tabs = st.tabs(
            [
                models[i] if i < len(models) else f"Model {i + 1}"
                for i in range(len(st.session_state.results))
            ]
        )

        for i, (tab, result) in enumerate(zip(tabs, st.session_state.results)):
            with tab:
                # Display key metrics in an easy-to-read format
                metric_cols = st.columns(3)

                with metric_cols[0]:
                    st.metric("Target", result.get("target", "Unknown"))

                with metric_cols[1]:
                    if "confidence" in result:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                    elif "distance" in result:
                        st.metric("Distance", f"{result.get('distance', 0):.2f}")

                with metric_cols[2]:
                    st.metric("Class", result.get("class", "N/A"))

                # Convert result to DataFrame for table display
                df = pd.DataFrame([result])

                # Rename columns for better display
                column_renames = {
                    "url": "URL",
                    "class": "Class",
                    "target": "Target",
                    "distance": "Distance",
                    "confidence": "Confidence",
                }
                df = df.rename(
                    columns={k: v for k, v in column_renames.items() if k in df.columns}
                )

                # Display result as table
                st.write("#### Detailed Results:")
                st.dataframe(df, use_container_width=True)

    elif isinstance(st.session_state.results, dict):
        # Display dictionary results as table
        df = pd.DataFrame([st.session_state.results])
        st.dataframe(df, use_container_width=True)
    else:
        # Fallback for other result types
        st.write(st.session_state.results)

# Simple footer
st.caption("Created with Streamlit")
