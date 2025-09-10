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


def display_single_result(result, model_name):
    """Display a single model result in a clean format."""

    # Show overall prediction
    col1, col2, col3 = st.columns(3)

    with col1:
        # Determine if it's phishing based on class
        is_phishing = result.get("class") == 1
        status = "🚨 PHISHING" if is_phishing else "✅ BENIGN"
        st.markdown(f"### {status}")

    with col2:
        target = result.get("target", "Unknown")
        if target and target.lower() != "unknown":
            st.markdown(f"**Target Brand:** {target}")
        else:
            st.markdown("**Target Brand:** Not detected")

    with col3:
        # Show confidence/distance
        if "confidence" in result:
            confidence = result.get("confidence")
            if confidence is not None:
                st.markdown(f"**Confidence:** {confidence:.2%}")
            else:
                st.markdown("**Confidence:** Not available")
        elif "distance" in result:
            distance = result.get("distance")
            if distance is not None:
                st.markdown(f"**Distance:** {distance:.4f}")
            else:
                st.markdown("**Distance:** Not available")

    # Detailed results in expandable section
    with st.expander("View Detailed Results", expanded=False):
        # Create a clean DataFrame
        display_data = {}

        for key, value in result.items():
            # Format the key for better display
            formatted_key = key.replace("_", " ").title()

            # Format specific values
            if key == "class":
                display_data["Classification"] = "Phishing" if value == 1 else "Benign"
            elif key == "confidence" and value is not None:
                display_data[formatted_key] = f"{value:.2%}"
            elif key == "distance" and value is not None:
                display_data[formatted_key] = f"{value:.6f}"
            elif key == "confidence" and value is None:
                display_data[formatted_key] = "Not available"
            elif key == "distance" and value is None:
                display_data[formatted_key] = "Not available"
            elif value is not None:
                display_data[formatted_key] = str(value)

        # Display as a nice table
        df = pd.DataFrame([display_data]).T
        df.columns = ["Value"]
        st.dataframe(df, use_container_width=True)


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
                "Content-Type": "application/json",
            },
            json={"image": base64_image, "url": st.session_state.url_input},
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

    # Extract results from API response
    api_response = st.session_state.results

    # Show request ID if available
    if isinstance(api_response, dict) and "request_id" in api_response:
        st.caption(f"Request ID: `{api_response['request_id']}`")
        # Add a clickable link to view prediction details
        st.markdown(
            f"[View Prediction Details](http://localhost:8000/predictions/{api_response['request_id']})"
        )

    # Get the actual results array
    results_array = []
    if isinstance(api_response, dict) and "results" in api_response:
        results_array = api_response["results"]
    elif isinstance(api_response, list):
        results_array = api_response
    else:
        # Fallback - show raw response
        st.json(api_response)
        st.stop()

    # Check if we have results to display
    if not results_array:
        st.info("No results returned from the models.")
        st.stop()

    # Sort results to have a consistent order of tabs
    results_array.sort(key=lambda x: x.get("method", "unknown"))

    # Create tabs for different model results
    tab_names = [
        result.get("method", f"Model {i + 1}").replace("_", " ").title()
        for i, result in enumerate(results_array)
    ]

    if len(results_array) == 1:
        # Single result - no tabs needed
        result = results_array[0]
        model_name = result.get("method", "Model").replace("_", " ").title()
        display_single_result(result, model_name)
    else:
        # Multiple results - use tabs
        tabs = st.tabs(tab_names)

        for i, (tab, result) in enumerate(zip(tabs, results_array)):
            with tab:
                model_name = (
                    result.get("method", f"Model {i + 1}").replace("_", " ").title()
                )
                display_single_result(result, model_name)


# Simple footer
st.caption("Created with Streamlit")
