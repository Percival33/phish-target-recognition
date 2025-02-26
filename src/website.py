import streamlit as st

# Set up the page
st.set_page_config(page_title="Image Analysis Tool", layout="wide")

# Header
st.title("Image Analysis Tool")
st.subheader("Upload an image to analyze it with different methods.")

# Upload Section
uploaded_file = st.file_uploader(
    "Upload an image for analysis", type=["jpg", "jpeg", "png"]
)

# Placeholder for the results
if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=720,
    )

    # Simulate analysis
    method_1_result = "Result for Method 1: 85%"
    method_2_result = "Result for Method 2: 92%"
    method_3_result = "Result for Method 3: 78%"

    # Results Section
    st.write("### Analysis Results:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("#### Method 1")
        st.write(method_1_result)

    with col2:
        st.write("#### Method 2")
        st.write(method_2_result)

    with col3:
        st.write("#### Method 3")
        st.write(method_3_result)
else:
    st.info("Please upload an image to see the results.")

# Footer (Optional)
st.write("---")
st.caption("Created with Streamlit")
