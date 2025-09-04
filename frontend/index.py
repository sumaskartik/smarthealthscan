import streamlit as st  # type: ignore
import requests

st.set_page_config(page_title="🩺 Health Checkup Assistant", layout="wide")

st.title("🩺 Health Checkup Assistant")

# Sidebar: document management
st.sidebar.header("📂 Upload Medical Reports")
uploaded_files = st.sidebar.file_uploader(
    "Upload your lab reports or medical documents",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

API_URL_UPLOAD = "http://localhost:8000/add_document"
API_URL_CHAT = "http://localhost:8000/chat"

if uploaded_files:
    for file in uploaded_files:
        try:
            # Dynamically set MIME type based on extension
            ext = file.name.split(".")[-1].lower()
            mime_map = {
                "pdf": "application/pdf",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "png": "image/png"
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

            files = {"file": (file.name, file.getvalue(), mime_type)}
            response = requests.post(API_URL_UPLOAD, files=files)

            if response.status_code == 200:
                resp_text = response.text
                st.sidebar.success(f"✅ {file.name} uploaded successfully\nResponse: {resp_text}")
            else:
                st.sidebar.error(f"❌ Failed to upload {file.name}: {response.text}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error uploading {file.name}: {str(e)}")

# Main chat area
st.subheader("💬 Chat about your health reports")

# Chat input
user_input = st.text_input("Type your health-related question here...")

if st.button("Ask"):
    try:
        resp = requests.post(API_URL_CHAT, data={"query": user_input})
        if resp.status_code == 200:
            response = resp.json().get("analysis", "")
        else:
            response = f"Error: {resp.text}"
    except Exception as e:
        response = f"⚠️ API Error: {str(e)}"

    st.markdown(f"**Health Assistant:**\n\n{response}")
