import streamlit as st

# Set page title and icon
st.set_page_config(page_title="Top Hospital", page_icon="🏥")

# Define the main header with creative styling
st.title("Top Hospital")
st.markdown(
    """
    <div style="background-color: #007bff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: #ffffff;">Delivering Quality Healthcare</h1>
        <p style="color: #ffffff;">Your Health, Our Priority</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Create a navigation menu with unique icons and a creative layout
menu_selection = st.sidebar.selectbox("Menu", ["🏡 Home", "💼 Services", "👨‍⚕️ Doctors", "📅 Appointments", "✉️ Contact Us"])

# Create a dark mode toggle with a moon icon
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# Define the content for each menu option
if menu_selection == "🏡 Home":
    st.header("Welcome to Top Hospital")
    st.image(r"D:\drug interactions\images.jpeg", use_column_width=True, caption="Hospital Image")
    st.markdown("Welcome to Top Hospital. We are committed to providing world-class healthcare services.")

elif menu_selection == "💼 Services":
    st.header("Our Services")
    st.markdown("Discover our comprehensive healthcare services.")
    # List and describe hospital services

elif menu_selection == "👨‍⚕️ Doctors":
    st.header("Meet Our Doctors")
    st.markdown("Our dedicated team of healthcare professionals.")
    # Display doctor profiles with images and specialties

elif menu_selection == "📅 Appointments":
    st.header("Book an Appointment")
    st.markdown("Schedule an appointment with our healthcare experts.")
    # Include an appointment booking form

elif menu_selection == "✉️ Contact Us":
    st.header("Contact Us")
    st.markdown("Get in touch with us.")
    # Include contact information and a contact form

# Apply dark mode with creative styling
if dark_mode:
    st.markdown(
        """
        <style>
            body {
                background-color: #121212;
                color: #ffffff;
                font-family: Arial, sans-serif;
            }
            .sidebar .sidebar-content {
                background-color: #333;
                color: #ffffff;
            }
            .st-bw {
                background-color: #333;
                color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    /* Creative CSS styles */
    /* Customize creative styles here */
    </style>
    """,
    unsafe_allow_html=True,
)

# Run the Streamlit app
if __name__ == "__main__":
    st.run()
