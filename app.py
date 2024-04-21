import streamlit as st
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

def calculate_similarity(image1, image2):
    """
    Calculates the similarity between two images using face_recognition.
    """
    # Convert images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations1 = face_recognition.face_locations(image1_rgb)
    face_locations2 = face_recognition.face_locations(image2_rgb)

    if not face_locations1 or not face_locations2:
        return False, 0, False

    # Extract face encodings
    face_encodings1 = face_recognition.face_encodings(image1_rgb, face_locations1)[0]
    face_encodings2 = face_recognition.face_encodings(image2_rgb, face_locations2)[0]

    # Compare faces and calculate distance
    face_distance = face_recognition.face_distance([face_encodings1], face_encodings2)[0]

    # Convert distance to a similarity percentage
    similarity_percentage = max(0, (1 - face_distance) * 100)

    # Determine if faces are similar based on a threshold
    similarity_threshold = 0.6
    is_similar = face_distance < similarity_threshold

    return True, similarity_percentage, is_similar

def make_circle(percent):
    """
    Create a circular 'progress' bar showing the given percentage.
    Color is red for values below 50% and green for values at or above 50%.
    """
    color = 'red' if percent < 50 else 'green'
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(1.5, 1.5), subplot_kw=dict(aspect="equal"))

    # The start and end angles in degrees of the arc in the circle
    start_angle = 0  # Starting from the right
    end_angle = 360 * (percent / 100)  # Filling clockwise

    # Create a full circle as the 'background'
    ax.add_artist(Circle((0.5, 0.5), 0.4, color='lightgray', zorder=0))

    # Create a 'wedge' representing the percentage
    ax.add_artist(Wedge((0.5, 0.5), 0.4, start_angle, end_angle, color=color, zorder=1))

    # Add a smaller white circle in the middle to make it look like a donut chart
    ax.add_artist(Circle((0.5, 0.5), 0.3, color='white', zorder=2))

    # Add the text in the middle
    plt.text(0.5, 0.5, f"{percent:.2f}%", ha='center', va='center', fontsize=8, zorder=3)

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    return fig

def main():
    st.title("Face Similarity Checker")

    # Create two columns for the file uploaders
    col1, col2, col3 = st.columns(3)
    
    with col1:
        file1 = st.file_uploader("Choose the first image", type=["jpg", "jpeg", "png"], key="file1")
    
    with col2:
        file2 = st.file_uploader("Choose the second image", type=["jpg", "jpeg", "png"], key="file2")

    # Initialize variables to hold the images and the similarity result
    image1, image2, similarity_percentage = None, None, None

    if file1 and file2:
        # Convert the uploaded files to OpenCV images
        file_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
        image1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)

        file_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
        image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

        # Calculate similarity
        has_faces, similarity_percentage, is_similar = calculate_similarity(image1, image2)

    # Display the images and progress bar in three columns
    col1, col2, col3 = st.columns([1, 1, 2])
    if image1 is not None:
        with col1:
            st.image(image1, channels="BGR", caption="First Image", width=150)
    if image2 is not None:
        with col2:
            st.image(image2, channels="BGR", caption="Second Image", width=150)
    if similarity_percentage is not None:
        with col3:
            fig = make_circle(similarity_percentage)
            st.pyplot(fig)

    if file1 and file2:
        if not has_faces:
            st.error("No faces detected in one or both images.")
        else:
            if is_similar:
                st.success("The faces are likely similar.")
            else:
                st.error("The faces are likely not similar.")

if __name__ == "__main__":
    main()
