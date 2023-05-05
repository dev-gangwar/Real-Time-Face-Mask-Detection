import streamlit as st
import cv2
from extract_face import *
import numpy as np
import time
from timeit import default_timer as timer
from datetime import timedelta
import tensorflow as tf
from tensorflow import keras
import webbrowser
import base64
from bokeh.models.widgets import Div
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("Real Time Face Mask Detection😷")
    col1, col2 = st.columns(2)
    col2.subheader("-Major Project By G17(AIML)")

    model_custom = tf.keras.models.load_model('model_custom.h5')
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Intro", "Kaggle Notebook", "Research Paper", "Report", "Dataset", "Comparision", "Demo: Image", "Demo: Live Feed"])

    with tab1:
        st.header("Intoduction")
        st.subheader("Hello! :smiley:")
        st.write("Welcome to our Webapp. This is a webapp for our project - Real Time Face Mask Detection in this project we have tried to compare and implement the best CNN Model with our Custom made model. You must have seen many Real Time Face Mask Detection Project and I agree it is a very common topic to work with as of COVID situation and I am not saying ours is different from them. It is just our knowledge we have tried to expand through this project. So that we can gain something and also can contribute to the cause.")
        st.write("We can also train our Custom model on any other dataset and can view results for the better understanding.")
    with tab2:
        st.write("Please Visit our Kaggle Account Before Downloading the Notebook...")
        goto = st.button("Goto ...")
        if goto:
            js = "window.open('https://www.kaggle.com/code/aryankansal2019/face-mask-detection-classsifier-using-cnn')"  # New tab or window
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)
        st.write("Download the Kaggle Notebook of our Face Mask Detection Project.")

        st.download_button('Download Notebook',
                           data="face-mask-detection-classsifier-using-cnn (1).ipynb",
                           mime="text/html")
    with tab3:
        st.write("Wanna See our Reseach Paper ?")
        col1, col2 = st.columns(2)
        btn1 = col1.button("View Paper")
        with open("Research Paper.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        col2.download_button(label="Download Paper",
                            data=PDFbyte,
                            file_name="ResearchPaper.pdf",
                            mime='application/octet-stream')
        if btn1:
            displayPDF("Research Paper.pdf")
    with tab4:
        st.write("Wanna See our Report ?")
        col1, col2 = st.columns(2)
        btn1 = col1.button("View Report")
        with open("Report (17).pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        col2.download_button(label="Download Report",
                            data=PDFbyte,
                            file_name="Report.pdf",
                            mime='application/octet-stream')
        if btn1:
            displayPDF("Report (17).pdf")
    with tab5:
        st.write("View DataSet on Kaggle")
        goto_dataset = st.button("Goto  Dataset")
        if goto_dataset:
            js = "window.open('https://www.kaggle.com/datasets/andrewmvd/face-mask-detection')"  # New tab or window
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)
    with tab6:
        st.write("Comparision of All three models including Custom, EfficientNet, MobileNet.")
        st.image("output.jpg")
    with tab7:
        input = st.radio("Please Select Input Type :",options=["Insert Image", "Take a Photo"], horizontal=True)
        if input == "Insert Image":
            st.header("Insert Image")
            uploaded_img = st.file_uploader("Choose a Image", type = ["png", "jpg", "jpeg"])
            if uploaded_img is not None:
                st.write("[INFO] Please Wait while image is loading...")
                with open("img.jpeg",'wb') as f:
                    f.write(uploaded_img.read())
                img = cv2.imread("img.jpeg")
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask_label = {0:'MASK INCORRECT',1:'MASK', 2:'NO MASK'}
                color_label = {0:(0,255,255),1:(0, 255,0), 2:(255,0,0)}
                cropped_faces, rectangle_lst = extract_face(img2)
                if(len(cropped_faces) != 0):
                    for idx, face in enumerate(cropped_faces):
                        (x,y,w,h) = rectangle_lst[idx]
                        # resized_face = cv2.resize(np.array(face[0]),(35,35))
                        # st.write(resized_face)
                        reshaped_face = np.reshape(face,[1,35,35,3])/255.0
                        face_result = model_custom.predict(reshaped_face)
                        cv2.putText(img2,mask_label[face_result.argmax()],(x, y-5),cv2.FONT_HERSHEY_SIMPLEX,0.3,color_label[face_result.argmax()],2)
                    st.write("[INFO] Loading Images...")
                    col1, col2 = st.columns(2)
                    col1.image(uploaded_img, caption="Input Image")
                    col2.image(img2, caption="Output Image")
                    st.write("[INFO] Successfully identified Face Masks...")
                else:
                    st.write(":heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark: No Face Found. Please Provide a Valid Image:heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark:")
        else:
            st.write("[INFO] Accessing Camera for taking picture...")
            picture = st.camera_input("Take a picture")
            if picture:
                st.write("[INFO] Picture taken...")
                with open("img.jpeg",'wb') as f:
                    f.write(picture.read())
                img = cv2.imread("img.jpeg")
                img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mask_label = {0: 'MASK INCORRECT', 1: 'MASK', 2: 'NO MASK'}
                color_label = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
                cropped_faces, rectangle_lst = extract_face(img2)
                if (len(cropped_faces) != 0):
                    st.write("[INFO] Faces extracted...")
                    for idx, face in enumerate(cropped_faces):
                        (x, y, w, h) = rectangle_lst[idx]
                        reshaped_face = np.reshape(face, [1, 35, 35, 3])/255.0
                        face_result = model_custom.predict(reshaped_face)
                        cv2.putText(img2,mask_label[face_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_label[face_result.argmax()],2)
                    st.write("[INFO] Loading Images...")
                    col1, col2 = st.columns(2)
                    col1.image(picture, caption="Input Image")
                    col2.image(img2, caption="Output Image")
                    st.write("[INFO] Successfully identified Face Masks...")
                else:
                    st.write(":heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark: No Face Found. Please Adjust lightening and Take a Photo again...:heavy_exclamation_mark::heavy_exclamation_mark::heavy_exclamation_mark:")

    with tab8:
        st.header("Webcam Live Feed")
        run = st.checkbox("Start Cam")
        dic = {"6 frames/sec" : 6,
               "12 frames/sec" : 12,
               "24 frames/sec" : 24,
               "60 frames/sec" : 60
               }
        fps = st.selectbox("Select FPS at which you have to give input to our model : (Note : Higher the FPS slower the results you will get)",
                           ("6 frames/sec", "12 frames/sec", "24 frames/sec", "60 frames/sec"))
        
        col1, col2 = st.columns(2)
        FRAME_WINDOW = col1.image([])
        FRAME_WINDOW_OUT = col2.image([])
        try :
            camera = cv2.VideoCapture(0)
            t = st.empty()
            while run:
                start = timer()
                _, frame_in = camera.read()
                st.image(frame_in)
                frame = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
                frame2 = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
                mask_label = {0: 'MASK INCORRECT', 1: 'MASK', 2: 'NO MASK'}
                color_label = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
                cropped_faces, rectangle_lst = extract_face(frame2)
                if (len(cropped_faces) != 0):
                    for idx, face in enumerate(cropped_faces):
                        (x, y, w, h) = rectangle_lst[idx]
                        reshaped_face = np.reshape(face, [1, 35, 35, 3])/255.0
                        face_result = model_custom.predict(reshaped_face)
                        cv2.putText(frame2,mask_label[face_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_label[face_result.argmax()],2)
                end = timer()
                t.markdown(f"Showing the feed with the delay of {timedelta(seconds= end - start)}")
                FRAME_WINDOW.image(frame, caption=f"Input Feed at {fps}.")
                FRAME_WINDOW_OUT.image(frame2, caption= f"Output feed at {fps}.")
                time.sleep(1/dic[fps])
            else:
                st.write("Live Feed Stoped !!!😵")
        except:
            st.header("Sorry!!! Error while loading the Cam.") 
