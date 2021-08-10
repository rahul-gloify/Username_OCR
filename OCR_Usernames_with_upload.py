import threading
from typing import Union
import av
import numpy as np
import cv2
import re
import pytesseract
import streamlit as st
import spacy
from PIL import Image
from streamlit_webrtc import (RTCConfiguration,
                              VideoProcessorBase,
                              WebRtcMode,
                              webrtc_streamer
)

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def click_extract():

    NER_model = spacy.load('./Username_NER')

    class VideoTransformer(VideoProcessorBase):
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]   
        
        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None

        def recv(self, frame:av.VideoFrame) -> av.VideoFrame:
            in_image = frame.to_ndarray(format='bgr24')       
            
            with self.frame_lock:
                self.in_image = in_image

            return av.VideoFrame.from_ndarray(in_image,format="bgr24")
    
    ctx = webrtc_streamer(key='snapshot',
                          video_processor_factory=VideoTransformer,
                          mode = WebRtcMode.SENDRECV,
                          rtc_configuration=RTC_CONFIGURATION,
                          async_processing=True)

    if ctx.video_processor:
        if st.button('Snapshot'):
            with ctx.video_transformer.frame_lock:
                in_image = ctx.video_transformer.in_image
            
            if in_image is not None:
                st.write('Input Image:')
                st.image(in_image, channels='BGR')
                
                kernel = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])

                (h,w,_) = in_image.shape
                if h<w:
                    resized = cv2.resize(in_image, (1024,512))
                else:
                    resized = cv2.resize(in_image, (512,1024))
                eroded = cv2.erode(resized, kernel=np.ones((1,1), np.uint8), iterations=1, borderType=cv2.BORDER_CONSTANT)
                contrast = cv2.filter2D(eroded, -1, kernel)
                gray_img = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

                extracted_text = pytesseract.image_to_string(gray_img).split('\n')
                extracted_text = [i for i in extracted_text if i not in ['', ' ']]
                ids = re.compile(r'(@[\w]+)')
                usernames = set()
                for i in extracted_text:
                    user_id_1 = ids.match(i)
                    if user_id_1!=None:
                        usernames.add(str(user_id_1.groups()[0]))
                    user_id_2 = NER_model(i)
                    if len(user_id_2)>0:
                        for ent in user_id_2.ents:
                            usernames.add(ent.text)

                st.write(f'Usernames in the image : {usernames}')
            
            else:
                st.warning('No frames available yet')


def upload_extract():
    
    NER_model = spacy.load('./Username_NER')

    file = st.file_uploader('Choose an image', type =['png','jpeg','jpg'])
    if file is not None:
        img = Image.open(file)
        st.write('Input image:')
        st.image(img, use_column_width=True)
        img_array = np.array(img)


        kernel = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])

        (h,w,_) = img_array.shape
        if h<w:
            resized = cv2.resize(img_array, (1024,512))
        else:
            resized = cv2.resize(img_array, (512,1024))
        eroded = cv2.erode(resized, kernel=np.ones((1,1), np.uint8), iterations=1, borderType=cv2.BORDER_CONSTANT)
        contrast = cv2.filter2D(eroded, -1, kernel)
        gray_img = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

        extracted_text = pytesseract.image_to_string(gray_img).split('\n')
        extracted_text = [i for i in extracted_text if i not in ['', ' ']]
        ids = re.compile(r'(@[\w]+)')
        usernames = set()
        for i in extracted_text:
            user_id_1 = ids.match(i)
            if user_id_1!=None:
                usernames.add(str(user_id_1.groups()[0]))
            user_id_2 = NER_model(i)
            if len(user_id_2)>0:
                for ent in user_id_2.ents:
                    usernames.add(ent.text)

        st.write(f'Usernames in the image : {usernames}')


def main():

    st.header('Extract the Username')
    mode = st.sidebar.selectbox('Select a mode',
                                ['Click a picture', 'Upload a picture'])
    
    if mode == 'Click a picture':
        click_extract()
    if mode == 'Upload a picture':
        upload_extract()

if __name__ == "__main__":
    main()           
