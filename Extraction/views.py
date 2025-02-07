from django.shortcuts import render
import PIL.Image
import pandas as pd
from ultralytics import YOLO
import numpy as np
from doctr.models import ocr_predictor
import json
import re
import os
import cv2
import locale
from django.conf import settings
from pdf2image import convert_from_path

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
# Create your views here.

table_model_path = os.path.join( os.path.dirname(__file__), 'table_detect_best.pt')
column_model_path = os.path.join( os.path.dirname(__file__), 'column_detect_best.pt')

table_detect= YOLO(table_model_path )
column_detect = YOLO(column_model_path)
docTr_model = ocr_predictor(pretrained=True)

def denoise_image(image):
    # Convert to 8-bit if image is in [0,1] range
    image = (image * 255).astype(np.uint8) if image.max() <= 1 else image
    # Apply denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

def extract_details(img):
    result1 = docTr_model([np.array(img)])
    print("Details extraction called....!")
    l=[]
    for page in result1.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                print(line_text)
                student_name = re.search(r'Student\s*(.*)', line_text)
                if student_name:
                    l.append(["Name",student_name.group(1)])
                parent_name = re.search(r'Parent\s*(.*)', line_text)
                if parent_name:
                    l.append(["Parent Name",parent_name.group(1)])
                course_info = re.search(r'Course\s*-\s*Exam\s*(.*)', line_text)
                if course_info:
                    l.append(["Course-Exam",course_info.group(1)])
                branch_info = re.search(r'Branch\s*(.*)', line_text)
                if branch_info:
                    l.append(["Branch ",branch_info.group(1)])
                pattern = r'\b\d{5}[A-Z]\d{4}\b'
                pin = re.findall(pattern, line_text)
                if pin:
                    l.append(["Pin",pin[0]])
    return l

def extract_from_columns(img):
    # results=column_detect([np.array(img)])
    results=column_detect(img)
    print("Column Detect Executed....!")
    print("No of columns detected:",len(results[0].boxes))
    columns=[]
    for result in results:
        for box in result.boxes:
            # Extract confidence score and bounding box coordinates
            confidence = box.conf.item()  # confidence score
            if confidence > 0.60:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                # Crop the image to the bounding box size
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                res = docTr_model([np.array(cropped_img)])
                for page in res.pages:
                    for block in page.blocks:
                        temp=[]
                        for line in block.lines:
                            line_text = " ".join([word.value for word in line.words])
                            temp.append(line_text)
                        columns.append(temp)
    return columns

def extract_table(img):
    print("Extract Table method called")
    # image=np.array(img)
    results=table_detect(img)
    for result in results:
        for box in result.boxes:
            print("Inside the boxes...:::")
            # Extract confidence score and bounding box coordinates
            confidence = box.conf.item()  #confidence score
            print("COnfidence Score",confidence)
            if confidence > 0.70:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                # Crop the image to the bounding box size
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                res=extract_from_columns(cropped_img)
                return res
    return []

def extract(file):
    try:
        if file.content_type.startswith('image/'):
            # img = PIL.Image.open(file)
            # Read the image as a byte stream
            image_stream = file.read()
        
            # Convert the byte stream to a NumPy array
            image_data = np.frombuffer(image_stream, np.uint8)
        
            # Decode the image using OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            image=denoise_image(image)
            table_data=extract_table(image)
            print("Table Detection Completed...!")
            details=extract_details(image)
            print("Details Extraction Completed...!")
            details.extend(table_data)
            data_dict = {col[0]: col[1:] for col in details}
            json_data = [data_dict]
            #json_output = json.dumps(json_data, indent=4)
            print("Json conversion completed")
            return json_data
                                        
        elif file.content_type == 'application/pdf':
            temp_path = os.path.join(settings.MEDIA_ROOT, file.name)
            print("Temp path creation started")
            with open(temp_path, 'wb') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
            processed_files=[]
            print("Temp path created")
            images = convert_from_path(temp_path, dpi=300)
            for img in images:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img=denoise_image(img)
                extracted=extract_table(img)
                details=extract_details(img)
                details.extend(extracted)
                data_dict = {col[0]: col[1:] for col in details}
                processed_files.append(data_dict)
            os.remove(temp_path)
            #json_output = json.dumps(processed_files, indent=4)
            #return json_output
            return processed_files

        elif file.content_type == 'text/plain':
            file_content = file.read().decode('utf-8')
            return file_content
        
        else:
            return f"Unsupported file type: {file.content_type}"

    except Exception as e:
        print(f"An error occurred while processing the file {file.name}: {e}")
        return None

def upload_files(request):
    try:
        if request.method == 'POST':
            uploaded_files = request.FILES.getlist('files')
            processed_files = []
            data=[]
            for uploaded_file in uploaded_files:
                #file_content = uploaded_file.read()
                # file = request.files['image']
                extracted=extract(uploaded_file)
                #print(extracted)
                processed_files.extend( extracted)
                print("Processing completed...")
            #print("Processed Files:",processed_files)
            return render(request,'upload_success.html',{"data":json.dumps(processed_files, indent=4)})
            
        return render(request, 'testing.html')
    except Exception as e:
        print("Error Ocuured",e)
        return None
def extract_bill_details():
    result1 = docTr_model([np.array(img)])
    print("Bill details extraction called....!")
    l=[]
    for page in result1.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                print(line_text)
                student_name = re.search(r'Name\s*(.*)', line_text)
                if student_name:
                    l.append(["Name",student_name.group(1)])
                parent_name = re.search(r'Receipt\sNo\s*(.*)', line_text)
                if parent_name:
                    l.append(["Receipt No",parent_name.group(1)])
                course_info = re.search(r'Date\s*(.*)', line_text)
                if course_info:
                    l.append(["Date",course_info.group(1)])
                branch_info = re.search(r'Branch\s*(.*)', line_text)
                if branch_info:
                    l.append(["Branch ",branch_info.group(1)])
                # pattern = r'\b\d{5}[A-Z]\d{4}\b'
                pin = re.search(r'PIN\s*(.*)', line_text)
                if pin:
                    l.append(["Pin",pin.group(1)])
                year = re.search(r'Year\s*(.*)', line_text)
                if year:
                    l.append(["Year",year.group(1)])
    return l

def extract_bills(file):
    try:
        if file.content_type.startswith('image/'):
            image_stream = file.read()
            image_data = np.frombuffer(image_stream, np.uint8)
        
            # Decode the image using OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            image=denoise_image(image)
            table_data=extract_table(image)
            details=extract_bill_details(image)
            details.extend(table_data)
            data_dict = {col[0]: col[1:] for col in details}
            json_data = [data_dict]
            #json_output = json.dumps(json_data, indent=4)
            #print("Json conversion completed")
            return json_data
                                        
        elif file.content_type == 'application/pdf':
            temp_path = os.path.join(settings.MEDIA_ROOT, file.name)
            with open(temp_path, 'wb') as temp_file:
                for chunk in file.chunks():
                    temp_file.write(chunk)
            processed_files=[]
            images = convert_from_path(temp_path, dpi=300)
            for img in images:
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img=denoise_image(img)
                extracted=extract_table(img)
                details=extract_details(img)
                details.extend(extracted)
                data_dict = {col[0]: col[1:] for col in details}
                processed_files.append(data_dict)
            os.remove(temp_path)
            #json_output = json.dumps(processed_files, indent=4)
            #return json_output
            return processed_files

        elif file.content_type == 'text/plain':
            file_content = file.read().decode('utf-8')
            return file_content
        
        else:
            return f"Unsupported file type: {file.content_type}"

    except Exception as e:
        print(f"An error occurred while processing the file {file.name}: {e}")
        return None

def upload_Bills(request):
    try:
        if request.method == 'POST':
            uploaded_files = request.FILES.getlist('files')
            processed_files = []
            data=[]
            for uploaded_file in uploaded_files:
                
                extracted=extract_bills(uploaded_file)
                processed_files.extend( extracted)
                print("Processing completed...")
            #print("Processed Files:",processed_files)
            return render(request,'upload_success.html',{"data":json.dumps(processed_files, indent=4)})
            
        return render(request, 'testing.html')
    except Exception as e:
        print("Error Ocuured",e)
        return None