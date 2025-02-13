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
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

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
    l=[]
    for page in result1.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                #print(line_text)
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
            print("Confidence Score",confidence)
            if confidence > 0.70:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                # Crop the image to the bounding box size
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                res=extract_from_columns(cropped_img)
                return res
    return []

def handle_grade_key(d):
    if "Grade" in d:
        grade_list = d["Grade"]
        valid_grades=['A','A+','O','B','B+','F','o']
        for i in range(len(grade_list)):
            # If current value is None or not in valid grades list
            if grade_list[i] is None or grade_list[i] not in valid_grades:
                # Look for the next valid value
                next_val = None
                for j in range(i + 1, len(grade_list)):
                    if grade_list[j] in valid_grades:  # Check if it's a valid grade
                        next_val = grade_list[j]
                        break
                
                if next_val is not None:
                    grade_list[i] = next_val
                elif i > 0 and grade_list[i - 1] in valid_grades:
                    grade_list[i] = grade_list[i - 1]
        d['Grade']=grade_list
    return d

def combine_extract(image):
    try:
        image=denoise_image(image)
        table_data=extract_table(image)
        details=extract_details(image)
        #details.extend(table_data)
        data = {col[0]: col[1:] for col in table_data}
        c=0
        for key in data:
            #k=key.lower()
            pattern = r'\b(s[il])[\s\.]*no[\.\s]*\b'
            # if k=="si.no" or k=="slno" or k=="sino" or k=="sl.no" :
            if re.match(pattern, key, re.IGNORECASE):
                for j in data[key]:
                    if j.isdigit():
                        c+=1
                break
        print(data)
        for key in data:
            data[key] = data[key][:c] + [None] * (c - len(data[key])) if len(data[key]) < c else data[key][:c]
        print("Table data completed",data)
        data=handle_grade_key(data)
        for col in details:
            data[col[0]]=col[1:]
        return data
    except Exception as e:
        print("An exception occured in combine_extract_bills method: ",e)

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
            data_dict=combine_extract(image)
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
                data_dict=combine_extract(img)
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

@csrf_exempt
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
            #return render(request,'upload_success.html',{"data":json.dumps(processed_files, indent=4)})
            return JsonResponse({
                "status": True,
                "data": processed_files
            })
            
        else:
            return JsonResponse({
            "status": False,
            "msg": "Only POST method allowed"
            }, status=400)
    except Exception as e:
        print("Error Ocuured",e)
        return JsonResponse({
                "status": False,
                "data": [],
                "msg":f"Error occured: {str(e)}"
            })

# def extract_bill_details(img):
#     result1 = docTr_model([np.array(img)])
#     print("Bill details extraction called....!")
#     res=[]
#     prev_line=""
#     for page in result1.pages:
#         for block in page.blocks:
#             for line in block.lines:
#                 line_text = " ".join([word.value for word in line.words])
#                 print(line_text)
#                 if "Receipt No" in prev_line:
#                     res.append(["Receipt No", line_text])
#                 if "Date" in prev_line:
#                     res.append(["Date", line_text])
#                 if "Name" in prev_line:
#                     res.append(["Name", line_text])
#                 if "Branch" in prev_line:
#                     res.append(["Branch", line_text])
#                 if "PIN" in prev_line:
#                     res.append(["Pin", line_text])
#                 if "Year" in prev_line:
#                     res.append(["Year", line_text])
#                 if "Total" in prev_line:
#                     res.append(["Amount",line_text])
#                 prev_line = line_text
#     return res

def extract_bill_details(img):
    result1 = docTr_model([np.array(img)])
    l = [] 
    last_key = None  # Store the last key if value is on the next line
    
    for page in result1.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join([word.value for word in line.words])
                #print(line_text)
                # Check for keys and values in the same line
                name = re.search(r'Name\s*:\s*(.*)', line_text)
                if name:
                    l.append(["Name", name.group(1)])
                    last_key = None  # Reset last_key if value is found on the same line
                    continue
                
                receipt = re.search(r'Receipt\s*No\s*:\s*(.*)', line_text)
                if receipt:
                    l.append(["Receipt No", receipt.group(1)])
                    last_key = None
                    continue
                
                date = re.search(r'Date\s*:\s*(.*)', line_text)
                if date:
                    l.append(["Date", date.group(1)])
                    last_key = None
                    continue
                
                branch = re.search(r'Branch\s*:\s*(.*)', line_text)
                if branch:
                    l.append(["Branch", branch.group(1)])
                    last_key = None
                    continue
                
                pin = re.search(r'PIN\s*:\s*(.*)', line_text)
                if pin:
                    l.append(["Pin", pin.group(1)])
                    last_key = None
                    continue
                
                year = re.search(r'Year\s*:\s*(.*)', line_text)
                if year:
                    l.append(["Year", year.group(1)])
                    last_key = None
                    continue
                
                # If no value on the same line, check if it's just the key
                if re.search(r'Name\s*:', line_text):
                    last_key = "Name"
                elif re.search(r'Receipt\s*No\s*:', line_text):
                    last_key = "Receipt No"
                elif re.search(r'Date\s*:', line_text):
                    last_key = "Date"
                elif re.search(r'Branch\s*:', line_text):
                    last_key = "Branch"
                elif re.search(r'PIN\s*:', line_text):
                    last_key = "Pin"
                elif re.search(r'Year\s*:', line_text):
                    last_key = "Year"
                else:
                    # If last_key is set, treat this line as the value
                    if last_key:
                        l.append([last_key, line_text])
                        last_key = None  # Reset last_key after using it
    return l


def combine_extract_bills(image):
    try:
        image=denoise_image(image)
        details=extract_bill_details(image)
        data = {col[0]: col[1:] for col in details}
        max_length = max(len(v) for v in data.values())
        for key in data:
            data[key] += [None] * (max_length - len(data[key]))
        return data
    except Exception as e:
        print("An exception occured in combine_extract_bills method: ",e)


def extract_bills(file):
    try:
        if file.content_type.startswith('image/'):
            image_stream = file.read()
            image_data = np.frombuffer(image_stream, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            data=combine_extract_bills(image)
            json_data = [data]
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
                data=combine_extract_bills(img)
                processed_files.append(data)
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

@csrf_exempt
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
            # return render(request,'upload_success.html',{"data":json.dumps(processed_files, indent=4)})
            return JsonResponse({
                "status": True,
                "data": processed_files
            })
            
        else:
            return JsonResponse({
                "status": False,
                "msg": "Only POST method allowed"
            }, status=400)
        #return render(request, 'testing.html')
    except Exception as e:
        print("Error Ocuured",e)
        return JsonResponse({
                "status": False,
                "data": [],
                "msg":f"Error occured: {str(e)}"
            })