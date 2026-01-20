from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import shutil
import os
import uuid
from app.services.alignment import align_images
from app.services.discriminator import compare_images

router = APIRouter()

@router.post("/analyze")
async def analyze_images(master: UploadFile = File(...), test: UploadFile = File(...)):
    # 1. Create unique filenames so we don't overwrite old uploads
    master_filename = f"uploads/master/{uuid.uuid4()}_{master.filename}"
    test_filename = f"uploads/test/{uuid.uuid4()}_{test.filename}"
    
    # 2. Save the uploaded files to your hard drive
    with open(master_filename, "wb") as buffer:
        shutil.copyfileobj(master.file, buffer)
    with open(test_filename, "wb") as buffer:
        shutil.copyfileobj(test.file, buffer)

    # 3. Read images into OpenCV
    img_master = cv2.imread(master_filename)
    img_test = cv2.imread(test_filename)

    if img_master is None or img_test is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    try:
        # 4. Align the Test Image to match the Master Image's angle
        aligned_test = align_images(img_test, img_master)
        
        # 5. Compare them to find differences
        report = compare_images(img_master, aligned_test)
        
        # 6. Save the Result Image (the one with red boxes)
        output_filename = f"report_{uuid.uuid4()}.jpg"
        output_path = f"outputs/{output_filename}"
        cv2.imwrite(output_path, report["processed_image"])

        # 7. Return the data to the user
        return {
            "status": "success",
            "similarity_score": report['similarity_score'],
            "color_match_score": report['color_match_score'],
            "anomalies_found": report["anomalies_detected"],
            "report_image_url": f"/outputs/{output_filename}"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed. Ensure images have enough common features.")