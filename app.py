import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import tempfile
import easyocr
import os
import torch
from segment_anything import sam_model_registry, SamPredictor

yolo_model = YOLO('runs/train/product_model/weights/best.pt')
ocr_reader = easyocr.Reader(['en']) # ocr model

st.markdown("<h1 style='text-align: center;'>Key Product Features</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Annie Illing, Katherine Davis, and Andrea Noh</h3>", unsafe_allow_html=True)
st.markdown("""
    Take a picture of your product on a solid, contrasting background next to a U.S. quarter.\
    The site will return the product size, product dimensions, and the product on a \
    promotional background!        
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"]) # upload image

backgrounds_list = ["Dark Background and Base", "Dark Background White Base", "White Background White Base"]
selected_background = st.selectbox("Select a background:", backgrounds_list)
blank = False
if selected_background == "Click here":
    blank = True
elif selected_background == "Dark Background and Base":
    promo_img_path = "images/promo_dark.jpeg"
elif selected_background == "Dark Background White Base":
    promo_img_path = "images/promo.png"
elif selected_background == "White Background White Base":
    promo_img_path = "images/promo2.jpeg"

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded image', use_container_width=True)
    
    # save the uploaded image to a temporary file (needed for YOLO and OpenCV)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    results = yolo_model.predict(source=tmp_file_path, conf=0.5)
    quarter_diameter = 0.955  

    # extract bounding boxes and class indices
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    quarter_bbox = None
    product_bbox = None

    # identify the quarter and product bounding boxes
    for bbox, cls in zip(bboxes, classes):
        label = yolo_model.names[int(cls)].lower()
        if label == "quarter":
            quarter_bbox = bbox
        elif label == "product":
            product_bbox = bbox

    if quarter_bbox is None or product_bbox is None:
        st.error("Both quarter and product bounding boxes must be detected.")
    else:
        # compute the quarter's pixel diameter (average of width and height)
        q_width = quarter_bbox[2] - quarter_bbox[0]
        q_height = quarter_bbox[3] - quarter_bbox[1]
        quarter_pixel_diameter = (q_width + q_height) / 2.0

        pixel_to_inch = quarter_diameter / quarter_pixel_diameter  # conversion factor: inches per pixel

        # get product dimensions in pixels and convert to inches
        p_width = product_bbox[2] - product_bbox[0]
        p_height = product_bbox[3] - product_bbox[1]
        product_width_in = p_width * pixel_to_inch
        product_height_in = p_height * pixel_to_inch

        st.success(f"Product dimensions: width = {product_width_in:.2f} inches, height = {product_height_in:.2f} inches")

        image_cv = cv2.imread(tmp_file_path)
        x1, y1, x2, y2 = map(int, product_bbox)
        product_crop = image_cv[y1:y2, x1:x2]

        ocr_results = ocr_reader.readtext(product_crop)
        product_size_text = None
        for bbox, text, conf in ocr_results:
            if "net wt." in text.lower() and conf >= 0.75:
                product_size_text = text
                break

        if product_size_text:
            st.success(f"Product Size: {product_size_text}")
        else:
            st.write("No product size information detected.")

        image_bgr = cv2.imread(tmp_file_path, cv2.IMREAD_COLOR) # original image
        HOME = os.getcwd()
        CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        MODEL_TYPE = "vit_h"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
        mask_predictor = SamPredictor(sam)

        boxes_np = np.array([product_bbox])
        boxes_tensor = torch.from_numpy(boxes_np).to(torch.float)
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(boxes_tensor, image_bgr.shape[:2])
        mask_predictor.set_image(image_bgr)

        # segmentation mask for the product
        masks, scores, logits = mask_predictor.predict_torch(
            boxes=transformed_boxes,
            multimask_output=False,  # Single mask prediction
            point_coords=None,
            point_labels=None
        )
        final_mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255  # Scale mask for visualization

        # load and resize the promotional background image
        if blank == True:
            st.write("Product on background will appear here.")
        else:
            promo_bgr = cv2.imread(promo_img_path, cv2.IMREAD_COLOR)
            promo_resized = cv2.resize(promo_bgr, (640, 640))
            product_image = cv2.resize(image_bgr, (640, 640))
            final_mask_resized = cv2.resize(final_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
            # product_only = cv2.bitwise_and(product_image, product_image, mask=final_mask_resized) # product from the original image using the mask
            # final_composite = promo_resized.copy()
            # final_composite[final_mask_resized != 0] = product_only[final_mask_resized != 0] # product onto the promo background where the mask is non-zero
            # Get mask indices of the product
            product_mask_indices = np.argwhere(final_mask_resized > 0)
            if product_mask_indices.size == 0:
                st.error("No product pixels found in the mask.")
            else:
                top_left = product_mask_indices.min(axis=0)
                bottom_right = product_mask_indices.max(axis=0)

                # Crop product and mask using bounding box of the mask
                product_crop = product_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
                mask_crop = final_mask_resized[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

                # Resize product to be 1/2 of background height
                background_h, background_w = promo_resized.shape[:2]
                target_product_h = background_h // 2
                orig_h, orig_w = product_crop.shape[:2]
                scale_factor = target_product_h / orig_h
                new_w = int(orig_w * scale_factor)
                new_h = target_product_h

                product_resized = cv2.resize(product_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Position product so bottom is at 3/4 of background height
                y_bottom_target = int(background_h * 0.8)
                y_offset = y_bottom_target - new_h
                y_offset = max(0, min(background_h - new_h, y_offset))  # ensure on-screen

                x_offset = max(0, (background_w - new_w) // 2)  # center horizontally

                # Composite product onto promo background
                roi = promo_resized[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
                mask_inv = cv2.bitwise_not(mask_resized)
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                fg = cv2.bitwise_and(product_resized, product_resized, mask=mask_resized)
                dst = cv2.add(bg, fg)

                final_composite = promo_resized.copy()
                final_composite[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = dst

                final_composite_rgb = cv2.cvtColor(final_composite, cv2.COLOR_BGR2RGB)
                st.image(final_composite_rgb, caption="Product Resized and Positioned at 3/4 Height", use_container_width=True)
