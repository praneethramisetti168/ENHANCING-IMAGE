from flask import Flask, request, render_template, send_from_directory
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import cv2
from scipy.signal import convolve2d
import numpy as np

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('final.htm.html')

# Other routes and functions...

@app.route('/About')
def About():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        option = request.form['option']
        
        edited_image = process_image(filename, option)  # Call the image processing function with the selected option
        edited_image.save(filename)
        
        return f"File uploaded and processed successfully! <a href='/result/{file.filename}'>View Edited Image</a>"

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
def process_image(filename, option):
    img = Image.open(filename)
    
    if option == 'grayscale':
        img = ImageOps.grayscale(img)
    elif option == 'contrast_stretch':
        img = contrast_stretching(img)
    elif option == 'restoration':
        img = image_restoration(img)
    elif option == 'brightness':
        img = adjust_brightness(img)
    elif option == 'sharpen':
        img = sharpen_image(img)
    elif option == 'rotate':
        img = rotate_image(img)
    elif option == 'deblur':
        img = deblur_image(img)
    elif option == ' edge_detection':
        img = edge_detection(img)
    elif option == 'histogram_equalization':
        img = histogram_equalization(img)
    elif option == 'inpainting':
        img = inpainting(img)
    elif option == 'invert_colors':
        img = invert_colors(img)
    elif option == 'flip_horizontal':
        img = flip_horizontal(img)
    elif option == 'flip_vertical':
        img = flip_vertical(img)
    elif option == 'crop_image':
        img = crop_image(img,100,100,1100,800)
    elif option == 'noise_reduction':
        img = noise_reduction(img)
    elif option == 'sepia':
        img = apply_sepia_filter(img)
    elif option == 'equalizer':
        img = equalizer(img)
    elif option == 'morph_gradient':
        img = morphological_gradient(img)
    elif option == 'colorize_image':
        img = colorize_image(img)
    return img
def contrast_stretching(img):
   # Convert the image to a NumPy array for processing
    img_array = np.array(img)
    
    # Compute the minimum and maximum pixel values in the image
    min_pixel = np.min(img_array)
    max_pixel = np.max(img_array)
    
    # Apply contrast stretching to expand the pixel values
    img_array = (img_array - min_pixel) * (255.0 / (max_pixel - min_pixel))
    
    # Convert the NumPy array back to an image
    enhanced_img = Image.fromarray(img_array)
    
    return enhanced_img

def image_restoration(img):
     # Implement your image restoration technique here
    # This could involve deep learning, image inpainting, or other advanced methods
    # For simplicity, we'll apply a Gaussian blur in this example
    restored_img = img.filter(ImageFilter.GaussianBlur(radius=3))
    
    return restored_img


def adjust_brightness(img):
    enhancer = ImageEnhance.Brightness(img)
    enhanced_img = enhancer.enhance(1.5)  # Adjust the brightness factor as needed
    return enhanced_img

def sharpen_image(img):
    sharpened_img = img.filter(ImageFilter.SHARPEN)
    return sharpened_img

def rotate_image(img):
    rotated_img = img.rotate(90)  # Rotate the image by 90 degrees
    return rotated_img


def deblur_image(img):
    image = cv2.imread(img)
    
    # Assuming you have a known point spread function (PSF) kernel
    psf = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    
    # Deconvolve the image using Wiener deconvolution
    deblurred_img = cv2.filter2D(image, -1, psf)
    
    # Convert the deblurred image back to a format usable by Pillow (PIL)
    deblurred_img = Image.fromarray(cv2.cvtColor(deblurred_img, cv2.COLOR_BGR2RGB))
    
    return deblurred_img


def edge_detection(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges, 'L')  # 'L' mode for grayscale image



def histogram_equalization(img):
    img = ImageOps.grayscale(img)
    img_array = np.array(img)
    equalized_img = cv2.equalizeHist(img_array)
    return Image.fromarray(equalized_img)


'''def inpainting(img):
    img_array = np.array(img)
    mask = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    result = cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(result)'''

def invert_colors(img):
    inverted_img = ImageOps.invert(img)
    return inverted_img


def crop_image(img, x, y, width, height):
    img_cropped = img.crop((x, y,  width, height))
    return img_cropped

def flip_horizontal(img):
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return flipped_img

def flip_vertical(img):
    flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return flipped_img

def noise_reduction(img):
    return img.filter(ImageFilter.SMOOTH_MORE)


def apply_sepia_filter(img):
    width, height = img.size
    sepia_img = Image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
            new_r = int((r * 0.393) + (g * 0.769) + (b * 0.189))
            new_g = int((r * 0.349) + (g * 0.686) + (b * 0.168))
            new_b = int((r * 0.272) + (g * 0.534) + (b * 0.131))
            sepia_img.putpixel((x, y), (new_r, new_g, new_b))
    return sepia_img

def equalizer(img):
    img = ImageOps.grayscale(img)
    img_array = np.array(img)
    equ = cv2.equalizeHist(img_array)
    equ_img = Image.fromarray(equ)
    return equ_img


def morphological_gradient(img):
    img_array = np.array(img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    binarized_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    inverted_img = cv2.bitwise_not(binarized_img)
    kernel = np.ones((3, 3), np.uint8)
    morph_gradient = cv2.morphologyEx(inverted_img, cv2.MORPH_GRADIENT, kernel)
    return Image.fromarray(morph_gradient)

def colorize_image(img):
    img_array = np.array(img)
    height, width, _ = img_array.shape

    # Create a simple color map (e.g., brown for the entire image)
    color_map = np.zeros((height, width, 3), dtype=np.uint8)
    color_map[:, :] = [0, 50, 100]  # Change these values for different colors

    # Blend the color map with the grayscale image
    alpha = 0.5  # Adjust the blending strength
    colorized_img = cv2.addWeighted(img_array, alpha, color_map, 1 - alpha, 0)

    return Image.fromarray(colorized_img)












if __name__ == '__main__':
    app.run(debug=True)
