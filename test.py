from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
model = load_model('PId_Best.h5')
labels = ['Alpinia Galanga (Rasna)','Amaranthus Viridis (Arive-Dantu)','Artocarpus Heterophyllus (Jackfruit)', 'Azadirachta Indica (Neem)', 'Basella Alba (Basale)','Brassica Juncea (Indian Mustard)','Carissa Carandas (Karanda)','Citrus Limon (Lemon)','Ficus Auriculata (Roxburgh fig)','Ficus Religiosa (Peepal Tree)','Hibiscus Rosa-sinensis','Jasminum (Jasmine)','Mangifera Indica (Mango)','Mentha (Mint)','Moringa Oleifera (Drumstick)','Muntingia Calabura (Jamaica Cherry-Gasagase)','Murraya Koenigii (Curry)','Nerium Oleander (Oleander)','Nyctanthes Arbor-tristis (Parijata)','Ocimum Tenuiflorum (Tulsi)','Piper Betle (Betel)','Plectranthus Amboinicus (Mexican Mint)','Pongamia Pinnata (Indian Beech)','Psidium Guajava (Guava)','Punica Granatum (Pomegranate)','Santalum Album (Sandalwood)','Syzygium Cumini (Jamun)','Syzygium Jambos (Rose Apple)','Tabernaemontana Divaricata (Crape Jasmine)','Trigonella Foenum-graecum (Fenugreek)','']
img_heigh, img_with = 150, 150
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image and converts  into an object 
        that can be used as input to a trained model, returns an Numpy array.

        Arguments
        ---------
        image_path: string, path of the image.
    '''
    
   

def process(img_path):
    #model = load_model("PId_Best.h5")
    im=cv2.imread(img_path)
    cv2.imshow("Input Image",im)
    img = load_img(img_path, target_size = (img_heigh, img_with))

    x = img_to_array(img)
    x = x/255
    x = x.reshape(1, img_heigh, img_with, 3)
    pred = model.predict(x)
    print("pred==",pred)
    pclass=labels[np.argmax(pred[0])]
    conf=round(np.max(pred), 2)*100
    return pclass,conf


    
   
#print("Predicted=",process("./Dataset/Basella Alba (Basale)/BA-S-001.jpg"))
    
