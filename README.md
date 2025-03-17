#SugarCare: Diabetes Companion
**SugarCare: Diabetes Companion** is a web application developed as part of the Major Project provided by the Institute of Engineering, Tribhuvan University. This app aims to assist in the prediction 
and management of diabetes by utilizing advanced machine learning and deep learning techniques. The diabetes prediction system uses a Support Vector Machine (SVM), incorporating parameters such as age, 
gender, hypertension, heart disease, smoking habits, BMI, HbA1c level, and blood glucose level. In addition, the system includes a diabetic foot ulcer detection model built with a Convolutional Neural 
Network (CNN), which achieves 94% accuracy in classifying images into ulcer and non-ulcer categories. Furthermore, the application features a diabetic retinopathy detection model that achieves 80% accuracy 
in classifying images from No Diabetic Retinopathy (No DR) to Proliferative Diabetic Retinopathy (Proliferative DR), offering a comprehensive solution for diabetes care.

Tools Used: 
Python: Used for machine learning, algorithms, and data processing.
NumPy: Used for numerical operations and managing datasets.
Pandas: Used for data preprocessing and cleaning.
Scikit-Learn: Used for implementing SVM for diabetes prediction.
Bootstrap: Used for designing a responsive UI.
TensorFlow: Used for training the CNN model for foot ulcer detection.
OpenCV: Used for image preprocessing and augmentation.
Keras: Used for designing and training the CNN model.
PIL (Pillow): Used for loading and resizing medical images.
PyTorch: Used for training CNN models for retinopathy detection.
Blender: Used for creating 3D yoga animations.
Visual Studio Code: Used for writing and debugging the code.

How to run?
pip install virtualenv
python -m venv hamro_environment        for creating virtual environment of name venv
hamro_environment\scripts\activate           to activate virtual environment
then install all modules using pip install 
like 
pip install django
pip install django-crispy-forms
pip install django-bootstrap4
pip install crispy-bootstrap4
pip install Pillow
pip install Scikit-Learn
pip install matplotlib
pip install pandas

then go inside sugarcare directory suing cd sugarcare 
then run 
python manage.py runserver
then go to werbrowser and run
http://127.0.0.1:8000/
