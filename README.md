# recognising_galaxy_types

Installation instructions
----------------------------

The project is coded in Python using the Keras and TensorFlow libraries. 
- Requires the latest versions of Keras, Keras-Preprocessing.
- Requires the latest versions of tensorflow, tensorflow-estimator.
- The version of Python needed is 3.8.

To install Keras and TensorFlow2, python has to be installed on your machine.

STEP 1
------

Either download the source code or clone it to wherever you plan to run it on your local machine.


STEP 2
------

Create a virtual environment in the directory where you're running the project.

**Virtualenv** is used to manage Python packages for different projects. This will be helpful to avoid breaking the packages installed in the other environments.

For **Linux/Mac OS** users:
- Go to the project root directory and type the below command to create a virtual environment
- *python3 -m venv kerasenv*
- After executing the above command, “kerasenv” directory is created with bin, lib and include folders in your installation location.

For **Windows** users:
- Windows user can use the below command,
- *py -m venv keras*


STEP 3
------

Activate the environment. This step will configure python and pip executables in your shell path.

For **Linux/Mac OS** users:
- Now we have created a virtual environment named “kerasvenv”. Move to the folder and type the below command,
- *$ cd kerasvenv kerasvenv $ source bin/activate*

For **Windows** users:
- Windows users move inside the “kerasenv” folder and type the below command,
- *.\env\Scripts\activate*


STEP 4
------

Keras depends on the following python libaries:
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Scipy

If these are not installed in your system, follow the instructions below:

**numpy**:
- *pip install numpy*

**pandas**:
- *pip install pandas*

**matplotlib**:
- *pip install matplotlib*

**scipy**:
- *pip install scipy*

**scikit-learn**:

- It is an open source machine learning library. It is used for classification, regression and clustering algorithms. Before moving to the installation, it requires the following − **Python version 3.5 or higher**, **NumPy version 1.11.0 or higher**, **SciPy version 0.17.0 or higher**, **joblib 0.11 or higher.**

- *pip install -U scikit-learn*


STEP 5
------

Keras installation using python.

As of now, we have completed basic requirements for the installtion of Kera. Now, install the Keras using same procedure as specified below −

- *pip install keras*



STEP 6
-------

Install the TensorFlow pip package

- *pip install --upgrade tensorflow*

Verify the installation:

- *python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"*

*SUCCESS: If a tensor is returned, you've installed TensorFlow successfully.** 


STEP 7
-------

Quit virtual environment

After finishing all your changes in your project, then simply run the below command to quit the environment −

- *deactivate*


STEP 8
-------

Now that everything's installed, open model.py from the git repo in your IDE.

In model.py, there are a list of functions that have comments that describe what their purpose is. For you to run the models on the dataset that comes in the project directory, uncomment one of the four lines at the end of the code that call an instance of their respective method.

**e.g., if you want to run the base VGG16 model:**
- * #vgg16_model() * <---- **UNCOMMENT THIS LINE**

From there, you can run the rest of the models and have a look at whatever parameters you'd like to change if you navigate to the commented section that shows where the configuration is for each model.


















