
# Steps to Run Locally:

- Clone [this](https://github.com/fizyr/keras-retinanet.git) repository
- `cd keras-retinanet`
- `python -m pip install .`
- if this error  comes `error: command 'gcc' failed with exit status 1`
- install this first `conda install -c conda-forge fbprophet` or via pip.
- __requirements installation__:
     - make a virtual environment -
         * `pip install virtualenv`
         * `virtualenv {env name}`
         * `{env name}\Scripts\activate`
    - Install all requirements: `pip install -r requirements.txt`
(Note: there might be errors in some installation due other module dependencies, therefore, install basic modules like numpy first)
- `python setup.py build_ext --inplace`
- Download the classes.csv file and the templates folder, place them in the cloned repository's folder.
- Also create a empty folder named `imgs` in the cloned folder.
- Create a folder named static in the same folder with the following tree:
    * Static
        * images
            * place the image named img.jpg to show in welcome page
        * output
            * leave this folder empty, output will get saved here.

- Place the `flask_server.py` also in the cloned repo's folder.
- [Link to the trained model(250 Mb)](https://drive.google.com/file/d/1N1ZazBRPI-tt4iG4Jt-MhyTWW8aQREDQ/view?usp=sharing)



# Changes in the `flask_server.py` file:
- `line 29` : Change the absolute path.
- `line 31` : Change the absolute path.
- `line 92` : Change the absolute path.
- `line 95` : Change the absolute path.
- `line 97` : Change the absolute path.
- `line 98` : Change the absolute path.
