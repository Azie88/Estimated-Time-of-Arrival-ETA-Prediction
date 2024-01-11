# Estimated Time of Arrival (ETA) Prediction
![App (1)](https://github.com/Azie88/Estimated-Time-of-Arrival-ETA-Prediction/assets/101363399/62edada4-6a6a-415a-9770-774859666d41)

Creating an ML model to predict the estimated time of arrival at the dropoff point for a single journey on a ride-hailing app.

## Project Summary 📄

| Name | Deployed App | Presentation |
|------|------|-------------------|
| Yassir ETA prediction | <a href="https://huggingface.co/spaces/Azie88/ETA_prediction">Gradio App on Huggingface</a> | <a href="https://tome.app/club-unbrick/machine-learning-project-eta-prediction-for-yassir-clr6mqgby0349o463g2lgpfjr">Tome Presentation</a>

## Project Structure 📂

- `Dataset/`: Contains the dataset used for analysis.
- `toolkit/`: Pieline with ML model
- `.gitignore`: Holds files to be ignored by Git.
- `app.py`: Working Gradio app for prediction
- `LICENSE`: Project license.
- `Project_notebook.ipynb`: The jupyter notebook with full end-to-end ML process
- `README.md`: Project overview, links, highlights, and information.
- `requirements.txt`: Required libraries & packages

## Getting Started🏁

You need to have [`Python 3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's `root :: repository_name> ...`

1. Clone this repository: `git clone https://github.com/Azie88/LP1-Data-Analysis.git`
2. On your IDE, create A Virtual Environment and Install the required packages for the project:

- Windows:
        
        python -m venv venv; 
        venv\Scripts\activate; 
        python -m pip install -q --upgrade pip; 
        python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; 
        source venv/bin/activate; 
        python -m pip install -q --upgrade pip; 
        python -m pip install -qr requirements.txt  

The two long command-lines have the same structure. They pipe multiple commands using the symbol ` ; ` but you can manually execute them one after the other.

- **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
- **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
- **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
- **Install the required libraries/packages** listed in the `requirements.txt` file so that they can be imported into the python script and notebook without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.

3. Explore the Jupyter notebook for detailed steps and code execution.
4. Check out the Power BI dashboard for interactive visualizations.
5. Read the published article for a comprehensive understanding of the project.

## Author✍️

Andrew Obando

Connect with me on LinkedIn: [Andrew Obando](https://www.linkedin.com/in/andrewobando/)

---

Feel free to star ⭐ this repository if you find it helpful!
