# Smartphone Addiction Prediction System (SAps)

## Simple Guide to Execute on a New Laptop (using VS Code)

Follow these steps to set up and run the project on a different machine:

### Step 1: Open the Project in VS Code

1. Copy the entire project folder to the new laptop.
2. Open **VS Code**.
3. Go to `File` > `Open Folder...` and select the project folder (`smartphone-addiction-prediction`).

### Step 2: Open a New Terminal

1. In VS Code, go to the top menu and select `Terminal` > `New Terminal`.

### Step 3: Create and Activate a Virtual Environment

Run the following commands in the terminal:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

_(You should see `(venv)` appear in your terminal prompt. If you get an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)._

### Step 4: Install Dependencies

With the virtual environment activated, install the required packages:

```powershell
pip install -r requirements.txt
```

### Step 5: (Optional) Generate Data & Train the Model

If the `data/` and `models/` folders are missing, you must generate the dataset and train the model before running the server:

```powershell
python generate_dataset.py
python train_model.py
```

### Step 6: Run the App

Start the Flask server:

```powershell
python app.py
```

**Access the Application:**
Once it is running, open your web browser and go to: [http://localhost:5000](http://localhost:5000). To stop the server later, click inside the terminal and press `Ctrl+C`.
