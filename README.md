# 📦 Project Setup & Environment Reproducibility

This project uses a Conda environment to ensure consistent and reproducible development and execution across different systems.

## 🔁 Reproducing the Environment

To get started, please use the provided `environment.yml` file. It includes all necessary dependencies and their specific versions.

### ✅ Steps to Set Up the Environment

1. **Install Conda (if not already installed)**  
   Download and install Miniconda or Anaconda:  
   👉 [Miniconda Installation Guide](https://docs.conda.io/en/latest/miniconda.html)

2. **Create the environment from the YAML file**  
   Run the following command in your terminal:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**  
   After creation, activate the environment:
   ```bash
   conda activate your_environment_name
   ```
   > Replace `your_environment_name` with the actual name defined in the `environment.yml` file.

4. ✅ You're all set! The environment is now ready, and you can run the project as intended.

---

## 📤 Exporting the Environment (For Developers)

If you make changes to the environment and want to update the `environment.yml` file:

```bash
conda env export --from-history > environment.yml
```

This ensures only explicitly installed packages are included, keeping the file clean.

---

## 📄 License

Include your project’s license information here.

---

## 🙌 Contributions

Feel free to open issues or pull requests to improve this project!
