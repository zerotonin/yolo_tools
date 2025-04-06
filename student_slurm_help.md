# Getting Started with Python Modules and Scripts

This guide explains why and how we run our Python scripts as modules using the `-m` flag, which is especially important for larger projects with a structured directory layout.

## Why Use `python -m`?

- **Package Awareness:**  
  When you run a script with `python -m <module>`, Python treats the script as part of a package. This ensures that any relative imports within your modules work correctly. For example, if your script is part of the `scripts` package, running it as a module lets Python find and load other modules (or packages) that are in your project.

- **Consistent Import Paths:**  
  Using the `-m` flag requires you to specify the module name in dot notation (e.g., `scripts.start_lyall`). This avoids issues that can occur when running a script as a standalone file (e.g., incorrect module search paths or missing `__init__.py` markers).

- **Project Organization:**  
  Projects with multiple directories (like `config`, `database`, `yolo_tools`, and `scripts`) benefit from running scripts as modules. It enforces a clear, organized structure where every folder with an `__init__.py` file is recognized as a Python package.

## How to Run Scripts as Modules

1. **Ensure Directory Structure is Correct:**
   - Every folder meant to be a package should contain an `__init__.py` file.
   - For example, if you want to run a script located in the `scripts` folder, make sure there is an `__init__.py` file inside `scripts`.

2. **Use Dot Notation Instead of File Paths:**
   - **Incorrect:**  
     ```bash
     python -m scripts/start_lyall
     ```
   - **Correct:**  
     ```bash
     python -m scripts.start_lyall
     ```
     The dot `.` replaces the slash `/` to denote the module hierarchy.

3. **Set the Working Directory Appropriately:**
   - Run the command from the project's root directory so that Python can correctly locate all the packages and modules.

## Benefits for Our Project

- **Reliable Module Imports:**  
  When you use `-m`, Python sets up the module search path correctly. This helps avoid common errors like "No module named ..." when using relative imports.

- **Easier Collaboration and Deployment:**  
  Having a clear module structure makes it easier for others (and for automated systems like Slurm) to run your code reliably without needing to adjust paths or modify the script.

- **Better Integration with Tools:**  
  Tools that manage job submissions (like Slurm) or environments (like conda) work more predictably when your scripts are run as part of a structured package.

## Summary Checklist

- [ ] **Make a GitHub Account:**  
  Sign up at [GitHub](https://github.com/) to collaborate on code.
- [ ] **Get Invited to the Repository:**  
  Accept the invitation to join the project repository.
- [ ] **Create and Add an SSH Key:**  
  Secure your connection to the repository.
- [ ] **Add the Aoraki Cluster:**  
  Follow the instructions to connect to the cluster.
- [ ] **Install Spack:**  
  Instructions available [here](https://rtis.cspages.otago.ac.nz/research-computing/cluster/software/spack.html).
- [ ] **Install Miniconda:**  
  Set up your Python environment with Miniconda.
- [ ] **Run Scripts Correctly:**  
  Always use dot notation with `python -m`, e.g.,  
  ```bash
  python -m scripts.start_lyall
