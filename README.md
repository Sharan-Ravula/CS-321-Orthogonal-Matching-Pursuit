# Orthogonal Matching Pursuit
---

> Hello Viewer, Hope you are doing good! 

> Please run these commands in terminal/powershell ;)

---

1. Create a Virtual Environment in the path you open the file:
    
	- macOS / Linux:

      ```bash
	  python3 -m venv venv
	  source venv/bin/activate
      
    - Windows:
      
	  ```powershell
	  python -m venv venv
	  .\venv\Scripts\activate

2. Install Homebrew (optional but recommended):
   
    - macOS:
      
      ```bash	
	  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

 	- Windows: On Windows, you can use Chocolatey as a package manager.

	   >  Note: If you prefer to use Homebrew on Windows, consider installing it via WSL (Windows Subsystem for Linux) and following the macOS instructions within your WSL terminal.

	   + Open PowerShell as Administrator.
	   + Run the following command to install Chocolatey:
   
	     ```powershell
		 Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

	   + Close and reopen your PowerShell, then verify Chocolatey is installed:
   
		 ```powershell
		 choco --version

4. Install Dependencies: First, make sure pip is updated:
		
	- macOS/Linux:

	  ```bash	
	  python3 -m pip install --upgrade pip
	  pip --version
	  pip install scikit-learn numpy matplotlib

    - Windows:
      
	  ```powershell
	  python -m pip install --upgrade pip
	  pip --version
	  pip install scikit-learn numpy matplotlib

5. To run the code:
	
	- macOS/Linux:

	  ```bash
	  python3 OMP_Feature-Selection_in_Linear_Regression.py
	  python3 Sparse_signal_recovery_using_OMP.py
	  python3 OMP_using_Scikit-learn.py

    - Windows:
      
	  ```powershell
	  python OMP_Feature-Selection_in_Linear_Regression.py
	  python Sparse_signal_recovery_using_OMP.py
	  python OMP_using_Scikit-learn.py

> To exit from the virtual environment just type `deactivate` in the command shell and press enter
