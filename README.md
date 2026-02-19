# Orthogonal Matching Pursuit (OMP) Project

## 📌 Project Overview

This repository contains the implementation and analysis of the **Orthogonal Matching Pursuit** algorithm, completed as part of the CS/MA 321 Introduction to Numerical Methods final project. OMP is a greedy algorithm designed for finding sparse solutions to underdetermined linear systems where there are more variables than constraints ($`m \ll n`$).

---

## 📂 Repository Structure

```text
CS-321-Orthogonal-Matching-Pursuit/
│
│
├── docs/                               				# Project documentation details
│   ├── cs_ma 321_final.pdf                         	# The original project prompt and technical requirements for the Final Project
│   ├── cs_ma_321_omp.pdf        						# The comprehensive slide deck created by the team, covering OMP theory, least squares connection, and real-world applications
│
├── scripts/                                       		# Python implementations demonstrating different use cases for the OMP algorithm
│   ├── omp_feature-selection_linear_regression.py 		# Demonstrates how OMP can be used as a feature selection tool to identify the most significant variables in a regression model
│   ├── omp_using_scikit_learn.py                  		# A foundational script showing the implementation of the `OrthogonalMatchingPursuit` class on a basic sparse signal
│   ├── sparse_signal_recovery_using_omp.py        		# Illustrates the recovery of a sparse signal from noisy measurements, including visual comparisons between original and recovered data
│
├── output/                                           	# Visual evidence of the algorithm's performance and execution results
│   ├── omp_graph.py      								# Generated plot showing the original sparse signal, the noisy observed signal, and the successfully recovered signal using OMP
│   ├── omp-output.png                           		# Terminal output showing the original signal vs. estimated coefficients and the identified support set
│	├── linear_regression_graph.png                     # Graph illustrates the effectiveness of Orthogonal Matching Pursuit (OMP) as a feature selection tool within a linear regression model
│
├── team_5.png                           				# Team_5 Group Selfie, Unfortunately Mekha was not present
├── README.md                           				# Project overview and instructions
└── LICENSE                             				# License information for the repository
```

---

## 🧠 Algorithm Overview

OMP approximates the solution of $`(P_0)` `:` `\min_x` `\|x\|_0`$ subject to $`Ax` `=` `b`$. It operates by iteratively building up an approximation atom by atom in a greedy fashion.

### The Iterative Process

- Sweep/Selection: Compute the errors for all columns $`j`$ and find a minimizer $`j_0`$ that best correlates with the current residual.
   
- Update Support: Add the index of this column to the support set $`S^k` `=` `S^{k-1}` `\cup` `\{j_0\}`$.
  
- Update Provisional Solution (Least Squares): Compute $`x^k`$ as the minimizer of $`\|Ax` `-` `b\|_2^2`$ subject to the current support.
  
- Update Residual: Compute the new residual $`r^k` `=` `b` `-` `Ax^k`$.
  
- Stopping Rule: Terminate if $`\|r^k\|_2` `<` `\epsilon_0`$ ; otherwise, apply another iteration.

---

## ✨ Analogy

What is OMP? 🤔

Imagine you have a giant box of 1,000 different Lego bricks (the Dictionary or Matrix $`A`$). I show you a finished tower (the Signal or Vector $`b`$) and tell you that this tower was actually built using only 4 specific bricks from that box

### The Iterative Process: How do you find those 4 bricks?

- Search (Sweep): You look through the box and pick the one brick that looks most like a part of the tower
  
- Test (Update Support): You put that brick down and see how much of the tower it explains.
  
- Adjust (Least Squares): You adjust the position of that brick to make sure it fits perfectly.
  
- Repeat: Now you look at the "leftover" part of the tower (the Residual) and go back to the box to find the next best brick.

> You keep doing this until you’ve rebuilt the tower exactly, using only those 4 pieces. OMP is "greedy" because it picks the best-fitting piece at every single step instead of trying to look at all 1,000 pieces at once.

---

## 📐 Mathematical Foundation: The Least Squares Connection

> OMP is fundamentally rooted in the Method of Least Squares to ensure global progress at each greedy step.

### Coefficient Update

Once a new index is added to the support, OMP re-calculates the values for all indices currently in $`S^k`$ by solving a linear least squares problem restricted to the active columns of $`A`$. This is often computed using the Moore-Penrose pseudoinverse:

$`\hat{\theta}_{S}` `=` `(X_{S}^T X_{S})^{-1}` `X_{S}^T` `y`$

## 📐 Mathematical Foundation: OMP and Least Squares

The Orthogonal Matching Pursuit (OMP) algorithm is a greedy approach used to solve the sparse recovery problem $`(P_0)`$:

$`\min_{x}` `\|x\|_0` \text{ `subject to` } `Ax` `=` `b`$

Where $`A` \in `\mathbb{R}^{m \times n}`$ is an underdetermined sensing matrix ($`m \ll n`$). OMP is fundamentally connected to the Method of Least Squares, which it uses iteratively to refine the solution.

---

## 🔁 The Iterative Process

OMP builds a sparse solution by selecting one "atom" (column of $`A`$) at each iteration that best represents the remaining signal (the residual).

### 1. Atom Selection (Matching)

At each iteration $`k`$, we compute the correlation between each column $`a_j`$ of the matrix $`A`$ and the current residual $`r^{k-1}`$. We look for the index $`j_0`$ that minimizes the error $`\epsilon(j)`$:

$`\epsilon(j)` `=` `\min_{z_j}` `\|a_j z_j` `-` `r^{k-1}\|_2^2`$

Using the optimal choice for the coefficient $`z_j^*`$:

$`z_j^*` `=` `\frac{a_j^T r^{k-1}}{\|a_j\|_2^2}`$

### 2. Support Update

The index $`j_0`$ of the best-performing atom is added to the active support set $`S^k`$:

$`S^k` `=` `S^{k-1}` `\cup` `\{j_0\}`$

### 3. Provisional Solution (The Least Squares Step)

Once the support is updated, OMP computes a new provisional solution $`x^k`$. This is defined as the minimizer of the least squares error restricted to the current support:

$`\min_{x}` `\|Ax - b\|_2^2` `\quad` \text{`subject to` } `Support\{x\}` `=` `S^k`$

The solution to this sub-problem is found using the Moore-Penrose pseudoinverse $`A_{S}^+`$ of the matrix $`A`$ restricted to the columns in $`S^k`$:

$`x^k_{S}` `=` `(A_{S}^T` `A_{S})^{-1}` `A_{S}^T b`$

### 4. Residual Update

The residual is updated by subtracting the new approximation from the observed signal:

$`r^k` `=` `b` `-` `Ax^k`$

This ensures the new residual is orthogonal to all atoms currently in the support set, preventing the algorithm from re-selecting the same atoms.

---

## ⚖️ Comparison Table

| Feature | Standard Least Squares | Orthogonal Matching Pursuit |
| :--- | :--- | :--- |
| Objective | Minimize $`\|Ax` `-` `b\|_2^2`$ | Minimize $`\|x\|_0`$ s.t. $`Ax=b`$ |
| System Type | Usually Overdetermined ($`m > n`$) | Underdetermined ($`m \ll n`$) |
| Logic | Global optimization | Greedy, iterative selection |
| Result | Dense solution (mostly non-zeros) | Sparse solution (mostly zeros) |

---

## 🛜 Connection to Project Code

* Signal Recovery: In `Sparse_signal_recovery_using_OMP.py`, we observe the residual decreasing as more atoms are added until it hits the error threshold $`\epsilon_0`$.

### Residual Decorrelation

By solving the least squares problem at every step, OMP ensures that the new residual is orthogonal to all columns currently in the support set. This prevents the algorithm from picking the same column twice and ensures the residual norm decreases efficiently.

---

## 🛠 Installation & Usage

1. Create a Virtual Environment in the path you open the file:
    
	- macOS / Linux:

      ```bash
	  python3 -m venv venv
	  source venv/bin/activate

    > To exit from the virtual environment just type `deactivate` in the command shell and press enter
   
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

	   > Open PowerShell as Administrator.
   
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
	  python3 omp_feature_selection_linear_regression.py
	  python3 sparse_signal_recovery_using_omp.py
	  python3 omp_using_scikit_learn.py

    - Windows:
      
	  ```powershell
	  python omp_feature_selection_linear_regression.py
	  python sparse_signal_recovery_using_omp.py
	  python omp_using_scikit_learn.py

---

## 📊 Real-world Applications

1. Medical Imaging (MRI Scans):
MRI machines take a long time to scan your body. By using OMP, doctors can take a "sparse" or incomplete scan (fewer data points) and use the algorithm to "fill in the blanks" to create a high-quality image. This means you spend less time inside the loud MRI machine.

2. Cleaning Up Photos (Image Denoising):
If you take a photo in the dark, it often looks "grainy" or "noisy". Because noise is random and doesn't follow a pattern, it isn't "sparse". OMP looks at the photo, finds the clear patterns that do belong there, and ignores the random grain, giving you a much cleaner picture.

3. Smart Data Selection (Feature Selection):
If a scientist is trying to figure out why a disease happens, they might have 20,000 different pieces of data about a patient. OMP can act like a filter, picking out the 5 or 10 most important "features" (like specific genes or habits) that actually matter, ignoring the thousands of other distracting details.

4. Zoom and Video Streaming:
When you watch a video online, the app doesn't send you every single pixel for every single frame—that would be too much data. It sends a "sparse" version, and algorithms like OMP help your device reconstruct the full, sharp video on your screen in real-time.

---

## 📚 References

- Code: [geeksforgeeks_omp](https://www.geeksforgeeks.org/data-science/orthogonal-matching-pursuit-omp-using-sklearn/)
  
- Theory: [sciencedirect_omp](https://www.sciencedirect.com/topics/engineering/orthogonal-matching-pursuit)
  
- Book: Sauer - Numerical Analysis 2e
  
- Rubinstein, R., Zibulevsky, M., & Elad, M. (2010). Efficient implementation of the K-SVD algorithm using batch orthogonal matching pursuit. IEEE Transactions on Signal Processing, 57(12), 5636-5646. https://doi.org/10.1109/TSP.2009.2039178

---

## 💥 Contributors:
   
   - Sharan Ravula
     
   - Shane Wojcicki
     
   - Mekha Rajesh
     
   - Fayek Sharaf

---

> I had a lot of fun with the team when we were working on the project and during the presentation (got bonus points for it aswell)

<img width="1715" height="1291" alt="team_5" src="https://github.com/user-attachments/assets/dd96fd46-0d7f-46bb-841b-f85e706c2321" /
