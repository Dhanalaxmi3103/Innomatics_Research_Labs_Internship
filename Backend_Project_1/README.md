#  Regular Expression Matcher

I recently explored **regex101.com** to better understand how regular expressions are tested and visualized.  
As a hands-on challenge, I built a **Flask-based web application** that replicates its **core functionality**.
Users can test a regular expression against a given string and instantly view all matches, their positions, and highlighted results.

This project focuses on **hands-on regular expression(regex) exploration**, backend logic, and dynamic web rendering.


##  Features

-  Test any **regular expression** against a custom input string
-  Display **all matched substrings**
-  Show **start and end index positions** for each match
-  Highlight matches directly inside the test string
-  Graceful handling of **invalid regex patterns**
-  Display total match count
-  Clean, minimal UI inspired by regex playgrounds



##  Tech Stack

- **Python 3**
- **Flask**
- **Regular Expressions (`re` module)**
- HTML (generated dynamically)


##  How to Run the Application

## 1.Install Dependencies
```bash
pip install flask
```
## 2.Start the Flask Server
```
python regex.py
```
## 3. Open in Browser

http://127.0.0.1:5000

## How It Works

--> User input is collected via an HTML form

--> Regex is compiled using Python’s re module

--> re.finditer() is used to retrieve:

   --> Matched text

   --> Start index

   --> End index

--> Matches are dynamically highlighted and rendered in the browser




